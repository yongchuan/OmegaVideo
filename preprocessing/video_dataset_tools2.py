import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#from models.video_vae import VideoVAE
from diffusers import AutoencoderKLLTXVideo

try:
    from torchvision.io import read_video
except Exception:
    read_video = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import av
except Exception:
    av = None


TEMPORAL_STRIDE = 4
SPATIAL_STRIDE = 8
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
DEFAULT_TARGET_HEIGHT = 352
DEFAULT_TARGET_WIDTH = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a video dataset and encode each clip into VideoVAE latents."
    )
    parser.add_argument("--source", type=str, required=True, help="Directory containing source videos.")
    parser.add_argument("--dest", type=str, required=True, help="Output dataset root directory.")
    parser.add_argument("--vae-path", type=str, required=True, help="Path to VideoVAE weights.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device used for VideoVAE inference.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype used by VideoVAE.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help="Target frame height after resize.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=DEFAULT_TARGET_WIDTH,
        help="Target frame width after resize.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=17,
        help="Only keep the first N frames from each source video before VAE encoding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Compatibility flag. Sequential mode always encodes one video at a time.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Compatibility flag. Sequential mode does not use worker threads.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on how many input videos to process.",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=24.0,
        help="FPS used when the input video metadata is missing.",
    )
    parser.add_argument(
        "--normalize-latents",
        action="store_true",
        help="Enable latent normalization inside VideoVAE.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode samples even if the processed clip and latent already exist.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def next_compatible_frame_count(frame_count: int) -> int:
    remainder = (frame_count - 1) % TEMPORAL_STRIDE
    if remainder == 0:
        return frame_count
    return frame_count + (TEMPORAL_STRIDE - remainder)


def collect_video_files(source_dir: Path, max_videos: Optional[int]) -> List[Path]:
    video_files = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in DEFAULT_VIDEO_EXTENSIONS:
            video_files.append(path)
            if max_videos is not None and len(video_files) >= max_videos:
                break
    return video_files


def load_video_with_torchvision(video_path: Path, fallback_fps: float) -> Tuple[torch.Tensor, float]:
    if read_video is None:
        raise RuntimeError("torchvision.io.read_video is unavailable.")

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"Input video has no frames: {video_path}")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T, H, W, C] frames, got {tuple(frames.shape)}")

    fps = float(info.get("video_fps") or fallback_fps)
    return frames.permute(3, 0, 1, 2).contiguous(), fps


def load_video_with_pyav(video_path: Path, fallback_fps: float) -> Tuple[torch.Tensor, float]:
    if av is None:
        raise RuntimeError("PyAV is unavailable.")

    container = av.open(str(video_path))
    try:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in: {video_path}")

        fps_value = video_stream.average_rate
        fps = float(fps_value) if fps_value is not None else fallback_fps
        frames = []
        for frame in container.decode(video=video_stream.index):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
        if not frames:
            raise ValueError(f"Input video has no decodable frames: {video_path}")

        video = torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous()
        return video, fps
    finally:
        container.close()


def load_video_with_cv2(video_path: Path, fallback_fps: float) -> Tuple[torch.Tensor, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video with OpenCV: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps or fps <= 0:
            fps = fallback_fps

        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))

        if not frames:
            raise ValueError(f"Input video has no decodable frames: {video_path}")

        video = torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous()
        return video, fps
    finally:
        capture.release()


def load_video(video_path: Path, fallback_fps: float) -> Tuple[torch.Tensor, float]:
    last_error = None
    for loader in (load_video_with_torchvision, load_video_with_cv2, load_video_with_pyav):
        try:
            return loader(video_path, fallback_fps)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Unable to read video. Install OpenCV, PyAV, or a torchvision video backend."
    ) from last_error


def resize_video(video: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    frames = video.permute(1, 0, 2, 3).float()
    frames = F.interpolate(
        frames,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return frames.permute(1, 0, 2, 3).contiguous()


def pad_video_for_vae(video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _, frame_count, height, width = video.shape
    target_frames = next_compatible_frame_count(frame_count)
    target_height = round_up(height, SPATIAL_STRIDE)
    target_width = round_up(width, SPATIAL_STRIDE)

    pad_t = target_frames - frame_count
    pad_h = target_height - height
    pad_w = target_width - width
    if pad_t or pad_h or pad_w:
        video = F.pad(video.unsqueeze(0), (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate").squeeze(0)
    return video, (frame_count, height, width)


def preprocess_video(
    video: torch.Tensor,
    target_height: int,
    target_width: int,
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    original_shape = list(video.shape)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"--max-frames must be positive, got {max_frames}")
        video = video[:, :max_frames]
        if video.shape[1] == 0:
            raise ValueError("Input video has no frames after applying --max-frames.")

    resized = resize_video(video, target_height=target_height, target_width=target_width)
    processed_uint8 = resized.clamp(0, 255).round().to(torch.uint8)
    normalized = processed_uint8.float().div(127.5).sub(1.0)
    padded, crop_shape = pad_video_for_vae(normalized)
    metadata = {
        "source_shape": original_shape,
        "selected_frame_count": int(processed_uint8.shape[1]),
        "processed_shape": list(processed_uint8.shape),
        "vae_input_shape": list(padded.shape),
        "crop_shape": list(crop_shape),
    }
    return processed_uint8, padded, metadata


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.save(f, array)


def encode_and_save_video(
    vae,
    processed_video: torch.Tensor,
    vae_input: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    output_relative_npy: Path,
    videos_dir: Path,
    latents_dir: Path,
) -> None:
    #print("vae_input:", vae_input.shape)
    vae_video = vae_input.unsqueeze(0)
    #print("vae_video:", vae_video.shape)
    vae_video = vae_video.to(device=device, dtype=dtype)

    with torch.inference_mode():
        with torch.no_grad():
            with torch.amp.autocast(
                    "cuda",
                    enabled=(True),
            ):
                print("vae_video:", vae_video.shape)
                latent = vae.encode(vae_video).latent_dist.sample()
                print("latent:", latent.shape)
                processed_video_array = processed_video.cpu().numpy()
                latent_array = latent.detach().cpu().numpy()
                save_numpy(videos_dir / output_relative_npy, processed_video_array)
                save_numpy(latents_dir / output_relative_npy, latent_array)


def main() -> None:
    args = parse_args()
    device="cuda"
    if not args.device.startswith("cuda"):
        raise ValueError("VideoVAE preprocessing requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required, but no CUDA device is available.")

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)
    vae_path = Path(args.vae_path)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE weights not found: {vae_path}")

    video_files = collect_video_files(source_dir, max_videos=args.max_videos)
    if not video_files:
        raise FileNotFoundError(f"No supported video files found under: {source_dir}")

    videos_dir = dest_dir / "videos"
    latents_dir = dest_dir / "vae-in"
    videos_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype)
    vae = AutoencoderKLLTXVideo.from_pretrained("/root/gpufree-data/models/ltx_vae/").to(device).eval()

    manifest = []

    for video_path in tqdm(video_files, desc="Encoding videos with VideoVAE"):
        relative_path = video_path.relative_to(source_dir)
        relative_npy = Path(relative_path).with_suffix(".npy")
        raw_output_path = videos_dir / relative_npy
        latent_output_path = latents_dir / relative_npy

        if (
            not args.overwrite
            and raw_output_path.exists()
            and latent_output_path.exists()
        ):
            manifest.append(
                {
                    "id": relative_npy.with_suffix("").as_posix(),
                    "source": relative_path.as_posix(),
                    "processed_video": relative_npy.as_posix(),
                    "latent": relative_npy.as_posix(),
                    "skipped": True,
                }
            )
            continue

        video, fps = load_video(video_path, fallback_fps=args.fallback_fps)
        processed_video, vae_input, metadata = preprocess_video(
            video,
            target_height=args.target_height,
            target_width=args.target_width,
            max_frames=args.max_frames,
        )
        encode_and_save_video(
            vae=vae,
            processed_video=processed_video,
            vae_input=vae_input,
            device=args.device,
            dtype=dtype,
            output_relative_npy=relative_npy,
            videos_dir=videos_dir,
            latents_dir=latents_dir,
        )

        manifest.append(
            {
                "id": relative_npy.with_suffix("").as_posix(),
                "source": relative_path.as_posix(),
                "processed_video": relative_npy.as_posix(),
                "latent": relative_npy.as_posix(),
                "fps": fps,
                **metadata,
            }
        )
    with (latents_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": None}, f, indent=2, ensure_ascii=False)

    with (dest_dir / "video_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with (dest_dir / "preprocess_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"Processed {len(video_files)} videos into {dest_dir}")


if __name__ == "__main__":
    main()

