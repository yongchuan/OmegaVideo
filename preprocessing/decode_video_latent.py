import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusers import AutoencoderKLLTXVideo

try:
    from torchvision.io import write_video
except Exception:
    write_video = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import av
except Exception:
    av = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a saved video latent .npy file back into an MP4 video."
    )
    parser.add_argument("--latent", type=str, required=True, help="Path to a latent .npy file.")
    parser.add_argument("--output", type=str, required=True, help="Output .mp4 path.")
    parser.add_argument(
        "--vae-path",
        type=str,
        required=True,
        help="Path to an AutoencoderKLLTXVideo pretrained directory or model id.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device used for video VAE decode.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype used by the video VAE.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. If omitted, tries to read it from --manifest, otherwise defaults to 24.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional path to video_manifest.json for fps/crop metadata.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Optional explicit manifest sample id. Defaults to latent relative path stem.",
    )
    parser.add_argument(
        "--normalize-latents",
        action="store_true",
        help="Compatibility flag for the old decoder path. Ignored by AutoencoderKLLTXVideo.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def load_latent(latent_path: Path) -> torch.Tensor:
    latent = np.load(latent_path)
    latent = np.ascontiguousarray(latent)
    if latent.ndim not in (4, 5):
        raise ValueError(
            f"Expected latent array with shape [C, T, H, W] or [B, C, T, H, W], got {tuple(latent.shape)}"
        )
    return torch.from_numpy(latent)


def ensure_batched_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 4:
        return latent.unsqueeze(0)
    if latent.ndim == 5:
        return latent
    raise ValueError(f"Expected latent tensor rank 4 or 5, got {latent.ndim}")


def ensure_unbatched_video(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected decoded batch size 1, got {video.shape[0]}")
        return video[0]
    if video.ndim == 4:
        return video
    raise ValueError(f"Expected decoded video tensor rank 4 or 5, got {video.ndim}")


def build_manifest_index(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, list):
        raise ValueError(f"Expected manifest list in {manifest_path}")
    index = {}
    for item in manifest:
        sample_id = item.get("id")
        if sample_id is not None:
            index[str(sample_id)] = item
    return index


def resolve_sample_id(latent_path: Path, sample_id_arg: Optional[str], manifest_path: Optional[Path]) -> Optional[str]:
    if sample_id_arg:
        return sample_id_arg
    if manifest_path is None:
        return None

    latent_parent = latent_path.parent
    manifest_root = manifest_path.parent
    try:
        # common case: manifest in dataset root and latent under dataset_root/vae-in/...
        if latent_parent.parent == manifest_root and latent_parent.name == "vae-in":
            return latent_path.relative_to(latent_parent).with_suffix("").as_posix()
    except Exception:
        pass

    try:
        vae_dir = manifest_root / "vae-in"
        return latent_path.relative_to(vae_dir).with_suffix("").as_posix()
    except Exception:
        return latent_path.stem


def crop_video(video: torch.Tensor, crop_shape: Tuple[int, int, int]) -> torch.Tensor:
    frame_count, height, width = crop_shape
    if video.shape[1] < frame_count or video.shape[2] < height or video.shape[3] < width:
        raise ValueError(
            f"Decoded video shape {tuple(video.shape)} is smaller than crop shape "
            f"(C, {frame_count}, {height}, {width})"
        )
    return video[:, :frame_count, :height, :width].contiguous()


def to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    return ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def save_video_with_torchvision(video: torch.Tensor, output_path: Path, fps: float) -> None:
    if write_video is None:
        raise RuntimeError("torchvision.io.write_video is unavailable.")

    frames = video.permute(1, 2, 3, 0).contiguous().cpu()
    write_video(
        str(output_path),
        frames,
        fps=float(fps),
        video_codec="h264",
        options={"crf": "18"},
    )


def save_video_with_pyav(video: torch.Tensor, output_path: Path, fps: float) -> None:
    if av is None:
        raise RuntimeError("PyAV is unavailable.")

    frames = video.permute(1, 2, 3, 0).contiguous().cpu()
    container = av.open(str(output_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(1000))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"

        for frame_tensor in frames:
            frame = av.VideoFrame.from_ndarray(frame_tensor.numpy(), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def save_video_with_cv2(video: torch.Tensor, output_path: Path, fps: float) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable.")

    frames = video.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open OpenCV VideoWriter for: {output_path}")

    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_video(video: torch.Tensor, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    for saver in (save_video_with_torchvision, save_video_with_cv2, save_video_with_pyav):
        try:
            saver(video, output_path, fps)
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to save video. Install OpenCV, PyAV, or a torchvision video backend."
    ) from last_error


def main() -> None:
    args = parse_args()

    if not args.device.startswith("cuda"):
        raise ValueError("Video latent decoding in this project requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required, but no CUDA device is available.")

    latent_path = Path(args.latent)
    output_path = Path(args.output)
    vae_path = Path(args.vae_path)
    manifest_path = Path(args.manifest) if args.manifest else None

    if not latent_path.exists():
        raise FileNotFoundError(f"Latent file not found: {latent_path}")
    if manifest_path is not None and not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest_item = None
    sample_id = None
    if manifest_path is not None:
        manifest_index = build_manifest_index(manifest_path)
        sample_id = resolve_sample_id(latent_path, args.sample_id, manifest_path)
        if sample_id not in manifest_index:
            raise KeyError(f"Sample id not found in manifest: {sample_id}")
        manifest_item = manifest_index[sample_id]

    fps = float(args.fps if args.fps is not None else (manifest_item.get("fps", 24.0) if manifest_item else 24.0))
    crop_shape = None
    if manifest_item is not None and manifest_item.get("crop_shape") is not None:
        crop_shape = tuple(int(x) for x in manifest_item["crop_shape"])

    latent = load_latent(latent_path)
    dtype = resolve_dtype(args.dtype)
    vae = AutoencoderKLLTXVideo.from_pretrained(str(vae_path)).to(args.device).eval()

    latent = ensure_batched_latent(latent).to(device=args.device, dtype=dtype)
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            decoded = vae.decode(latent).sample
        decoded = ensure_unbatched_video(decoded).float().cpu()

    if crop_shape is not None:
        decoded = crop_video(decoded, crop_shape)

    video = to_uint8_video(decoded)
    save_video(video, output_path, fps=fps)

    print(f"latent: {latent_path}")
    if sample_id is not None:
        print(f"sample_id: {sample_id}")
    print(f"decoded_shape: {tuple(video.shape)}")
    print(f"fps: {fps}")
    print(f"saved_to: {output_path}")


if __name__ == "__main__":
    main()

