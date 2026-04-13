import argparse
import gc
import math
from fractions import Fraction
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from linum_v2.models.vae import VideoVAE

try:
    from torchvision.io import read_video, write_video
except Exception:
    read_video = None
    write_video = None

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
TARGET_HEIGHT = 256
TARGET_WIDTH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a video, run VAE encode/decode, and save the reconstructed video."
    )
    parser.add_argument("--input", type=str, required=True, help="Input video path.")
    parser.add_argument("--output", type=str, required=True, help="Output video path.")
    parser.add_argument("--vae-path", type=str, required=True, help="VAE weights path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for VAE inference. Default: cuda",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Autocast dtype used by the VAE wrapper. Default: bfloat16",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=24.0,
        help="FPS used when the input video metadata is missing. Default: 24",
    )
    parser.add_argument(
        "--normalize-latents",
        action="store_true",
        help="Enable latent normalization in VideoVAE. Default: disabled",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def next_compatible_frame_count(frame_count: int) -> int:
    remainder = (frame_count - 1) % TEMPORAL_STRIDE
    if remainder == 0:
        return frame_count
    return frame_count + (TEMPORAL_STRIDE - remainder)


def load_video_with_torchvision(video_path: Path, fallback_fps: float) -> Tuple[torch.Tensor, float]:
    if read_video is None:
        raise RuntimeError("torchvision.io.read_video is unavailable.")

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"Input video has no frames: {video_path}")

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video frames in [T, H, W, C] RGB format, got {tuple(frames.shape)}")

    fps = float(info.get("video_fps") or fallback_fps)
    video = frames.permute(3, 0, 1, 2).contiguous()
    return video, fps


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

    if read_video is not None:
        try:
            return load_video_with_torchvision(video_path, fallback_fps)
        except Exception as exc:
            last_error = exc

    if cv2 is not None:
        try:
            return load_video_with_cv2(video_path, fallback_fps)
        except Exception as exc:
            last_error = exc

    if av is not None:
        try:
            return load_video_with_pyav(video_path, fallback_fps)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to read video. Install OpenCV, PyAV, or a torchvision video backend."
    ) from last_error


def center_crop_resize_video(
        video: torch.Tensor,
        target_height: int = TARGET_HEIGHT,
        target_width: int = TARGET_WIDTH) -> torch.Tensor:
    _, _, height, width = video.shape
    scale = max(target_height / height, target_width / width)
    resized_height = max(target_height, math.ceil(height * scale))
    resized_width = max(target_width, math.ceil(width * scale))

    frames = video.permute(1, 0, 2, 3).float()
    frames = F.interpolate(
        frames,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    frames = frames[:, :, top:top + target_height, left:left + target_width]
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


def crop_video(video: torch.Tensor, original_shape: Tuple[int, int, int]) -> torch.Tensor:
    frame_count, height, width = original_shape
    if video.shape[1] < frame_count or video.shape[2] < height or video.shape[3] < width:
        raise ValueError(
            f"Decoded video shape {tuple(video.shape)} is smaller than requested crop "
            f"(C, T, H, W) -> ({video.shape[0]}, {frame_count}, {height}, {width})"
        )
    return video[:, :frame_count, :height, :width].contiguous()


def save_video_with_torchvision(video: torch.Tensor, output_path: Path, fps: float) -> None:
    if write_video is None:
        raise RuntimeError("torchvision.io.write_video is unavailable.")

    frames = video.permute(1, 2, 3, 0).contiguous().cpu()
    write_video(
        str(output_path),
        frames,
        fps=fps,
        video_codec="h264",
        options={"crf": "18"},
    )


def save_video_with_pyav(video: torch.Tensor, output_path: Path, fps: float) -> None:
    if av is None:
        raise RuntimeError("PyAV is unavailable.")

    frame_rate = Fraction(str(fps)).limit_denominator(1000)
    frames = video.permute(1, 2, 3, 0).contiguous().cpu()

    container = av.open(str(output_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=frame_rate)
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Unable to open OpenCV VideoWriter for: {output_path}")

    try:
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
    finally:
        writer.release()


def save_video(video: torch.Tensor, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    if write_video is not None:
        try:
            save_video_with_torchvision(video, output_path, fps)
            return
        except Exception as exc:
            last_error = exc

    if cv2 is not None:
        try:
            save_video_with_cv2(video, output_path, fps)
            return
        except Exception as exc:
            last_error = exc

    if av is not None:
        try:
            save_video_with_pyav(video, output_path, fps)
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to save video. Install OpenCV, PyAV, or a torchvision video backend."
    ) from last_error


def preprocess_video(video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    video = center_crop_resize_video(video)
    video = video.float().div(127.5).sub(1.0)
    return pad_video_for_vae(video)


def postprocess_video(video: torch.Tensor, original_shape: Tuple[int, int, int]) -> torch.Tensor:
    video = crop_video(video, original_shape)
    video = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return video


def main() -> None:
    args = parse_args()

    if not args.device.startswith("cuda"):
        raise ValueError(
            "VideoVAE.encode/decode in this project uses CUDA autocast internally. "
            "Please run with a CUDA device."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required, but no CUDA device is available.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    vae_path = Path(args.vae_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE weights not found: {vae_path}")

    dtype = resolve_dtype(args.dtype)
    print(f"Loading video from: {input_path}")
    input_video, fps = load_video(input_path, fallback_fps=args.fallback_fps)
    print(f"Input video shape (C, T, H, W): {tuple(input_video.shape)}, fps={fps:.3f}")

    normalized_video, original_shape = preprocess_video(input_video)
    print(f"Video shape after crop-resize: {original_shape}")
    print(f"VAE input shape after padding: {tuple(normalized_video.shape)}")

    vae = VideoVAE(
        vae_pth=str(vae_path),
        device=args.device,
        dtype=dtype,
        normalize_latents=args.normalize_latents,
    )

    video_for_vae = normalized_video.to(device=args.device, dtype=dtype)

    with torch.inference_mode():
        latents = vae.encode([video_for_vae])[0]
        print(f"Latent shape (C, T, H, W): {tuple(latents.shape)}")

        del video_for_vae
        gc.collect()
        torch.cuda.empty_cache()

        reconstructed = vae.decode([latents])[0]
        print(f"Decoded shape before crop: {tuple(reconstructed.shape)}")

    output_video = postprocess_video(reconstructed.cpu(), original_shape)
    print(f"Output video shape after crop: {tuple(output_video.shape)}")

    save_video(output_video, output_path, fps=fps)
    print(f"Saved reconstructed video to: {output_path}")


if __name__ == "__main__":
    main()
