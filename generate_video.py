# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.sit_video import SiT_models
from preprocessing.encoders import load_invae
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_maruyama_sampler, euler_maruyama_sampler_path_drop
from utils import download_model
from typing import List, Optional, Union
from modelscope import CLIPModel, CLIPProcessor, CLIPConfig
from preprocessing.encoders import load_invae, load_vavae
from pathlib import Path
from typing import Tuple
from diffusers import AutoencoderKLLTXVideo

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

def get_clip_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
):
    """Get CLIP text embeddings for prompts.

    Reference: img_label_dataset.py get_clip_prompt_embeds()
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    #print(text_inputs)

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    # Use pooled output of CLIPTextModel
    # prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(device=device)

    return prompt_embeds

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

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def crop_video(video: torch.Tensor, original_shape: Tuple[int, int, int]) -> torch.Tensor:
    frame_count, height, width = original_shape
    if video.shape[1] < frame_count or video.shape[2] < height or video.shape[3] < width:
        raise ValueError(
            f"Decoded video shape {tuple(video.shape)} is smaller than requested crop "
            f"(C, T, H, W) -> ({video.shape[0]}, {frame_count}, {height}, {width})"
        )
    return video[:, :frame_count, :height, :width].contiguous()

def get_latent_stats(features_dir):
        latent_stats_cache_file = os.path.join(features_dir, "latents_stats.pt")
        latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']

def to_uint8_video(video: torch.Tensor) -> torch.Tensor:
    video = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return video

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 16  # invae uses 16x downsampling
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=128,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if ckpt_path is None:
        args.ckpt = download_model(repo_id="SwayStar123/SpeedrunDiT", filename="256/0400000.pt")
        assert args.model == 'SiT-B/1'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        assert args.qk_norm is True
        assert args.resolution == 256
        assert args.mode == "sde"
        ckpt = torch.load(args.ckpt, map_location=f'cuda:{device}', weights_only=False)
        #print(ckpt)
        state_dict = ckpt['ema'] if isinstance(ckpt, dict) and 'ema' in ckpt else ckpt
    else:
        print("Loading checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=f'cuda:{device}', weights_only=False)
        #print(ckpt['model'])
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    model.load_state_dict(state_dict)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    model.eval()  # important!
    # Load invae model using load_invae function
    # if rank == 0:
    #     _ = load_invae("REPA-E/e2e-invae", device=torch.device("cpu"))
    # dist.barrier()
    # vae = load_invae("REPA-E/e2e-invae", device=torch.device(f"cuda:{device}"))
    # vae = load_vavae("tokenizer/configs/vavae_f16d32_vfdinov2.yaml", device=device)
    # vae.model.requires_grad_(False)
    # channels = 32  # invae uses 32 channels
    vae = AutoencoderKLLTXVideo.from_pretrained("/root/gpufree-data/models/ltx_vae/").to(device).eval()
    scaling_factor = 0.41407

    latent_mean, latent_std = get_latent_stats("/root/gpufree-data/OmegaDiT-master/videos_test/video_latents/")
    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-invae-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-{args.guidance_high}-{args.cls_cfg_scale}-pathdrop-{args.path_drop}"
    if args.balanced_sampling:
        folder_name += "-balanced"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

        # Check for existing PNG samples to optionally skip resampling.
        existing_pngs = [f for f in os.listdir(sample_folder_dir) if f.endswith(".png")]
        existing_count = len(existing_pngs)
        if existing_count >= args.num_fid_samples:
            print(
                f"Found {existing_count} existing PNG samples in {sample_folder_dir}, "
                f"skipping sampling and only rebuilding the .npz file."
            )
            need_sampling = False
        else:
            need_sampling = True
    else:
        need_sampling = True

    # Broadcast need_sampling decision from rank 0 to all ranks.
    need_sampling_tensor = torch.tensor(int(need_sampling), device=device)
    dist.broadcast(need_sampling_tensor, src=0)
    need_sampling = bool(need_sampling_tensor.item())
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0 and need_sampling:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        #print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    model_id = "AI-ModelScope/CLIP-GmP-ViT-L-14"
    clip_config = CLIPConfig.from_pretrained(model_id)
    max_tokens = 77
    clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32, config=clip_config).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_id, padding="max_length", max_length=max_tokens,
                                                   return_tensors="pt", truncation=True)

    if need_sampling:
        pbar = range(iterations)
        pbar = tqdm(pbar) if rank == 0 else pbar
        total = 0

        labels = [None] * n
        labels[0] = "a girl"
        labels[1] = "The video features a man with a beard and short hair, wearing a brown shirt. "
        labels[2] = "a woman with short brown hair is seen sitting in the backseat of a car. "
        labels[3] = "The video features a young woman with dark hair and red lipstick, looking directly at the camera."

        y_null = model.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None]
        y_null = y_null.reshape(n, 77, 768)

        for idx in pbar:
            # Sample inputs:
            z = torch.randn(n, model.in_channels, 3, 11, 20, device=device)
            y = get_clip_prompt_embeds(clip_processor.tokenizer, clip_model.text_model, labels, device=device)

            # Sample images:
            sampling_kwargs = dict(
                model=model,
                latents=z,
                y=y,
                y_null=y_null,
                num_steps=args.num_steps,
                heun=args.heun,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
                path_type=args.path_type,
                cls_latents=None,
                args=args
            )
            with torch.no_grad():
                if args.mode == "sde":
                    if args.path_drop:
                        samples = euler_maruyama_sampler_path_drop(**sampling_kwargs).to(torch.float32)
                    else:
                        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
                elif args.mode == "ode":# will support
                    exit()
                    #samples = euler_sampler(**sampling_kwargs).to(torch.float32)
                else:
                    raise NotImplementedError()

                with torch.amp.autocast(
                        "cuda",
                        enabled=(True),
                ):
                    # For invae, apply 0.3099 scaling factor
                    # samples = vae.decode(samples / scaling_factor).sample
                    latent = (samples * latent_std) + latent_mean
                    #latent = (samples / scaling_factor)
                    reconstructed = vae.decode(latent).sample
	
            original_shape = (17, 352, 640)

            for i in range(reconstructed.shape[0]):
                output_path = "results/"+str(idx)+"-"+str(i)+".mp4"
                output_path = Path(output_path)
                once = reconstructed[i]
                decoded_video = crop_video(once.cpu(), original_shape)
                output_video = to_uint8_video(decoded_video)
                print(f"Output video shape after crop: {tuple(output_video.shape)}")

                save_video(output_video, output_path, fps=30)
                print(f"Saved reconstructed video to: {output_path}")
            total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/1")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent-stats-dir", type=str, default="samples")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=10)
    parser.add_argument("--num-fid-samples", type=int, default=100)

    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="If enabled, sample class labels in a balanced way so each class index appears equally often.")

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--cfg-scale",  type=float, default=2.5)
    parser.add_argument("--cls-cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--cls', default=768, type=int)
    parser.add_argument('--path-drop', default=True, action=argparse.BooleanOptionalAction,)

    parser.add_argument("--time-shifting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shift-base", type=int, default=4096)


    args = parser.parse_args()
    main(args)

