import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class VideoLatentDataset(Dataset):
    """Dataset for preprocessed videos and VideoVAE latents.

    Expected directory layout under ``data_dir``:

    - ``videos/``: preprocessed raw clips stored as ``.npy`` in ``[C, T, H, W]`` format.
    - ``vae-in/``: corresponding latent clips stored as ``.npy`` in ``[C, T, H, W]`` or
      ``[1, C, T, H, W]`` format.
    This dataset is caption-only. ``label_file`` is required and should match the
    ``JsonLabelDataset`` JSON format:

    [
      {"id": "subdir/sample_name", "en": "caption text"}
    ]
    """

    def __init__(
        self,
        data_dir: str,
        label_file: Optional[str] = None,
        videos_subdir: str = "videos",
        latents_subdir: str = "vae-in",
    ):
        if label_file is None:
            raise ValueError("VideoLatentDataset requires --label-file with text captions.")

        self.data_dir = data_dir
        self.videos_dir = os.path.join(data_dir, videos_subdir)
        self.features_dir = os.path.join(data_dir, latents_subdir)

        if not os.path.isdir(self.videos_dir):
            raise FileNotFoundError(f"Video directory not found: {self.videos_dir}")
        if not os.path.isdir(self.features_dir):
            raise FileNotFoundError(f"Latent directory not found: {self.features_dir}")

        self.video_fnames = self._collect_npy_files(self.videos_dir)
        self.feature_fnames = self._collect_npy_files(self.features_dir)
        if not self.video_fnames:
            raise FileNotFoundError(f"No .npy video clips found under: {self.videos_dir}")
        if not self.feature_fnames:
            raise FileNotFoundError(f"No .npy latent clips found under: {self.features_dir}")

        self.video_by_id = {self._sample_id(fname): fname for fname in self.video_fnames}
        self.feature_by_id = {self._sample_id(fname): fname for fname in self.feature_fnames}

        missing_videos = sorted(set(self.feature_by_id) - set(self.video_by_id))
        missing_latents = sorted(set(self.video_by_id) - set(self.feature_by_id))
        if missing_videos or missing_latents:
            problems: List[str] = []
            if missing_videos:
                problems.append(f"missing videos for {len(missing_videos)} latent files")
            if missing_latents:
                problems.append(f"missing latents for {len(missing_latents)} video files")
            raise ValueError(f"Video dataset is incomplete: {', '.join(problems)}")

        self.sample_ids = sorted(self.feature_by_id.keys())
        caption_map = self._load_caption_map(label_file)
        self.labels = [caption_map[sample_id] for sample_id in self.sample_ids]
        
        # if latent_norm:
        self._latent_mean, self._latent_std = self.get_latent_stats()

    def _collect_npy_files(self, root_dir: str) -> List[str]:
        files = []
        for root, _dirs, fnames in os.walk(root_dir):
            for fname in fnames:
                if fname.lower().endswith(".npy"):
                    files.append(os.path.relpath(os.path.join(root, fname), root_dir))
        return sorted(files)

    def _sample_id(self, relative_fname: str) -> str:
        return str(Path(relative_fname).with_suffix("")).replace("\\", "/")

    def _load_caption_map(self, label_file: str) -> Dict[str, str]:
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        with open(label_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            caption_map = {item["id"]: item["en"] for item in data}
        elif isinstance(data, dict):
            caption_map = {str(key): str(value) for key, value in data.items()}
        else:
            raise ValueError(f"Unsupported label file format: {label_file}")

        missing = [sample_id for sample_id in self.sample_ids if sample_id not in caption_map]
        if missing:
            raise KeyError(f"Caption missing for sample id: {missing[0]}")
        return caption_map

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_video(self, path: str) -> torch.Tensor:
        video = np.load(path)
        video = np.ascontiguousarray(video)

        if video.ndim == 5 and video.shape[0] == 1:
            video = video[0]
        elif video.ndim == 4 and video.shape[-1] in (1, 3):
            video = np.transpose(video, (3, 0, 1, 2))

        if video.ndim != 4:
            raise ValueError(f"Expected raw video array with 4 dims, got {video.shape} from {path}")

        return torch.from_numpy(video)

    def _load_latent(self, path: str) -> torch.Tensor:
        latent = np.load(path)
        latent = np.ascontiguousarray(latent)
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 4:
            raise ValueError(f"Expected latent array with 4 dims, got {latent.shape} from {path}")
        return torch.from_numpy(latent)

    def compute_latent_stats(self):
        num_samples = min(10000, 10001)
        random_indices = np.random.choice(len(self.sample_ids), num_samples, replace=False)
        latents = []
        for idx in tqdm(random_indices):
            sample_id = self.sample_ids[idx]
            feature_fname = self.feature_by_id[sample_id]
            features = self._load_latent(os.path.join(self.features_dir, feature_fname))
            features = features.unsqueeze(0)
            #features = torch.from_numpy(features)
            latents.append(features)
        latents = torch.cat(latents, dim=0)
        mean = latents.mean(dim=[0, 2, 3, 4], keepdim=True)
        std = latents.std(dim=[0, 2, 3, 4], keepdim=True)
        latent_stats = {'mean': mean.squeeze(0), 'std': std.squeeze(0)}
        print(latent_stats)
        return latent_stats
    
    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        video_fname = self.video_by_id[sample_id]
        feature_fname = self.feature_by_id[sample_id]

        video = self._load_video(os.path.join(self.videos_dir, video_fname))
        latents = self._load_latent(os.path.join(self.features_dir, feature_fname))
        latents = (latents - self._latent_mean) / self._latent_std
        label = self.labels[idx]
        return video, latents, label


def _pad_video_tensor(x: torch.Tensor, target_t: int, target_h: int, target_w: int) -> torch.Tensor:
    pad_t = target_t - x.shape[1]
    pad_h = target_h - x.shape[2]
    pad_w = target_w - x.shape[3]
    if pad_t < 0 or pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"Cannot pad tensor from {tuple(x.shape)} to "
            f"(C, {target_t}, {target_h}, {target_w})"
        )
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate").squeeze(0)


def pad_video_latent_collate(batch):
    videos, latents, labels = zip(*batch)

    max_video_t = max(video.shape[1] for video in videos)
    max_video_h = max(video.shape[2] for video in videos)
    max_video_w = max(video.shape[3] for video in videos)
    max_latent_t = max(latent.shape[1] for latent in latents)
    max_latent_h = max(latent.shape[2] for latent in latents)
    max_latent_w = max(latent.shape[3] for latent in latents)

    padded_videos = [
        _pad_video_tensor(video, max_video_t, max_video_h, max_video_w)
        for video in videos
    ]
    padded_latents = [
        _pad_video_tensor(latent, max_latent_t, max_latent_h, max_latent_w)
        for latent in latents
    ]

    return torch.stack(padded_videos, dim=0), torch.stack(padded_latents, dim=0), list(labels)
