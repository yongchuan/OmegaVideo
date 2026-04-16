import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

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


OPENVID_REPO_ID = "nkp37/OpenVid-1M"
ARCHIVE_PATTERN = re.compile(r"^OpenVid_part(\d+)\.zip$")
SPLIT_PATTERN = re.compile(r"^(OpenVid_part\d+)_part.*$")
ARCHIVE_HINT_PATTERN = re.compile(r"(OpenVid_part\d+)")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
COPY_BUFFER_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True)
class SampleCandidate:
    archive_name: str
    video_name: str
    member_name: str
    member_basename: str
    caption: str


@dataclass(frozen=True)
class SavedVideo:
    archive_name: str
    video_name: str
    source_member: str
    saved_filename: str
    caption: str
    fps: float
    decoded_frames: int
    saved_frames: int
    padded_frames: int
    saved_bytes: int

    @property
    def source_basename(self) -> str:
        return Path(self.source_member).name

    @property
    def label_id(self) -> str:
        return Path(self.saved_filename).with_suffix("").as_posix()


@dataclass(frozen=True)
class MaterializedArchive:
    archive_path: Path
    cleanup_paths: Tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample OpenVid-1M proportionally by archive, save each selected sample as a short "
            "MP4 clip, and generate label_file.json."
        )
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Destination directory.")
    parser.add_argument("--max-videos", type=int, default=200000, help="How many videos to save.")
    parser.add_argument("--repo-id", type=str, default=OPENVID_REPO_ID, help="OpenVid dataset repo id.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to read.")
    parser.add_argument(
        "--videos-dir-name",
        type=str,
        default="source_videos",
        help="Subdirectory under output-dir used to store the saved MP4 clips.",
    )
    parser.add_argument(
        "--archive-cache-dir",
        type=str,
        default=None,
        help="Optional directory used for downloaded archive files and assembled split ZIPs.",
    )
    parser.add_argument(
        "--keep-archive-cache",
        action="store_true",
        help="Keep downloaded ZIP files under archive cache instead of deleting them after each archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing saved clips in the output directory.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="openvid_selection_manifest.json",
        help="Filename used for the selected-sample manifest JSON.",
    )
    parser.add_argument(
        "--archive-stats-name",
        type=str,
        default="openvid_archive_stats.json",
        help="Filename used for the per-archive counting and sampling summary JSON.",
    )
    parser.add_argument(
        "--metadata-db-name",
        type=str,
        default="openvid_captions.sqlite",
        help="Filename used for the on-disk caption SQLite cache. Default: openvid_captions.sqlite",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=17,
        help="Number of frames kept from the start of each selected video. Default: 17",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=24.0,
        help="FPS used when the source video metadata is missing. Default: 24",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for per-archive reservoir sampling. Default: 0",
    )
    parser.add_argument(
        "--num-archives",
        type=int,
        default=20,
        help="How many random ZIP archives to select and download. Default: 20",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=20,
        help="How many selected ZIP archives to download in parallel. Default: 20",
    )
    return parser.parse_args()


def archive_sort_key(name: str) -> int:
    match = ARCHIVE_PATTERN.match(name)
    return int(match.group(1)) if match else 10**18


def build_saved_filename(target_basename: str, collision_index: int) -> str:
    if collision_index == 0:
        return target_basename
    target_path = Path(target_basename)
    return f"{target_path.stem}__dup{collision_index:06d}{target_path.suffix}"


def build_target_basename(source_basename: str) -> str:
    return f"{Path(source_basename).stem}.mp4"


def get_archive_hint(video_name: str) -> Optional[str]:
    match = ARCHIVE_HINT_PATTERN.search(video_name)
    if match is None:
        return None
    return f"{match.group(1)}.zip"


def normalize_video_reference(value: str) -> str:
    value = value.strip()
    value = value.split("?", 1)[0]
    value = value.split("#", 1)[0]
    return value.replace("\\", "/")


def extract_video_references(video_field: Any) -> List[str]:
    candidates: List[str] = []

    if isinstance(video_field, str):
        candidates.append(video_field)
    elif isinstance(video_field, dict):
        for key in ("path", "filename", "file_name", "name", "url"):
            value = video_field.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
    else:
        path_value = getattr(video_field, "path", None)
        if isinstance(path_value, str) and path_value:
            candidates.append(path_value)

    normalized: List[str] = []
    seen: Set[str] = set()
    for candidate in candidates:
        normalized_candidate = normalize_video_reference(candidate)
        if not normalized_candidate or normalized_candidate in seen:
            continue
        normalized.append(normalized_candidate)
        seen.add(normalized_candidate)
    return normalized


def extract_video_basename(video_field: Any) -> Optional[str]:
    video_references = extract_video_references(video_field)
    if video_references:
        return Path(video_references[0]).name

    if isinstance(video_field, str):
        basename = Path(normalize_video_reference(video_field)).name
        return basename or None

    return None


def resolve_archive_name_from_references(
    video_references: Sequence[str],
    valid_archives: Optional[Set[str]] = None,
) -> Optional[str]:
    for video_reference in video_references:
        archive_name = get_archive_hint(video_reference)
        if archive_name is None:
            continue
        if valid_archives is None or archive_name in valid_archives:
            return archive_name
    return None


def select_primary_reference(video_references: Sequence[str], archive_name: str) -> Optional[str]:
    for video_reference in video_references:
        if get_archive_hint(video_reference) == archive_name:
            return video_reference
    return video_references[0] if video_references else None


def list_repo_files(repo_id: str) -> Set[str]:
    api = HfApi()
    return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))


def list_archive_names(repo_files: Iterable[str]) -> List[str]:
    archive_names = set()
    for path in repo_files:
        direct_match = ARCHIVE_PATTERN.match(path)
        if direct_match:
            archive_names.add(path)
            continue

        split_match = SPLIT_PATTERN.match(path)
        if split_match:
            archive_names.add(f"{split_match.group(1)}.zip")

    return sorted(archive_names, key=archive_sort_key)


def resolve_archive_filenames(zip_file: str, repo_files: Set[str]) -> List[str]:
    if zip_file in repo_files:
        return [zip_file]

    stem = Path(zip_file).stem
    split_prefix = f"{stem}_part"
    split_files = sorted(path for path in repo_files if path.startswith(split_prefix))
    if split_files:
        return split_files

    raise FileNotFoundError(f"Archive {zip_file} was not found in the dataset repo.")


def download_archive_part(repo_id: str, filename: str, archive_cache_dir: Path) -> Path:
    archive_cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(archive_cache_dir),
        )
    )


def assemble_split_archive(split_parts: Iterable[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    with output_path.open("wb") as output_file:
        for part_path in split_parts:
            with part_path.open("rb") as part_file:
                shutil.copyfileobj(part_file, output_file, length=COPY_BUFFER_SIZE)
    return output_path


def materialize_archive(
    repo_id: str,
    zip_file: str,
    repo_files: Set[str],
    archive_cache_dir: Path,
) -> MaterializedArchive:
    resolved_files = resolve_archive_filenames(zip_file, repo_files)
    if len(resolved_files) == 1:
        archive_path = download_archive_part(repo_id, resolved_files[0], archive_cache_dir)
        return MaterializedArchive(
            archive_path=archive_path,
            cleanup_paths=(archive_path,),
        )

    split_paths = [
        download_archive_part(repo_id, filename, archive_cache_dir)
        for filename in resolved_files
    ]
    assembled_dir = archive_cache_dir / "assembled"
    assembled_path = assembled_dir / zip_file
    archive_path = assemble_split_archive(split_paths, assembled_path)
    return MaterializedArchive(
        archive_path=archive_path,
        cleanup_paths=tuple(split_paths) + (assembled_path,),
    )


def cleanup_materialized_archive(materialized_archive: MaterializedArchive, archive_cache_dir: Path) -> None:
    cache_root = archive_cache_dir.resolve()
    seen_paths: Set[Path] = set()

    for path in materialized_archive.cleanup_paths:
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)

        try:
            resolved_path.relative_to(cache_root)
        except ValueError:
            continue

        if resolved_path.is_file():
            resolved_path.unlink()

    for relative_dir in ("assembled", ".cache"):
        candidate_dir = archive_cache_dir / relative_dir
        if candidate_dir.exists():
            try:
                candidate_dir.rmdir()
            except OSError:
                pass

    if archive_cache_dir.exists():
        try:
            archive_cache_dir.rmdir()
        except OSError:
            pass


def materialize_archives_parallel(
    repo_id: str,
    archive_names: Sequence[str],
    repo_files: Set[str],
    archive_cache_dir: Path,
    max_workers: int,
) -> Dict[str, MaterializedArchive]:
    if not archive_names:
        return {}

    effective_workers = max(1, min(max_workers, len(archive_names)))
    materialized_archives: Dict[str, MaterializedArchive] = {}

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_archive = {
            executor.submit(
                materialize_archive,
                repo_id,
                archive_name,
                repo_files,
                archive_cache_dir,
            ): archive_name
            for archive_name in archive_names
        }

        for future in tqdm(as_completed(future_to_archive), total=len(future_to_archive), desc="Downloading archives"):
            archive_name = future_to_archive[future]
            materialized_archives[archive_name] = future.result()

    return materialized_archives


def open_metadata_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def initialize_metadata_db(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS captions (
            video_basename TEXT PRIMARY KEY,
            caption TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS archive_members (
            archive_name TEXT NOT NULL,
            member_name TEXT NOT NULL,
            member_basename TEXT NOT NULL,
            PRIMARY KEY (archive_name, member_name)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_archive_members_archive ON archive_members(archive_name)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_archive_members_basename ON archive_members(member_basename)"
    )
    connection.commit()


def rebuild_caption_index(
    repo_id: str,
    split: str,
    db_path: Path,
) -> Dict[str, int]:
    dataset = load_dataset(repo_id, split=split, streaming=True)
    connection = open_metadata_db(db_path)
    initialize_metadata_db(connection)
    connection.execute("DELETE FROM captions")
    connection.commit()

    total_rows = 0
    rows_with_video = 0
    batch: List[Tuple[str, str]] = []

    try:
        for row in tqdm(dataset, desc="Indexing metadata captions"):
            total_rows += 1
            video_basename = extract_video_basename(row.get("video"))
            if not video_basename:
                continue

            rows_with_video += 1
            batch.append((video_basename, str(row.get("caption") or "")))
            if len(batch) >= 2048:
                connection.executemany(
                    "INSERT OR REPLACE INTO captions(video_basename, caption) VALUES (?, ?)",
                    batch,
                )
                connection.commit()
                batch.clear()

        if batch:
            connection.executemany(
                "INSERT OR REPLACE INTO captions(video_basename, caption) VALUES (?, ?)",
                batch,
            )
            connection.commit()

        unique_rows = int(connection.execute("SELECT COUNT(*) FROM captions").fetchone()[0])
        return {
            "total_rows": total_rows,
            "rows_with_video": rows_with_video,
            "unique_video_rows": unique_rows,
        }
    finally:
        connection.close()


def rebuild_archive_inventory(
    repo_id: str,
    repo_files: Set[str],
    archive_names: Sequence[str],
    archive_cache_dir: Path,
    db_path: Path,
    keep_archive_cache: bool,
) -> Dict[str, int]:
    connection = open_metadata_db(db_path)
    initialize_metadata_db(connection)
    connection.execute("DELETE FROM archive_members")
    connection.commit()

    counts_by_archive: Dict[str, int] = {}
    try:
        for archive_name in tqdm(archive_names, desc="Scanning archive members"):
            materialized_archive = materialize_archive(
                repo_id=repo_id,
                zip_file=archive_name,
                repo_files=repo_files,
                archive_cache_dir=archive_cache_dir,
            )
            member_rows: List[Tuple[str, str, str]] = []
            try:
                with zipfile.ZipFile(materialized_archive.archive_path, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        suffix = Path(info.filename).suffix.lower()
                        if suffix not in VIDEO_EXTENSIONS:
                            continue
                        member_rows.append((archive_name, info.filename, Path(info.filename).name))
            finally:
                if not keep_archive_cache:
                    cleanup_materialized_archive(materialized_archive, archive_cache_dir)

            if member_rows:
                connection.executemany(
                    "INSERT OR REPLACE INTO archive_members(archive_name, member_name, member_basename) "
                    "VALUES (?, ?, ?)",
                    member_rows,
                )
                connection.commit()
            counts_by_archive[archive_name] = len(member_rows)

        return counts_by_archive
    finally:
        connection.close()


def count_sampleable_videos_by_archive(db_path: Path) -> Dict[str, int]:
    connection = open_metadata_db(db_path)
    initialize_metadata_db(connection)
    try:
        rows = connection.execute(
            """
            SELECT archive_members.archive_name, COUNT(*)
            FROM archive_members
            INNER JOIN captions
                ON captions.video_basename = archive_members.member_basename
            GROUP BY archive_members.archive_name
            """
        ).fetchall()
        return {str(archive_name): int(count) for archive_name, count in rows}
    finally:
        connection.close()


def load_caption_map(
    repo_id: str,
    split: str,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    dataset = load_dataset(repo_id, split=split, streaming=True)
    caption_map: Dict[str, str] = {}
    total_rows = 0
    rows_with_video = 0

    for row in tqdm(dataset, desc="Loading metadata captions"):
        total_rows += 1
        video_basename = extract_video_basename(row.get("video"))
        if not video_basename:
            continue
        rows_with_video += 1
        caption_map[video_basename] = str(row.get("caption") or "")

    stats = {
        "total_rows": total_rows,
        "rows_with_video": rows_with_video,
        "unique_video_rows": len(caption_map),
    }
    return caption_map, stats


def lookup_caption(connection: sqlite3.Connection, video_basename: str) -> Optional[str]:
    row = connection.execute(
        "SELECT caption FROM captions WHERE video_basename = ?",
        (video_basename,),
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def sample_candidates_from_inventory(
    db_path: Path,
    selection_quotas: Dict[str, int],
    random_seed: int,
) -> Dict[str, List[SampleCandidate]]:
    connection = open_metadata_db(db_path)
    initialize_metadata_db(connection)
    rng = random.Random(random_seed)
    candidates_by_archive: Dict[str, List[SampleCandidate]] = {}

    try:
        for archive_name, sample_size in selection_quotas.items():
            if sample_size <= 0:
                candidates_by_archive[archive_name] = []
                continue

            rows = connection.execute(
                """
                SELECT archive_members.member_name, archive_members.member_basename, captions.caption
                FROM archive_members
                INNER JOIN captions
                    ON captions.video_basename = archive_members.member_basename
                WHERE archive_members.archive_name = ?
                """,
                (archive_name,),
            ).fetchall()

            if not rows:
                candidates_by_archive[archive_name] = []
                continue

            if len(rows) > sample_size:
                sampled_rows = rng.sample(rows, sample_size)
            else:
                sampled_rows = list(rows)

            candidates_by_archive[archive_name] = [
                SampleCandidate(
                    archive_name=archive_name,
                    video_name=str(member_basename),
                    member_name=str(member_name),
                    member_basename=str(member_basename),
                    caption=str(caption or ""),
                )
                for member_name, member_basename, caption in sampled_rows
            ]

        return candidates_by_archive
    finally:
        connection.close()


def sample_candidates_from_archive(
    archive_name: str,
    archive_path: Path,
    caption_connection: sqlite3.Connection,
    sample_size: int,
    rng: random.Random,
) -> Tuple[List[SampleCandidate], int, int]:
    sampled_candidates: List[SampleCandidate] = []
    archive_video_count = 0
    sampleable_count = 0

    if sample_size <= 0:
        return sampled_candidates, archive_video_count, sampleable_count

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                continue

            archive_video_count += 1
            member_basename = Path(info.filename).name
            caption = lookup_caption(caption_connection, member_basename)
            if caption is None:
                continue

            sampleable_count += 1
            candidate = SampleCandidate(
                archive_name=archive_name,
                video_name=member_basename,
                member_name=info.filename,
                member_basename=member_basename,
                caption=caption,
            )

            if len(sampled_candidates) < sample_size:
                sampled_candidates.append(candidate)
                continue

            replace_index = rng.randint(0, sampleable_count - 1)
            if replace_index < sample_size:
                sampled_candidates[replace_index] = candidate

    return sampled_candidates, archive_video_count, sampleable_count


def trim_saved_videos_to_target(
    saved_videos: List[SavedVideo],
    videos_dir: Path,
    target_count: int,
    random_seed: int,
) -> List[SavedVideo]:
    if len(saved_videos) <= target_count:
        return saved_videos

    rng = random.Random(random_seed)
    kept_indices = set(rng.sample(range(len(saved_videos)), target_count))
    kept_videos: List[SavedVideo] = []

    for index, video in enumerate(saved_videos):
        if index in kept_indices:
            kept_videos.append(video)
            continue

        output_path = videos_dir / video.saved_filename
        if output_path.exists():
            output_path.unlink()

    return kept_videos


def count_rows_by_archive(
    repo_id: str,
    split: str,
    valid_archives: Set[str],
) -> Tuple[Dict[str, int], int, int]:
    counts_by_archive: Dict[str, int] = defaultdict(int)
    total_rows = 0
    unresolved_rows = 0
    dataset = load_dataset(repo_id, split=split, streaming=True)

    for row in tqdm(dataset, desc="Counting metadata rows"):
        total_rows += 1
        video_references = extract_video_references(row.get("video"))
        archive_name = resolve_archive_name_from_references(video_references, valid_archives)
        if archive_name is None:
            unresolved_rows += 1
            continue
        counts_by_archive[archive_name] += 1

    return counts_by_archive, total_rows, unresolved_rows


def allocate_sample_quotas(counts_by_archive: Dict[str, int], target_count: int) -> Dict[str, int]:
    total_available = sum(counts_by_archive.values())
    if target_count > total_available:
        raise ValueError(
            f"Requested max_videos={target_count}, but only {total_available} metadata rows "
            "were resolvable to a source archive."
        )

    ordered_archives = [
        archive_name
        for archive_name in sorted(counts_by_archive, key=archive_sort_key)
        if counts_by_archive[archive_name] > 0
    ]
    quotas = {archive_name: 0 for archive_name in ordered_archives}
    if not ordered_archives or target_count <= 0:
        return quotas

    if target_count >= len(ordered_archives):
        for archive_name in ordered_archives:
            quotas[archive_name] = 1

    remaining_target = target_count - sum(quotas.values())
    if remaining_target <= 0:
        return quotas

    remaining_capacity = {
        archive_name: counts_by_archive[archive_name] - quotas[archive_name]
        for archive_name in ordered_archives
    }
    total_remaining_capacity = sum(remaining_capacity.values())
    if total_remaining_capacity <= 0:
        return quotas

    fractional_shares: List[Tuple[float, str]] = []
    used = 0
    for archive_name in ordered_archives:
        capacity = remaining_capacity[archive_name]
        if capacity <= 0:
            fractional_shares.append((0.0, archive_name))
            continue

        ideal = remaining_target * capacity / total_remaining_capacity
        addition = min(capacity, math.floor(ideal))
        quotas[archive_name] += addition
        used += addition
        fractional_shares.append((ideal - addition, archive_name))

    leftovers = remaining_target - used
    for _, archive_name in sorted(
        fractional_shares,
        key=lambda item: (-item[0], archive_sort_key(item[1])),
    ):
        if leftovers <= 0:
            break
        if quotas[archive_name] >= counts_by_archive[archive_name]:
            continue
        quotas[archive_name] += 1
        leftovers -= 1

    if sum(quotas.values()) != target_count:
        raise RuntimeError(
            f"Failed to allocate exact quotas for {target_count} videos; got {sum(quotas.values())}."
        )

    return quotas


def build_selection_quotas(
    counts_by_archive: Dict[str, int],
    sample_quotas: Dict[str, int],
    reserve_per_archive: int,
    reserve_ratio: float,
) -> Dict[str, int]:
    selection_quotas: Dict[str, int] = {}
    for archive_name, quota in sample_quotas.items():
        if quota <= 0:
            selection_quotas[archive_name] = 0
            continue
        reserve = max(max(0, reserve_per_archive), int(math.ceil(quota * max(0.0, reserve_ratio))))
        selection_quotas[archive_name] = min(counts_by_archive[archive_name], quota + reserve)
    return selection_quotas


def reservoir_sample_by_archive(
    repo_id: str,
    split: str,
    selection_quotas: Dict[str, int],
    valid_archives: Set[str],
    random_seed: int,
) -> Tuple[Dict[str, List[SampleCandidate]], Dict[str, int], int]:
    dataset = load_dataset(repo_id, split=split, streaming=True)
    reservoirs = {
        archive_name: []
        for archive_name, quota in selection_quotas.items()
        if quota > 0
    }
    seen_by_archive: Dict[str, int] = defaultdict(int)
    unresolved_rows = 0
    rng = random.Random(random_seed)

    for row in tqdm(dataset, desc="Sampling metadata rows"):
        video_references = extract_video_references(row.get("video"))
        archive_name = resolve_archive_name_from_references(video_references, valid_archives)
        if archive_name is None:
            unresolved_rows += 1
            continue

        sample_size = selection_quotas.get(archive_name, 0)
        if sample_size <= 0:
            continue

        primary_reference = select_primary_reference(video_references, archive_name)
        if primary_reference is None:
            continue

        candidate = SampleCandidate(
            archive_name=archive_name,
            video_name=primary_reference,
            member_name=Path(primary_reference).name,
            member_basename=Path(primary_reference).name,
            caption=str(row.get("caption") or ""),
        )
        seen_by_archive[archive_name] += 1
        reservoir = reservoirs[archive_name]

        if len(reservoir) < sample_size:
            reservoir.append(candidate)
            continue

        replace_index = rng.randint(0, seen_by_archive[archive_name] - 1)
        if replace_index < sample_size:
            reservoir[replace_index] = candidate

    return reservoirs, seen_by_archive, unresolved_rows


def save_archive_stats(
    output_path: Path,
    repo_id: str,
    split: str,
    metadata_stats: Dict[str, int],
    archive_video_counts: Dict[str, int],
    sampleable_counts: Dict[str, int],
    sample_quotas: Dict[str, int],
    selection_quotas: Dict[str, int],
) -> None:
    ordered_archives = [
        archive_name
        for archive_name in sorted(archive_video_counts, key=archive_sort_key)
        if archive_video_counts[archive_name] > 0
    ]
    payload = {
        "repo_id": repo_id,
        "split": split,
        "metadata_total_rows": int(metadata_stats.get("total_rows", 0)),
        "metadata_rows_with_video": int(metadata_stats.get("rows_with_video", 0)),
        "metadata_unique_video_rows": int(metadata_stats.get("unique_video_rows", 0)),
        "archive_video_total": int(sum(archive_video_counts.values())),
        "sampleable_video_total": int(sum(sampleable_counts.values())),
        "archive_count": len(ordered_archives),
        "sample_quota_total": sum(sample_quotas.values()),
        "selection_quota_total": sum(selection_quotas.values()),
        "per_archive": [
            {
                "archive": archive_name,
                "archive_video_count": archive_video_counts[archive_name],
                "sampleable_count": sampleable_counts.get(archive_name, 0),
                "sample_quota": sample_quotas.get(archive_name, 0),
                "selection_quota": selection_quotas.get(archive_name, 0),
            }
            for archive_name in ordered_archives
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_video_with_torchvision(
    video_path: Path,
    fallback_fps: float,
    max_frames: int,
) -> Tuple[torch.Tensor, float]:
    if read_video is None:
        raise RuntimeError("torchvision.io.read_video is unavailable.")

    frames, _, info = read_video(str(video_path), pts_unit="sec")
    if frames.numel() == 0:
        raise ValueError(f"Input video has no frames: {video_path}")

    if max_frames > 0:
        frames = frames[:max_frames]

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames in [T, H, W, C] RGB format, got {tuple(frames.shape)}")

    fps = float(info.get("video_fps") or fallback_fps)
    return frames.permute(3, 0, 1, 2).contiguous(), fps


def load_video_with_ffmpeg(
    video_path: Path,
    fallback_fps: float,
    max_frames: int,
) -> Tuple[torch.Tensor, float]:
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    probe_result = subprocess.run(
        probe_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    probe_payload = json.loads(probe_result.stdout.decode("utf-8"))
    streams = probe_payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in: {video_path}")

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video resolution in ffprobe output: {video_path}")

    avg_frame_rate = str(stream.get("avg_frame_rate") or "0/0")
    fps = fallback_fps
    if avg_frame_rate not in ("0/0", "0", ""):
        try:
            numerator, denominator = avg_frame_rate.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value != 0:
                fps = float(numerator) / denominator_value
        except Exception:
            fps = fallback_fps

    ffmpeg_command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    ffmpeg_result = subprocess.run(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw_video = ffmpeg_result.stdout
    frame_stride = width * height * 3
    if frame_stride <= 0 or len(raw_video) < frame_stride:
        raise ValueError(f"Unable to decode frames with ffmpeg: {video_path}")

    frame_count = len(raw_video) // frame_stride
    frame_tensor = torch.frombuffer(memoryview(raw_video), dtype=torch.uint8).clone()
    frame_tensor = frame_tensor.view(frame_count, height, width, 3)
    return frame_tensor.permute(3, 0, 1, 2).contiguous(), float(fps)


def load_video_with_pyav(
    video_path: Path,
    fallback_fps: float,
    max_frames: int,
) -> Tuple[torch.Tensor, float]:
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
            if max_frames > 0 and len(frames) >= max_frames:
                break

        if not frames:
            raise ValueError(f"Input video has no decodable frames: {video_path}")

        return torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous(), fps
    finally:
        container.close()


def load_video_with_cv2(
    video_path: Path,
    fallback_fps: float,
    max_frames: int,
) -> Tuple[torch.Tensor, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video with OpenCV: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = fallback_fps

        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
            if max_frames > 0 and len(frames) >= max_frames:
                break

        if not frames:
            raise ValueError(f"Input video has no decodable frames: {video_path}")

        return torch.stack(frames, dim=0).permute(3, 0, 1, 2).contiguous(), fps
    finally:
        capture.release()


def load_video(video_path: Path, fallback_fps: float, max_frames: int) -> Tuple[torch.Tensor, float]:
    last_error = None

    for loader in (load_video_with_ffmpeg, load_video_with_cv2, load_video_with_pyav):
        try:
            return loader(video_path, fallback_fps, max_frames)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to read video. Install ffmpeg, OpenCV, or PyAV."
    ) from last_error


def trim_or_pad_video(video: torch.Tensor, target_frames: int) -> Tuple[torch.Tensor, int]:
    if video.shape[1] >= target_frames:
        return video[:, :target_frames].contiguous(), 0

    if video.shape[1] == 0:
        raise ValueError("Decoded video contains zero frames.")

    pad_frames = target_frames - video.shape[1]
    last_frame = video[:, -1:, :, :].repeat(1, pad_frames, 1, 1)
    return torch.cat([video, last_frame], dim=1).contiguous(), pad_frames


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

    from fractions import Fraction

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

    for saver in (save_video_with_cv2, save_video_with_pyav, save_video_with_torchvision):
        try:
            saver(video, output_path, fps)
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to save video. Install OpenCV, PyAV, or a torchvision video backend."
    ) from last_error


def transcode_member_to_short_mp4(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    clip_frames: int,
    fallback_fps: float,
    output_path: Path,
) -> Tuple[float, int, int]:
    suffix = Path(info.filename).suffix or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        with zf.open(info, "r") as src, temp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=COPY_BUFFER_SIZE)

        probe_command = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_read_frames,nb_frames",
            "-of",
            "json",
            str(temp_path),
        ]
        probe_result = subprocess.run(
            probe_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        probe_payload = json.loads(probe_result.stdout.decode("utf-8"))
        streams = probe_payload.get("streams") or []
        if not streams:
            raise ValueError(f"No video stream found in: {temp_path}")

        stream = streams[0]
        avg_frame_rate = str(stream.get("avg_frame_rate") or "0/0")
        fps = fallback_fps
        if avg_frame_rate not in ("0/0", "0", ""):
            try:
                numerator, denominator = avg_frame_rate.split("/", 1)
                denominator_value = float(denominator)
                if denominator_value != 0:
                    fps = float(numerator) / denominator_value
            except Exception:
                fps = fallback_fps

        decoded_frames = 0
        for field_name in ("nb_read_frames", "nb_frames"):
            field_value = stream.get(field_name)
            if field_value not in (None, "N/A", ""):
                try:
                    decoded_frames = int(field_value)
                    break
                except Exception:
                    continue
        if decoded_frames <= 0:
            decoded_frames = clip_frames

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(temp_path),
            "-vf",
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2,tpad=stop_mode=clone:stop={max(0, clip_frames - 1)},trim=end_frame={clip_frames}",
            "-an",
            "-frames:v",
            str(clip_frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        padded_frames = max(0, clip_frames - min(decoded_frames, clip_frames))
        return float(fps), min(decoded_frames, clip_frames), padded_frames
    finally:
        temp_path.unlink(missing_ok=True)


def extract_selected_videos_from_archive(
    archive_name: str,
    archive_path: Path,
    candidates: List[SampleCandidate],
    required_count: int,
    videos_dir: Path,
    basename_counts: Dict[str, int],
    overwrite: bool,
    clip_frames: int,
    fallback_fps: float,
) -> Tuple[List[SavedVideo], Dict[str, object]]:
    saved_videos: List[SavedVideo] = []
    decode_failures = 0
    failure_examples: List[str] = []
    processed_video_count = 0

    if required_count <= 0 or not candidates:
        return saved_videos, {
            "archive": archive_name,
            "requested_candidates": len(candidates),
            "required_count": required_count,
            "saved_count": 0,
            "decode_failures": 0,
            "failure_examples": [],
            "remaining_candidates": 0,
        }

    pending_by_member_name: DefaultDict[str, List[SampleCandidate]] = defaultdict(list)
    for candidate in candidates:
        pending_by_member_name[candidate.member_name].append(candidate)

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if len(saved_videos) >= required_count:
                break
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                continue

            matched_candidates = pending_by_member_name.pop(info.filename, None)
            if not matched_candidates:
                continue

            try:
                transcode_results: List[Tuple[SampleCandidate, float, int, int, Path]] = []
                remaining_slots = required_count - len(saved_videos)
                for candidate in matched_candidates[:remaining_slots]:
                    target_basename = build_target_basename(candidate.member_basename)
                    collision_index = basename_counts[target_basename]
                    saved_filename = build_saved_filename(target_basename, collision_index)
                    basename_counts[target_basename] += 1

                    output_path = videos_dir / saved_filename
                    if not output_path.exists() or overwrite:
                        fps, decoded_frames, padded_frames = transcode_member_to_short_mp4(
                            zf=zf,
                            info=info,
                            clip_frames=clip_frames,
                            fallback_fps=fallback_fps,
                            output_path=output_path,
                        )
                    else:
                        fps = fallback_fps
                        decoded_frames = clip_frames
                        padded_frames = 0
                    transcode_results.append((candidate, fps, decoded_frames, padded_frames, output_path))
            except Exception as exc:
                decode_failures += len(matched_candidates)
                if len(failure_examples) < 10:
                    failure_examples.append(f"{archive_name}:{info.filename} -> {exc}")
                continue

            for candidate, fps, decoded_frames, padded_frames, output_path in transcode_results:
                saved_videos.append(
                    SavedVideo(
                        archive_name=archive_name,
                        video_name=candidate.video_name,
                        source_member=info.filename,
                        saved_filename=output_path.name,
                        caption=candidate.caption,
                        fps=float(fps),
                        decoded_frames=decoded_frames,
                        saved_frames=clip_frames,
                        padded_frames=padded_frames,
                        saved_bytes=output_path.stat().st_size,
                    )
                )

                if len(saved_videos) >= required_count:
                    break

            processed_video_count += 1
            if processed_video_count % 32 == 0:
                gc.collect()

    remaining_candidates = sum(len(queue) for queue in pending_by_member_name.values())
    return saved_videos, {
        "archive": archive_name,
        "requested_candidates": len(candidates),
        "required_count": required_count,
        "saved_count": len(saved_videos),
        "decode_failures": decode_failures,
        "failure_examples": failure_examples,
        "remaining_candidates": remaining_candidates,
    }


def save_label_file(saved_videos: List[SavedVideo], label_file_path: Path) -> None:
    labels = [{"id": video.label_id, "en": video.caption} for video in saved_videos]
    label_file_path.parent.mkdir(parents=True, exist_ok=True)
    with label_file_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def save_manifest(saved_videos: List[SavedVideo], manifest_path: Path) -> None:
    payload = [
        {
            "id": video.label_id,
            "archive": video.archive_name,
            "video": video.video_name,
            "source_member": video.source_member,
            "saved_as": video.saved_filename,
            "caption": video.caption,
            "fps": video.fps,
            "decoded_frames": video.decoded_frames,
            "saved_frames": video.saved_frames,
            "padded_frames": video.padded_frames,
            "saved_bytes": video.saved_bytes,
        }
        for video in saved_videos
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_run_summary(
    output_path: Path,
    saved_videos: List[SavedVideo],
    archive_summaries: List[Dict[str, object]],
    target_count: int,
) -> None:
    payload = {
        "target_count": target_count,
        "saved_count": len(saved_videos),
        "total_saved_bytes": sum(video.saved_bytes for video in saved_videos),
        "archives": archive_summaries,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(100.0 * numerator / denominator):.2f}%"


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / args.videos_dir_name
    label_file_path = output_dir / "label_file.json"
    manifest_path = output_dir / args.manifest_name
    archive_stats_path = output_dir / args.archive_stats_name
    metadata_db_path = output_dir / args.metadata_db_name
    run_summary_path = output_dir / "openvid_download_summary.json"
    archive_cache_dir = (
        Path(args.archive_cache_dir)
        if args.archive_cache_dir is not None
        else output_dir / "_archive_cache"
    )

    repo_files = list_repo_files(args.repo_id)
    archive_names = list_archive_names(repo_files)
    if not archive_names:
        raise FileNotFoundError(f"No OpenVid archives found in dataset repo: {args.repo_id}")

    metadata_stats = rebuild_caption_index(
        repo_id=args.repo_id,
        split=args.split,
        db_path=metadata_db_path,
    )
    if int(metadata_stats.get("unique_video_rows", 0)) <= 0:
        raise RuntimeError("No captions were loaded from metadata.")

    rng = random.Random(args.random_seed)
    shuffled_archive_names = list(archive_names)
    rng.shuffle(shuffled_archive_names)
    selected_archive_count = min(max(1, args.num_archives), len(shuffled_archive_names))
    selected_archive_names = shuffled_archive_names[:selected_archive_count]
    archive_video_counts: Dict[str, int] = {}
    sampleable_counts: Dict[str, int] = {}
    sample_quotas: Dict[str, int] = {}
    selection_quotas: Dict[str, int] = {}

    print(
        f"Metadata rows: total={metadata_stats['total_rows']}, with_video={metadata_stats['rows_with_video']}, "
        f"unique_video={metadata_stats['unique_video_rows']}"
    )
    print(
        f"Archives={len(archive_names)}, selected={selected_archive_count}, "
        f"target={args.max_videos}, random_seed={args.random_seed}"
    )
    print(
        f"Caption cache rows={metadata_stats['unique_video_rows']}, "
        f"metadata_db={metadata_db_path}, stats will be written to {archive_stats_path}"
    )
    if args.keep_archive_cache:
        print(f"Archive cache retention: enabled at {archive_cache_dir}")
    else:
        print(f"Archive cache retention: disabled, processed ZIPs will be deleted from {archive_cache_dir}")
    print(
        f"Parallel download workers={max(1, min(args.download_workers, selected_archive_count))}, "
        f"selected_archives={selected_archive_names}"
    )

    videos_dir.mkdir(parents=True, exist_ok=True)
    basename_counts: Dict[str, int] = defaultdict(int)
    saved_videos: List[SavedVideo] = []
    archive_summaries: List[Dict[str, object]] = []
    processed_archives = 0
    caption_connection = open_metadata_db(metadata_db_path)
    materialized_archives = materialize_archives_parallel(
        repo_id=args.repo_id,
        archive_names=selected_archive_names,
        repo_files=repo_files,
        archive_cache_dir=archive_cache_dir,
        max_workers=args.download_workers,
    )
    processed_archive_names: List[str] = []

    try:
        for archive_name in tqdm(selected_archive_names, desc="Extracting downloaded archives"):
            remaining_needed = args.max_videos - len(saved_videos)
            if remaining_needed <= 0:
                break
            remaining_archives = selected_archive_count - processed_archives
            sample_target = int(math.ceil(remaining_needed / max(1, remaining_archives)))

            materialized_archive = materialized_archives[archive_name]
            try:
                sampled_candidates, archive_video_count, sampleable_count = sample_candidates_from_archive(
                    archive_name=archive_name,
                    archive_path=materialized_archive.archive_path,
                    caption_connection=caption_connection,
                    sample_size=sample_target,
                    rng=rng,
                )
                archive_video_counts[archive_name] = archive_video_count
                sampleable_counts[archive_name] = sampleable_count
                sample_quotas[archive_name] = sample_target
                selection_quotas[archive_name] = len(sampled_candidates)

                archive_saved_videos, archive_summary = extract_selected_videos_from_archive(
                    archive_name=archive_name,
                    archive_path=materialized_archive.archive_path,
                    candidates=sampled_candidates,
                    required_count=min(len(sampled_candidates), remaining_needed),
                    videos_dir=videos_dir,
                    basename_counts=basename_counts,
                    overwrite=args.overwrite,
                    clip_frames=args.clip_frames,
                    fallback_fps=args.fallback_fps,
                )
            finally:
                if not args.keep_archive_cache:
                    cleanup_materialized_archive(materialized_archive, archive_cache_dir)
            saved_videos.extend(archive_saved_videos)
            archive_summaries.append(archive_summary)
            processed_archive_names.append(archive_name)
            processed_archives += 1

            print(
                f"[{processed_archives}/{selected_archive_count}] {archive_name} | "
                f"videos={archive_video_count} | sampleable={sampleable_count} "
                f"({format_ratio(sampleable_count, archive_video_count)}) | "
                f"sample_target={sample_target} | sampled={len(sampled_candidates)} | saved={len(archive_saved_videos)} | "
                f"decode_failures={archive_summary['decode_failures']} | "
                f"saved_total={len(saved_videos)} / {args.max_videos}"
            )
            del sampled_candidates
            gc.collect()
    finally:
        caption_connection.close()

    if not args.keep_archive_cache:
        for archive_name in selected_archive_names:
            if archive_name in processed_archive_names:
                continue
            cleanup_materialized_archive(materialized_archives[archive_name], archive_cache_dir)

    save_archive_stats(
        output_path=archive_stats_path,
        repo_id=args.repo_id,
        split=args.split,
        metadata_stats=metadata_stats,
        archive_video_counts=archive_video_counts,
        sampleable_counts=sampleable_counts,
        sample_quotas=sample_quotas,
        selection_quotas=selection_quotas,
    )

    print(
        f"Overall match rate: {sum(sampleable_counts.values())} / {sum(archive_video_counts.values())} "
        f"({format_ratio(sum(sampleable_counts.values()), sum(archive_video_counts.values()))})"
    )

    if len(saved_videos) < args.max_videos:
        shortfall = args.max_videos - len(saved_videos)
        raise RuntimeError(
            f"Only saved {len(saved_videos)} clips, which is {shortfall} fewer than requested "
            f"max_videos={args.max_videos}. Processed all random archives but still ran out of matched videos."
        )

    save_label_file(saved_videos, label_file_path)
    save_manifest(saved_videos, manifest_path)
    save_run_summary(run_summary_path, saved_videos, archive_summaries, args.max_videos)

    total_saved_bytes = sum(video.saved_bytes for video in saved_videos)
    print(f"Saved {len(saved_videos)} clips under {videos_dir}")
    print(f"Saved captions to {label_file_path}")
    print(f"Saved manifest to {manifest_path}")
    print(f"Saved archive/run summaries to {archive_stats_path} and {run_summary_path}")
    print(f"Actual saved size: {format_bytes(total_saved_bytes)}")


if __name__ == "__main__":
    main()
