import argparse
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


OPENVID_REPO_ID = "nkp37/OpenVid-1M"
ARCHIVE_PATTERN = re.compile(r"^OpenVid_part(\d+)\.zip$")
SPLIT_PATTERN = re.compile(r"^(OpenVid_part\d+)_part.*$")
ARCHIVE_HINT_PATTERN = re.compile(r"(OpenVid_part\d+)")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


@dataclass(frozen=True)
class SelectedVideo:
    video_name: str
    caption: str
    saved_filename: str

    @property
    def source_basename(self) -> str:
        return Path(self.video_name).name

    @property
    def label_id(self) -> str:
        return Path(self.saved_filename).with_suffix("").as_posix()


@dataclass(frozen=True)
class ExtractedVideo:
    archive_name: str
    member_name: str
    saved_filename: str

    @property
    def source_basename(self) -> str:
        return Path(self.member_name).name

    @property
    def label_id(self) -> str:
        return Path(self.saved_filename).with_suffix("").as_posix()


def build_saved_filename(source_basename: str, collision_index: int) -> str:
    if collision_index == 0:
        return source_basename

    source_path = Path(source_basename)
    suffix = source_path.suffix
    stem = source_path.stem
    return f"{stem}__dup{collision_index:06d}{suffix}"


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


def remove_pending_video(
    extracted: ExtractedVideo,
    pending_exact: DefaultDict[Tuple[str, str], List[ExtractedVideo]],
    pending_basename: DefaultDict[str, List[ExtractedVideo]],
) -> None:
    exact_key = (extracted.archive_name, extracted.source_basename)
    exact_queue = pending_exact.get(exact_key)
    if exact_queue:
        pending_exact[exact_key] = [item for item in exact_queue if item.saved_filename != extracted.saved_filename]
        if not pending_exact[exact_key]:
            del pending_exact[exact_key]

    basename_queue = pending_basename.get(extracted.source_basename)
    if basename_queue:
        pending_basename[extracted.source_basename] = [
            item for item in basename_queue if item.saved_filename != extracted.saved_filename
        ]
        if not pending_basename[extracted.source_basename]:
            del pending_basename[extracted.source_basename]


def pop_matching_video(
    video_references: List[str],
    pending_exact: DefaultDict[Tuple[str, str], List[ExtractedVideo]],
    pending_basename: DefaultDict[str, List[ExtractedVideo]],
) -> Optional[Tuple[ExtractedVideo, str]]:
    for video_reference in video_references:
        basename = Path(video_reference).name
        archive_hint = get_archive_hint(video_reference)
        if archive_hint is not None:
            exact_key = (archive_hint, basename)
            exact_queue = pending_exact.get(exact_key)
            if exact_queue:
                extracted = exact_queue[0]
                remove_pending_video(extracted, pending_exact, pending_basename)
                return extracted, video_reference

    for video_reference in video_references:
        basename = Path(video_reference).name
        basename_queue = pending_basename.get(basename)
        if basename_queue:
            extracted = basename_queue[0]
            remove_pending_video(extracted, pending_exact, pending_basename)
            return extracted, video_reference

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the first N OpenVid-1M videos and generate label_file.json."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Destination directory.")
    parser.add_argument("--max-videos", type=int, default=10000, help="How many videos to save.")
    parser.add_argument("--repo-id", type=str, default=OPENVID_REPO_ID, help="OpenVid dataset repo id.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to read.")
    parser.add_argument(
        "--videos-dir-name",
        type=str,
        default="source_videos",
        help="Subdirectory under output-dir used to store the downloaded source videos.",
    )
    parser.add_argument(
        "--archive-cache-dir",
        type=str,
        default=None,
        help="Optional directory used for downloaded archive files and assembled split ZIPs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing source video files in the output directory.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="openvid_selection_manifest.json",
        help="Filename used for the selected-sample manifest JSON.",
    )
    return parser.parse_args()


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


def archive_sort_key(name: str):
    match = ARCHIVE_PATTERN.match(name)
    return int(match.group(1)) if match else float("inf")


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
                shutil.copyfileobj(part_file, output_file, length=16 * 1024 * 1024)
    return output_path


def materialize_archive(
    repo_id: str,
    zip_file: str,
    repo_files: Set[str],
    archive_cache_dir: Path,
) -> Path:
    resolved_files = resolve_archive_filenames(zip_file, repo_files)
    if len(resolved_files) == 1:
        return download_archive_part(repo_id, resolved_files[0], archive_cache_dir)

    split_paths = [
        download_archive_part(repo_id, filename, archive_cache_dir)
        for filename in resolved_files
    ]
    assembled_dir = archive_cache_dir / "assembled"
    assembled_path = assembled_dir / zip_file
    return assemble_split_archive(split_paths, assembled_path)


def extract_videos_from_archive(
    archive_name: str,
    archive_path: Path,
    videos_dir: Path,
    basename_counts: Dict[str, int],
    remaining_count: int,
    overwrite: bool,
) -> List[ExtractedVideo]:
    extracted: List[ExtractedVideo] = []
    if remaining_count <= 0:
        return extracted

    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            suffix = Path(info.filename).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                continue

            basename = Path(info.filename).name
            collision_index = basename_counts[basename]
            saved_filename = build_saved_filename(basename, collision_index)
            basename_counts[basename] += 1

            output_path = videos_dir / saved_filename
            if not output_path.exists() or overwrite:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, output_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

            extracted.append(
                ExtractedVideo(
                    archive_name=archive_name,
                    member_name=info.filename,
                    saved_filename=saved_filename,
                )
            )
            if len(extracted) >= remaining_count:
                break

    return extracted


def resolve_captions_for_extracted(
    repo_id: str,
    split: str,
    extracted_videos: List[ExtractedVideo],
) -> List[SelectedVideo]:
    pending_exact: DefaultDict[Tuple[str, str], List[ExtractedVideo]] = defaultdict(list)
    pending_basename: DefaultDict[str, List[ExtractedVideo]] = defaultdict(list)
    for video in extracted_videos:
        pending_exact[(video.archive_name, video.source_basename)].append(video)
        pending_basename[video.source_basename].append(video)

    dataset = load_dataset(repo_id, split=split, streaming=True)
    resolved_by_filename: Dict[str, SelectedVideo] = {}
    progress = tqdm(total=len(extracted_videos), desc="Resolving captions")

    try:
        for row in dataset:
            video_references = extract_video_references(row.get("video"))
            if not video_references:
                continue

            match = pop_matching_video(
                video_references=video_references,
                pending_exact=pending_exact,
                pending_basename=pending_basename,
            )
            if match is None:
                continue

            extracted, video_name = match
            resolved_by_filename[extracted.saved_filename] = SelectedVideo(
                video_name=video_name,
                caption=str(row["caption"]),
                saved_filename=extracted.saved_filename,
            )
            progress.update(1)

            if not pending_exact:
                break
    finally:
        progress.close()

    if pending_exact:
        missing_count = sum(len(queue) for queue in pending_exact.values())
        missing_examples = [
            f"{archive_name}:{basename}"
            for archive_name, basename in list(pending_exact.keys())[:10]
        ]
        raise RuntimeError(
            f"Failed to resolve captions for {missing_count} extracted videos. "
            f"Examples: {missing_examples}"
        )

    return [resolved_by_filename[video.saved_filename] for video in extracted_videos]


def save_label_file(selected_videos: List[SelectedVideo], label_file_path: Path) -> None:
    labels = [
        {"id": video.label_id, "en": video.caption}
        for video in selected_videos
    ]
    label_file_path.parent.mkdir(parents=True, exist_ok=True)
    with label_file_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def save_manifest(selected_videos: List[SelectedVideo], manifest_path: Path) -> None:
    payload = [
        {
            "video": video.video_name,
            "saved_as": video.saved_filename,
            "caption": video.caption,
            "id": video.label_id,
        }
        for video in selected_videos
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / args.videos_dir_name
    label_file_path = output_dir / "label_file.json"
    manifest_path = output_dir / args.manifest_name
    archive_cache_dir = (
        Path(args.archive_cache_dir)
        if args.archive_cache_dir is not None
        else output_dir / "_archive_cache"
    )

    repo_files = list_repo_files(args.repo_id)
    archive_names = list_archive_names(repo_files)
    if not archive_names:
        raise FileNotFoundError(f"No OpenVid archives found in dataset repo: {args.repo_id}")

    videos_dir.mkdir(parents=True, exist_ok=True)
    basename_counts: Dict[str, int] = defaultdict(int)
    extracted_videos: List[ExtractedVideo] = []

    for zip_file in tqdm(archive_names, desc="Processing archives"):
        remaining_count = args.max_videos - len(extracted_videos)
        if remaining_count <= 0:
            break

        archive_path = materialize_archive(
            repo_id=args.repo_id,
            zip_file=zip_file,
            repo_files=repo_files,
            archive_cache_dir=archive_cache_dir,
        )
        extracted_videos.extend(
            extract_videos_from_archive(
                archive_name=zip_file,
                archive_path=archive_path,
                videos_dir=videos_dir,
                basename_counts=basename_counts,
                remaining_count=remaining_count,
                overwrite=args.overwrite,
            )
        )

    if len(extracted_videos) < args.max_videos:
        raise RuntimeError(
            f"Only extracted {len(extracted_videos)} videos from sequential archives, "
            f"which is fewer than requested max_videos={args.max_videos}."
        )

    selected_videos = resolve_captions_for_extracted(
        repo_id=args.repo_id,
        split=args.split,
        extracted_videos=extracted_videos,
    )
    save_label_file(selected_videos, label_file_path)
    save_manifest(selected_videos, manifest_path)
    print(f"Saved {len(selected_videos)} videos under {videos_dir}")
    print(f"Saved captions to {label_file_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()

