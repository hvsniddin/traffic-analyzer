"""Extract random frames from every video in a folder."""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm", ".wmv"}


def find_videos(video_folder: Path) -> list[Path]:
    """Return supported video files directly inside a folder."""
    return sorted(
        path
        for path in video_folder.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_frames(video_path: Path, frame_count: int, output_folder: Path) -> int:
    """Write up to ``frame_count`` random readable frames from one video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine frame count: {video_path}")

    selected_frames = set(random.sample(range(total_frames), min(frame_count, total_frames)))
    output_folder.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    try:
        frame_index = 0
        while selected_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index in selected_frames:
                output_path = output_folder / (
                    f"{video_path.stem}_frame_{frame_index:06d}.jpg"
                )
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not write frame: {output_path}")
                selected_frames.remove(frame_index)
                saved_count += 1
            frame_index += 1
    finally:
        capture.release()

    return saved_count


def create_archive(output_folder: Path, archive_path: Path) -> None:
    """Create a ZIP archive containing the extracted images."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for image_path in sorted(output_folder.glob("*.jpg")):
            archive.write(image_path, image_path.relative_to(output_folder))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        type=Path,
        default=Path("data"),
        help="Folder containing videos (default: data).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Number of random frames to extract from each video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/random_frames"),
        help="Folder for extracted images (default: data/random_frames).",
    )
    parser.add_argument(
        "--archive",
        nargs="?",
        const="",
        metavar="PATH",
        help="Also create a ZIP archive; without PATH, derive it from --output.",
    )
    args = parser.parse_args()

    if args.num_frames < 1:
        parser.error("--num-frames must be at least 1")
    if not args.videos.is_dir():
        parser.error(f"Video folder does not exist: {args.videos}")
    if args.output.resolve() == args.videos.resolve():
        parser.error("--output must be different from --videos")

    videos = find_videos(args.videos)
    if not videos:
        parser.error(f"No supported videos found in: {args.videos}")

    total_saved = 0
    for video_path in videos:
        saved_count = extract_frames(video_path, args.num_frames, args.output)
        total_saved += saved_count
        if saved_count < args.num_frames:
            print(
                f"{video_path.name}: saved {saved_count} frames "
                f"(video has fewer than {args.num_frames} readable frames)"
            )
        else:
            print(f"{video_path.name}: saved {saved_count} frames")

    if args.archive is not None:
        archive_path = (
            args.output.with_suffix(".zip")
            if args.archive == ""
            else Path(args.archive)
        )
        if archive_path.resolve().parent == args.output.resolve():
            parser.error("Archive path must not be inside the output folder")
        create_archive(args.output, archive_path)
        print(f"Created archive: {archive_path}")

    print(f"Saved {total_saved} frames to {args.output}")


if __name__ == "__main__":
    main()