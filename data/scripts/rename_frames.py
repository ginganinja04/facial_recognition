from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

# USAGE:
# python3 scripts/rename_standardize_frames.py data/raw_frames/<CAMERA_VIEW>/<dayX>


def parse_metadata(metadata_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def rename_frames(folder: Path, dry_run: bool = False) -> None:
    metadata_path = folder / "capture_metadata.txt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = parse_metadata(metadata_path)

    camera_name = metadata["camera_name"]
    day_label = metadata["day_label"]
    start_time_str = metadata["capture_start_time"]
    seconds_between_frames = int(metadata["seconds_between_frames"])

    start_time = datetime.strptime(start_time_str, "%H:%M:%S")

    frame_files = sorted(folder.glob("frame_*.png"))

    if not frame_files:
        print(f"No frame files found in {folder}")
        return

    pattern = re.compile(r"frame_(\d+)\.png$")

    for frame_file in frame_files:
        match = pattern.match(frame_file.name)
        if not match:
            continue

        frame_index = int(match.group(1))
        elapsed_seconds = (frame_index - 1) * seconds_between_frames
        frame_time = start_time + timedelta(seconds=elapsed_seconds)

        timestamp_str = frame_time.strftime("%H-%M-%S")
        new_name = f"{camera_name}_{day_label}_{timestamp_str}.png"
        new_path = folder / new_name

        if dry_run:
            print(f"{frame_file.name} -> {new_name}")
        else:
            frame_file.rename(new_path)

    print(f"Done renaming frames in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename frame_000001.png files into camera_day_time.png format."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to a folder like data/raw_frames/balcony/day1"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing files"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    rename_frames(folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()