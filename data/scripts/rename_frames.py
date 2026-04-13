from __future__ import annotations

import argparse
import re
from pathlib import Path


# USAGE:
# python3 data/scripts/rename_frames.py "data/raw_frames/<CAMERA_VIEW>/<dayX>" --dry-run
# python3 data/scripts/rename_frames.py "data/raw_frames/<CAMERA_VIEW>/<dayX>"


def rename_frames(folder: Path, dry_run: bool = False) -> None:
    camera_name = folder.parent.name               # e.g., balcony
    day_label = folder.name.split("(")[0].strip() # e.g., day1 from "day1 (04-11)"

    frame_files = sorted(folder.glob("frame_*.jpg"))

    if not frame_files:
        print(f"No frame files found in {folder}")
        return

    pattern = re.compile(r"frame_(\d+)\.jpg$", re.IGNORECASE)

    for frame_file in frame_files:
        match = pattern.match(frame_file.name)
        if not match:
            continue

        frame_index = int(match.group(1))
        new_name = f"{camera_name}_{day_label}_{frame_index:04d}.jpg"
        new_path = folder / new_name

        if dry_run:
            print(f"{frame_file.name} -> {new_name}")
        else:
            frame_file.rename(new_path)

    print(f"Done renaming frames in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename frame_000001.jpg files into camera_day_XXXX.jpg format."
    )
    parser.add_argument(
        "folder",
        type=str,
        help='Path to a folder like "data/raw_frames/balcony/day1 (04-11)"'
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
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    rename_frames(folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()