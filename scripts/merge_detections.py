from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def collect_detection_csvs(detections_dir: Path) -> list[Path]:
    return sorted(detections_dir.rglob("*_detections.csv"))


def build_people_counts_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(
            ["camera", "day", "frame_file", "timestamp"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "person_count"})
        .sort_values(["camera", "day", "timestamp", "frame_file"])
    )
    return counts


def build_camera_daily_summary(frame_counts: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame_counts.groupby(["camera", "day"], as_index=False)
        .agg(
            total_frames=("frame_file", "nunique"),
            total_people_detections=("person_count", "sum"),
            avg_people_per_frame=("person_count", "mean"),
            max_people_in_frame=("person_count", "max"),
        )
        .sort_values(["camera", "day"])
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-camera/day detection CSVs into summary CSVs."
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("data/detections"),
        help="Directory containing per-camera/day detection CSVs.",
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=Path("data/summaries"),
        help="Directory to write merged summary CSVs.",
    )
    args = parser.parse_args()

    if not args.detections_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {args.detections_dir}")

    args.summaries_dir.mkdir(parents=True, exist_ok=True)

    csv_files = collect_detection_csvs(args.detections_dir)
    if not csv_files:
        raise FileNotFoundError(f"No *_detections.csv files found under {args.detections_dir}")

    print(f"[INFO] Found {len(csv_files)} detection CSV files")

    dfs: list[pd.DataFrame] = []
    for csv_file in csv_files:
        print(f"[INFO] Reading {csv_file}")
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError("All detection CSV files were empty.")

    all_detections = pd.concat(dfs, ignore_index=True)
    all_detections = all_detections.sort_values(
        ["camera", "day", "timestamp", "frame_file", "person_id_in_frame"]
    )

    all_detections_path = args.summaries_dir / "all_detections.csv"
    all_detections.to_csv(all_detections_path, index=False)
    print(f"[OK] Wrote {all_detections_path}")

    people_counts_by_frame = build_people_counts_by_frame(all_detections)
    people_counts_path = args.summaries_dir / "people_counts_by_frame.csv"
    people_counts_by_frame.to_csv(people_counts_path, index=False)
    print(f"[OK] Wrote {people_counts_path}")

    camera_daily_summary = build_camera_daily_summary(people_counts_by_frame)
    daily_summary_path = args.summaries_dir / "camera_daily_summary.csv"
    camera_daily_summary.to_csv(daily_summary_path, index=False)
    print(f"[OK] Wrote {daily_summary_path}")

    print("[DONE] Merge complete.")


if __name__ == "__main__":
    main()