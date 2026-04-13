from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def make_line_plots(frame_counts: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    for (camera, day), group in frame_counts.groupby(["camera", "day"]):
        group = group.sort_values(["timestamp", "frame_file"]).reset_index(drop=True)

        plt.figure(figsize=(12, 5))
        plt.plot(group.index, group["person_count"])
        plt.xlabel("Frame index")
        plt.ylabel("People detected")
        plt.title(f"People per frame over time: {camera} {day}")
        plt.tight_layout()

        out_path = out_dir / f"{camera}_{day}_people_over_time.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Wrote {out_path}")


def make_bar_chart_daily_totals(frame_counts: pd.DataFrame, out_dir: Path) -> None:
    ensure_dir(out_dir)

    daily = (
        frame_counts.groupby(["camera", "day"], as_index=False)["person_count"]
        .sum()
        .rename(columns={"person_count": "total_people_detections"})
        .sort_values(["camera", "day"])
    )

    cameras = sorted(daily["camera"].unique())
    for camera in cameras:
        group = daily[daily["camera"] == camera].copy()

        plt.figure(figsize=(8, 5))
        plt.bar(group["day"], group["total_people_detections"])
        plt.xlabel("Day")
        plt.ylabel("Total people detections")
        plt.title(f"Daily total detections: {camera}")
        plt.tight_layout()

        out_path = out_dir / f"{camera}_daily_totals.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Wrote {out_path}")


def make_heatmaps(all_detections: pd.DataFrame, out_dir: Path, bins: int = 40) -> None:
    ensure_dir(out_dir)

    required_cols = {"center_x", "center_y"}
    if not required_cols.issubset(all_detections.columns):
        print("[WARN] center_x/center_y not found. Skipping heatmaps.")
        return

    for (camera, day), group in all_detections.groupby(["camera", "day"]):
        if group.empty:
            continue

        plt.figure(figsize=(8, 6))
        plt.hist2d(group["center_x"], group["center_y"], bins=bins)
        plt.gca().invert_yaxis()
        plt.xlabel("center_x")
        plt.ylabel("center_y")
        plt.title(f"Detection heatmap: {camera} {day}")
        plt.colorbar(label="Detection count")
        plt.tight_layout()

        out_path = out_dir / f"{camera}_{day}_heatmap.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from merged detection summary CSVs."
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=Path("data/summaries"),
        help="Directory containing summary CSVs.",
    )
    parser.add_argument(
        "--visuals-dir",
        type=Path,
        default=Path("data/visuals"),
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    all_detections = load_csv(args.summaries_dir / "all_detections.csv")
    frame_counts = load_csv(args.summaries_dir / "people_counts_by_frame.csv")

    make_line_plots(frame_counts, args.visuals_dir / "line_plots")
    make_bar_chart_daily_totals(frame_counts, args.visuals_dir / "bar_charts")
    make_heatmaps(all_detections, args.visuals_dir / "heatmaps")

    print("[DONE] Plot generation complete.")


if __name__ == "__main__":
    main()