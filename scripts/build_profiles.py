from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_detections(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Detections file is empty: {path}")

    required = {
        "camera",
        "day",
        "frame_file",
        "timestamp",
        "bbox_area",
        "center_x",
        "center_y",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert timestamp like "18:00:03" to pandas time
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp_dt"]).copy()

    # Minutes since midnight
    df["minutes_since_midnight"] = (
        df["timestamp_dt"].dt.hour * 60 + df["timestamp_dt"].dt.minute
    )

    # 10-minute buckets
    df["time_bucket_start_min"] = (df["minutes_since_midnight"] // 10) * 10
    df["time_bucket_label"] = df["time_bucket_start_min"].apply(format_time_bucket)

    return df


def format_time_bucket(start_min: int) -> str:
    end_min = start_min + 9

    start_h = start_min // 60
    start_m = start_min % 60
    end_h = end_min // 60
    end_m = end_min % 60

    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"


def add_spatial_and_size_bins(
    df: pd.DataFrame,
    spatial_bin_size: int,
    size_quantiles: int,
) -> pd.DataFrame:
    df = df.copy()

    # Spatial bins
    df["x_bin"] = (df["center_x"] // spatial_bin_size).astype(int)
    df["y_bin"] = (df["center_y"] // spatial_bin_size).astype(int)
    df["zone_label"] = "zone_x" + df["x_bin"].astype(str) + "_y" + df["y_bin"].astype(str)

    # Size bins based on quantiles
    # duplicates="drop" handles repeated values safely
    df["size_bin"] = pd.qcut(
        df["bbox_area"],
        q=size_quantiles,
        labels=False,
        duplicates="drop",
    )

    size_names = {
        0: "small",
        1: "medium",
        2: "large",
        3: "xlarge",
        4: "xxlarge",
    }

    df["size_label"] = df["size_bin"].map(size_names).fillna("size_group")

    return df


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lightweight pseudonymous profiles by grouping detections
    that are similar in camera/day/location/size/time bucket.
    """
    group_cols = [
        "camera",
        "day",
        "zone_label",
        "size_label",
        "time_bucket_label",
    ]

    profiles = (
        df.groupby(group_cols, as_index=False)
        .agg(
            detection_count=("frame_file", "size"),
            unique_frames=("frame_file", "nunique"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            avg_center_x=("center_x", "mean"),
            avg_center_y=("center_y", "mean"),
            avg_bbox_area=("bbox_area", "mean"),
        )
        .sort_values(
            ["camera", "day", "detection_count", "unique_frames"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )

    profiles["profile_id"] = [
        f"profile_{i:04d}" for i in range(1, len(profiles) + 1)
    ]

    profiles["pattern_description"] = profiles.apply(make_pattern_description, axis=1)

    # Reorder columns
    profiles = profiles[
        [
            "profile_id",
            "camera",
            "day",
            "zone_label",
            "size_label",
            "time_bucket_label",
            "detection_count",
            "unique_frames",
            "first_seen",
            "last_seen",
            "avg_center_x",
            "avg_center_y",
            "avg_bbox_area",
            "pattern_description",
        ]
    ]

    return profiles


def make_pattern_description(row: pd.Series) -> str:
    return (
        f"Repeated {row['size_label']} detections in {row['zone_label']} "
        f"during {row['time_bucket_label']} on {row['camera']} {row['day']}"
    )


def save_profiles(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build lightweight pseudonymous profiles from merged detection data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("summaries/all_detections.csv"),
        help="Path to all_detections.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("summaries/pseudonymous_profiles.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--spatial-bin-size",
        type=int,
        default=100,
        help="Bin size in pixels for grouping nearby detections",
    )
    parser.add_argument(
        "--size-quantiles",
        type=int,
        default=3,
        help="Number of bbox size groups to make",
    )
    args = parser.parse_args()

    print(f"[INFO] Reading detections from {args.input}")
    df = load_detections(args.input)

    print("[INFO] Adding time buckets")
    df = add_time_features(df)

    print("[INFO] Adding spatial and size bins")
    df = add_spatial_and_size_bins(
        df,
        spatial_bin_size=args.spatial_bin_size,
        size_quantiles=args.size_quantiles,
    )

    print("[INFO] Building pseudonymous profiles")
    profiles = build_profiles(df)

    print(f"[INFO] Writing {len(profiles)} profiles to {args.output}")
    save_profiles(profiles, args.output)

    print("[DONE] Profile building complete.")


if __name__ == "__main__":
    main()