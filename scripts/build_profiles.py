from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
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
        "bbox_width",
        "bbox_height",
        "center_x",
        "center_y",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def parse_day_folder(raw_frames_dir: Path, camera: str, day: str) -> Path | None:
    camera_dir = raw_frames_dir / camera
    if not camera_dir.exists():
        return None

    for d in camera_dir.iterdir():
        if d.is_dir() and d.name.startswith(day):
            return d

    return None


def resolve_frame_path(row: pd.Series, raw_frames_dir: Path) -> Path | None:
    # Prefer frame_path if it exists and is valid
    if "frame_path" in row and isinstance(row["frame_path"], str) and row["frame_path"].strip():
        p = Path(row["frame_path"])
        if p.exists():
            return p

    day_folder = parse_day_folder(raw_frames_dir, str(row["camera"]), str(row["day"]))
    if day_folder is None:
        return None

    frame_path = day_folder / str(row["frame_file"])
    return frame_path if frame_path.exists() else None


def add_time_features(df: pd.DataFrame, time_bucket_minutes: int) -> pd.DataFrame:
    df = df.copy()

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp_dt"]).copy()

    df["minutes_since_midnight"] = (
        df["timestamp_dt"].dt.hour * 60 + df["timestamp_dt"].dt.minute
    )

    df["time_bucket_start_min"] = (
        df["minutes_since_midnight"] // time_bucket_minutes
    ) * time_bucket_minutes

    df["time_bucket_label"] = df["time_bucket_start_min"].apply(
        lambda x: format_time_bucket(x, time_bucket_minutes)
    )

    return df


def format_time_bucket(start_min: int, width_min: int) -> str:
    end_min = start_min + width_min - 1

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

    df["x_bin"] = (df["center_x"] // spatial_bin_size).astype(int)
    df["y_bin"] = (df["center_y"] // spatial_bin_size).astype(int)
    df["zone_label"] = "zone_x" + df["x_bin"].astype(str) + "_y" + df["y_bin"].astype(str)

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


def compute_crop_features(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> dict[str, float]:
    h, w = image.shape[:2]

    xi1 = max(0, min(w - 1, int(round(x1))))
    yi1 = max(0, min(h - 1, int(round(y1))))
    xi2 = max(0, min(w, int(round(x2))))
    yi2 = max(0, min(h, int(round(y2))))

    if xi2 <= xi1 or yi2 <= yi1:
        return {
            "mean_h": np.nan,
            "mean_s": np.nan,
            "mean_v": np.nan,
        }

    crop = image[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return {
            "mean_h": np.nan,
            "mean_s": np.nan,
            "mean_v": np.nan,
        }

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_h = float(hsv[:, :, 0].mean())
    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())

    return {
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
    }


def add_appearance_features(
    df: pd.DataFrame,
    raw_frames_dir: Path,
) -> pd.DataFrame:
    df = df.copy()

    mean_h_values: list[float] = []
    mean_s_values: list[float] = []
    mean_v_values: list[float] = []
    aspect_ratio_values: list[float] = []

    print("[INFO] Computing appearance features from frame crops")

    for idx, row in df.iterrows():
        frame_path = resolve_frame_path(row, raw_frames_dir)

        aspect_ratio = float(row["bbox_width"]) / max(float(row["bbox_height"]), 1.0)
        aspect_ratio_values.append(aspect_ratio)

        if frame_path is None:
            mean_h_values.append(np.nan)
            mean_s_values.append(np.nan)
            mean_v_values.append(np.nan)
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            mean_h_values.append(np.nan)
            mean_s_values.append(np.nan)
            mean_v_values.append(np.nan)
            continue

        feats = compute_crop_features(
            image,
            x1=float(row["x1"]),
            y1=float(row["y1"]),
            x2=float(row["x2"]),
            y2=float(row["y2"]),
        )

        mean_h_values.append(feats["mean_h"])
        mean_s_values.append(feats["mean_s"])
        mean_v_values.append(feats["mean_v"])

        if (idx + 1) % 5000 == 0:
            print(f"[INFO] Processed appearance features for {idx + 1} detections")

    df["mean_h"] = mean_h_values
    df["mean_s"] = mean_s_values
    df["mean_v"] = mean_v_values
    df["aspect_ratio"] = aspect_ratio_values

    return df


def add_appearance_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill missing values with medians so binning does not fail
    for col in ["mean_h", "mean_s", "mean_v", "aspect_ratio"]:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    # Coarse but useful bins
    df["h_bin"] = (df["mean_h"] // 15).astype(int)       # hue in bins of 15
    df["s_bin"] = (df["mean_s"] // 32).astype(int)       # saturation bins
    df["v_bin"] = (df["mean_v"] // 32).astype(int)       # brightness bins

    df["aspect_bin"] = pd.cut(
        df["aspect_ratio"],
        bins=[-np.inf, 0.4, 0.7, 1.0, 1.4, np.inf],
        labels=["very_tall", "tall", "medium", "wide", "very_wide"],
    ).astype(str)

    df["appearance_label"] = (
        "h" + df["h_bin"].astype(str) +
        "_s" + df["s_bin"].astype(str) +
        "_v" + df["v_bin"].astype(str) +
        "_" + df["aspect_bin"].astype(str)
    )

    return df


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    More specific pseudonymous profiles using:
    - camera
    - day
    - zone
    - size
    - time bucket
    - appearance bins
    """
    group_cols = [
        "camera",
        "day",
        "zone_label",
        "size_label",
        "time_bucket_label",
        "appearance_label",
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
            avg_confidence=("confidence", "mean"),
            avg_mean_h=("mean_h", "mean"),
            avg_mean_s=("mean_s", "mean"),
            avg_mean_v=("mean_v", "mean"),
            avg_aspect_ratio=("aspect_ratio", "mean"),
        )
        .sort_values(
            ["camera", "day", "detection_count", "unique_frames"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )

    profiles["profile_id"] = [
        f"profile_{i:05d}" for i in range(1, len(profiles) + 1)
    ]

    profiles["pattern_description"] = profiles.apply(make_pattern_description, axis=1)

    profiles = profiles[
        [
            "profile_id",
            "camera",
            "day",
            "zone_label",
            "size_label",
            "time_bucket_label",
            "appearance_label",
            "detection_count",
            "unique_frames",
            "first_seen",
            "last_seen",
            "avg_center_x",
            "avg_center_y",
            "avg_bbox_area",
            "avg_confidence",
            "avg_mean_h",
            "avg_mean_s",
            "avg_mean_v",
            "avg_aspect_ratio",
            "pattern_description",
        ]
    ]

    return profiles


def make_pattern_description(row: pd.Series) -> str:
    return (
        f"Repeated {row['size_label']} detections in {row['zone_label']} "
        f"with appearance {row['appearance_label']} during {row['time_bucket_label']} "
        f"on {row['camera']} {row['day']}"
    )


def save_profiles(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build more specific pseudonymous profiles from merged detection data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/summaries/all_detections.csv"),
        help="Path to all_detections.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/summaries/pseudonymous_profiles.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("data/raw_frames"),
        help="Root folder containing raw frames",
    )
    parser.add_argument(
        "--spatial-bin-size",
        type=int,
        default=80,
        help="Bin size in pixels for grouping nearby detections",
    )
    parser.add_argument(
        "--size-quantiles",
        type=int,
        default=4,
        help="Number of bbox size groups to make",
    )
    parser.add_argument(
        "--time-bucket-minutes",
        type=int,
        default=3,
        help="Width of time bucket in minutes",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Discard detections below this confidence before profiling",
    )
    args = parser.parse_args()

    print(f"[INFO] Reading detections from {args.input}")
    df = load_detections(args.input)

    print(f"[INFO] Filtering detections with confidence >= {args.min_confidence}")
    df = df[df["confidence"] >= args.min_confidence].copy()
    print(f"[INFO] Remaining detections: {len(df)}")

    if df.empty:
        raise ValueError("No detections remain after confidence filtering.")

    print("[INFO] Adding time buckets")
    df = add_time_features(df, time_bucket_minutes=args.time_bucket_minutes)

    print("[INFO] Adding spatial and size bins")
    df = add_spatial_and_size_bins(
        df,
        spatial_bin_size=args.spatial_bin_size,
        size_quantiles=args.size_quantiles,
    )

    df = add_appearance_features(df, raw_frames_dir=args.raw_frames_dir)
    df = add_appearance_bins(df)

    print("[INFO] Building pseudonymous profiles")
    profiles = build_profiles(df)

    print(f"[INFO] Writing {len(profiles)} profiles to {args.output}")
    save_profiles(profiles, args.output)

    print("[DONE] Profile building complete.")


if __name__ == "__main__":
    main()