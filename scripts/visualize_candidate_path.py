from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def parse_time_bucket_start(label: str) -> int:
    # "18:10-18:19" -> 1090
    start = label.split("-")[0]
    h, m = map(int, start.split(":"))
    return h * 60 + m


def parse_timestamp_minutes(label: str) -> int:
    # "18:10:03" -> 1090
    h, m, _ = map(int, label.split(":"))
    return h * 60 + m


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty file: {path}")
    return df


def get_path_rows(candidate_paths: pd.DataFrame, path_id: str) -> pd.DataFrame:
    """
    Handle either:
      - candidate_path_id / step_num / time_bucket
      - path_id / step / time
    """
    if "candidate_path_id" in candidate_paths.columns:
        path_rows = candidate_paths[candidate_paths["candidate_path_id"] == path_id].copy()
    elif "path_id" in candidate_paths.columns:
        path_rows = candidate_paths[candidate_paths["path_id"] == path_id].copy()
    else:
        raise ValueError("No valid path id column found in candidate_paths.csv")

    if path_rows.empty:
        raise ValueError(f"No rows found for path id: {path_id}")

    return path_rows


def resolve_step_and_time_columns(path_rows: pd.DataFrame) -> tuple[str, str]:
    step_col = "step_num" if "step_num" in path_rows.columns else "step"

    if "time_bucket" in path_rows.columns:
        time_col = "time_bucket"
    elif "time" in path_rows.columns:
        time_col = "time"
    else:
        raise ValueError("No time column found in candidate_paths.csv")

    return step_col, time_col


def find_day_folder(raw_frames_dir: Path, camera: str, day: str) -> Path | None:
    """
    Handles folders like:
      data/raw_frames/bar_stage/day3 (04-10)
    when day == "day3"
    """
    camera_dir = raw_frames_dir / camera
    if not camera_dir.exists():
        return None

    for d in camera_dir.iterdir():
        if d.is_dir() and d.name.startswith(day):
            return d

    return None


def get_profile_row_for_step(
    step_row: pd.Series,
    profiles: pd.DataFrame,
) -> pd.Series | None:
    profile_id = str(step_row["profile_id"])
    prof = profiles[profiles["profile_id"] == profile_id].copy()
    if prof.empty:
        return None
    return prof.iloc[0]


def pick_best_detection_for_step(
    step_row: pd.Series,
    profile_row: pd.Series,
    all_detections: pd.DataFrame,
) -> pd.Series | None:
    """
    Pick a representative detection using:
    - camera/day/time from candidate_paths.csv
    - average center/size from pseudonymous_profiles.csv

    This keeps the correct step metadata while selecting a box
    that better matches the profile cluster.
    """
    camera = str(step_row["camera"])
    day = str(step_row["day"])
    time_value = step_row["time_value"]

    avg_x = float(profile_row["avg_center_x"])
    avg_y = float(profile_row["avg_center_y"])
    avg_area = float(profile_row["avg_bbox_area"])

    det = all_detections[
        (all_detections["camera"] == camera) &
        (all_detections["day"] == day)
    ].copy()

    if det.empty:
        return None

    det["timestamp_min"] = det["timestamp"].apply(parse_timestamp_minutes)

    if isinstance(time_value, str) and "-" in time_value:
        bucket_start = parse_time_bucket_start(time_value)
        bucket_end = bucket_start + 9
    else:
        bucket_start = int(time_value)
        bucket_end = int(time_value)

    det = det[
        (det["timestamp_min"] >= bucket_start) &
        (det["timestamp_min"] <= bucket_end)
    ].copy()

    if det.empty:
        return None

    # Distance to profile center
    det["center_dist"] = (
        ((det["center_x"] - avg_x) ** 2 + (det["center_y"] - avg_y) ** 2) ** 0.5
    )

    # Relative area mismatch
    det["area_ratio"] = det["bbox_area"].apply(
        lambda a: max(a, avg_area) / max(min(a, avg_area), 1.0)
    )

    # Hard reject obvious mismatches
    det = det[
        (det["confidence"] >= 0.40) &
        (det["area_ratio"] <= 2.5)
    ].copy()

    if det.empty:
        return None

    # Normalize terms
    max_center = max(float(det["center_dist"].max()), 1.0)
    max_ratio = max(float((det["area_ratio"] - 1.0).max()), 1.0)

    det["match_score"] = (
        0.55 * (det["center_dist"] / max_center) +
        0.25 * ((det["area_ratio"] - 1.0) / max_ratio) +
        0.20 * (1.0 - det["confidence"])
    )

    det = det.sort_values(["match_score", "confidence"], ascending=[True, False])
    return det.iloc[0]


def draw_box_and_label(
    image,
    det_row: pd.Series,
    title: str,
):
    out = image.copy()

    x1 = int(round(det_row["x1"]))
    y1 = int(round(det_row["y1"]))
    x2 = int(round(det_row["x2"]))
    y2 = int(round(det_row["y2"]))

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)

    label = (
        f"{title} | {det_row['camera']} {det_row['day']} | "
        f"{det_row['timestamp']} | conf={float(det_row['confidence']):.2f}"
    )

    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def resize_to_height(image, target_h: int):
    h, w = image.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_h))


def make_contact_sheet(images):
    if not images:
        raise ValueError("No images to combine.")

    target_h = 500
    resized = [resize_to_height(img, target_h) for img in images]
    return cv2.hconcat(resized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a candidate path by drawing one representative detection per step."
    )
    parser.add_argument(
        "--path-id",
        required=True,
        help="Example: path_125 or candidate_path_001",
    )
    parser.add_argument(
        "--candidate-paths",
        type=Path,
        default=Path("data/summaries/candidate_paths.csv"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("data/summaries/pseudonymous_profiles.csv"),
    )
    parser.add_argument(
        "--all-detections",
        type=Path,
        default=Path("data/summaries/all_detections.csv"),
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("data/raw_frames"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visuals/path_visualization.png"),
    )
    args = parser.parse_args()

    candidate_paths = load_csv(args.candidate_paths)
    profiles = load_csv(args.profiles)
    all_detections = load_csv(args.all_detections)

    path_rows = get_path_rows(candidate_paths, args.path_id)
    step_col, time_col = resolve_step_and_time_columns(path_rows)
    path_rows = path_rows.sort_values(step_col).copy()

    # Normalize time value into one column for easier downstream handling
    path_rows["time_value"] = path_rows[time_col]

    print("[INFO] Path rows selected:")
    cols_to_show = [c for c in [step_col, "profile_id", "camera", "day", "time_value"] if c in path_rows.columns]
    print(path_rows[cols_to_show].to_string(index=False))

    annotated_images = []

    for _, step_row in path_rows.iterrows():
        profile_id = str(step_row["profile_id"])
        camera = str(step_row["camera"])
        day = str(step_row["day"])

        profile_row = get_profile_row_for_step(step_row, profiles)
        if profile_row is None:
            print(f"[WARN] Profile not found in pseudonymous_profiles.csv: {profile_id}")
            continue

        best_det = pick_best_detection_for_step(step_row, profile_row, all_detections)
        if best_det is None:
            print(
                f"[WARN] No good matching detection found for "
                f"profile={profile_id}, camera={camera}, day={day}, time={step_row['time_value']}"
            )
            continue

        day_folder = find_day_folder(args.raw_frames_dir, camera, day)
        if day_folder is None:
            print(f"[WARN] Could not find matching day folder for camera={camera}, day={day}")
            continue

        frame_path = day_folder / str(best_det["frame_file"])
        if not frame_path.exists():
            print(f"[WARN] Frame file not found: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"[WARN] Could not read image: {frame_path}")
            continue

        title = f"{args.path_id} step {int(step_row[step_col])} | {profile_id}"
        annotated = draw_box_and_label(image, best_det, title)
        annotated_images.append(annotated)

    if not annotated_images:
        raise ValueError("No frames could be visualized for this path.")

    contact_sheet = make_contact_sheet(annotated_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), contact_sheet)

    print(f"[OK] Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()