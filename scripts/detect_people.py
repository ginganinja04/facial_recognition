from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def parse_filename(image_path: Path) -> tuple[str, str, str]:
    """
    Parse filenames like:
        balcony_day2_18-00-00.jpg
        inside_bar_day1_19-38-09.jpg

    Returns:
        (camera, day, time_str)
    """
    stem = image_path.stem
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Filename does not match expected pattern: {image_path.name}")

    time_str = parts[-1]
    day = parts[-2]
    camera = "_".join(parts[:-2])

    if not day.startswith("day"):
        raise ValueError(f"Could not parse day from filename: {image_path.name}")

    return camera, day, time_str


def collect_images(raw_frames_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in raw_frames_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            images.append(path)
    return sorted(images)


def ensure_detection_dir(base_dir: Path, camera: str) -> Path:
    out_dir = base_dir / camera
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_detection_on_images(
    images: Iterable[Path],
    model: YOLO,
    conf_threshold: float,
) -> list[dict]:
    """
    Returns one row per detected person.
    """
    rows: list[dict] = []

    for idx, image_path in enumerate(images, start=1):
        try:
            camera, day, time_str = parse_filename(image_path)
        except ValueError as e:
            print(f"[WARN] Skipping {image_path}: {e}")
            continue

        print(f"[{idx}] Processing {image_path}")

        try:
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                verbose=False,
                device="cpu",
            )
        except Exception as e:
            print(f"[WARN] YOLO failed on {image_path}: {e}")
            continue

        if not results:
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        person_index_in_frame = 0

        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]

            if class_name != "person":
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            center_x = x1 + bbox_width / 2.0
            center_y = y1 + bbox_height / 2.0

            person_index_in_frame += 1

            rows.append(
                {
                    "camera": camera,
                    "day": day,
                    "frame_file": image_path.name,
                    "frame_path": str(image_path),
                    "timestamp": time_str.replace("-", ":"),
                    "person_id_in_frame": person_index_in_frame,
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "bbox_width": round(bbox_width, 2),
                    "bbox_height": round(bbox_height, 2),
                    "bbox_area": round(bbox_area, 2),
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                    "confidence": round(conf, 4),
                }
            )

    return rows


def write_grouped_csvs(rows: list[dict], detections_dir: Path) -> None:
    if not rows:
        print("[INFO] No person detections found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["camera", "day", "timestamp", "frame_file", "person_id_in_frame"])

    for (camera, day), group in df.groupby(["camera", "day"]):
        out_dir = ensure_detection_dir(detections_dir, camera)
        out_file = out_dir / f"{day}_detections.csv"
        group.to_csv(out_file, index=False)
        print(f"[OK] Wrote {len(group)} detections to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO person detection on raw frames and save one CSV per camera/day."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("data/raw_frames"),
        help="Path to raw frames directory.",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("data/detections"),
        help="Path to detections output directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model to use. yolov8n.pt is best for CPU.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    args = parser.parse_args()

    if not args.raw_frames_dir.exists():
        raise FileNotFoundError(f"Raw frames directory not found: {args.raw_frames_dir}")

    args.detections_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Collecting images from: {args.raw_frames_dir}")
    images = collect_images(args.raw_frames_dir)
    print(f"[INFO] Found {len(images)} images")

    rows = run_detection_on_images(
        images=images,
        model=model,
        conf_threshold=args.conf,
    )

    write_grouped_csvs(rows, args.detections_dir)
    print("[DONE] Detection finished.")


if __name__ == "__main__":
    main()