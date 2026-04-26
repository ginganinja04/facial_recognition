"""
Purpose:
--------
This script performs:
1. Person detection (YOLO)
2. Frame-to-frame tracking (Hungarian matching)
3. Cross-camera identity linking (global profiles)

It processes raw image frames and produces structured CSV outputs
containing detections, tracking IDs, and cross-camera IDs.

Pipeline Role:
--------------
This is the CORE perception + tracking stage of the system.

Input:
- Raw image frames (organized by camera)
- YOLO model

Output:
- CSV files with:
    - bounding boxes
    - per-frame track IDs
    - global IDs (cross-camera identity)
    - spatial + confidence metadata

Key Ideas:
----------
- Tracks are maintained per camera stream
- Hungarian matching assigns detections → existing tracks
- Simple motion model (velocity smoothing + prediction)
- Appearance features enable cross-camera matching
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def parse_filename(image_path: Path) -> tuple[str, str, str]:
    """
    Parse filenames like:
        balcony_18-00-00.jpg
        inside_bar_19-38-09.jpg
    
    Assumption:
        Filenames follow: camera_timestamp.jpg

    Returns:
        (camera, day, time_str)
    """
    stem = image_path.stem
    parts = stem.split("_")

    if len(parts) < 2:
        raise ValueError(f"Filename does not match expected pattern: {image_path.name}")

    time_str = parts[-1]
    camera = "_".join(parts[:-1])

    # since day is removed, just set a default
    day = "day1"

    return camera, day, time_str


def collect_images(raw_frames_dir: Path) -> list[Path]:
    """
    Recursively collect all image files from raw frames directory.
    """
    images: list[Path] = []
    for path in raw_frames_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            images.append(path)
    return sorted(images)


def ensure_detection_dir(base_dir: Path, camera: str) -> Path:
    """
    Ensure output directory exists per camera.
    """
    out_dir = base_dir / camera
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



def run_detection_on_images(images, model, conf_threshold):
    """
    Run YOLO person detection, assign local track IDs, and attempt cross-camera matching.

    This function is the main processing loop for the detection/tracking stage.
    It takes a sorted list of image frames, runs YOLO on each frame, keeps only
    detections classified as "person", then links detections across nearby frames
    using a lightweight tracker.

    Tracking Approach:
    ------------------
    Each active track stores:
    - last bounding box
    - center point
    - smoothed velocity
    - appearance feature
    - missed-frame count

    For every new frame, the code builds a cost matrix comparing each existing
    track to each new detection. The Hungarian algorithm is then used to choose
    the best one-to-one assignment between tracks and detections.

    Match Cost Uses:
    ----------------
    The matching score combines:
    - predicted center distance
    - bounding-box size difference
    - IoU overlap
    - HSV appearance distance

    Lower match score means the detection is more likely to belong to that track.

    Cross-Camera Matching:
    ----------------------
    When a detection cannot be matched to an existing local track, a new track is
    created. Before assigning a new global identity, the detection's appearance
    feature is compared against stored global profiles from other cameras on the
    same day. If the appearance distance is below the global threshold, the
    detection reuses that global ID. Otherwise, a new global ID is created.

    Output:
    -------
    Returns a list of dictionaries. Each dictionary becomes one CSV row and
    contains frame metadata, bounding-box coordinates, local track ID, global ID,
    confidence score, and matching scores.
    """

    # Stores all final detection rows that will later be written to CSV.
    rows = []

    # Tracks currently active in the current camera/day stream.
    # These are reset when the input switches to a different camera or day.
    active_tracks: list[Track] = []
    
    # Local track IDs are only used within the current run/camera sequence.
    next_track_id = 1
    
    # If a track is not matched for this many frames, it is removed.
    max_misses = 8
    
    # Used to detect when the loop moves from one camera/day group to another.
    prev_camera_day = None
    
    # Maximum allowed local matching cost.
    # Matches with a score above this are treated as unreliable and rejected.
    MAX_MATCH_SCORE = 0.65
    
    # Global profiles store appearance features used for cross-camera identity matching.
    global_profiles: list[GlobalProfile] = []
    
    # Global IDs are meant to represent possible same-person matches across cameras.
    next_global_id = 1
    
    # Maximum appearance distance allowed for cross-camera global matching.
    GLOBAL_MATCH_THRESHOLD = 0.65

    # Process each frame sequentially:
    # - detect people
    # - match to existing tracks
    # - update or create tracks
    # - assign global IDs
    for frame_index, image_path in enumerate(images, start=1):
        try:
            camera, day, time_str = parse_filename(image_path)

            current_camera_day = (camera, day)

            # If we switch to a new camera/day, reset active tracks
            # because tracking is only valid within a single stream.
            if prev_camera_day is not None and current_camera_day != prev_camera_day:
                active_tracks = []
            prev_camera_day = current_camera_day

        except ValueError as e:
            print(f"[WARN] Skipping {image_path}: {e}")
            continue

        print(f"[{frame_index}] Processing {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        frame_h, frame_w = image.shape[:2]

        # Run YOLO detection on the current frame
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

        # Build detection objects for this frame.
        # Each detection stores spatial info + appearance features.
        detections = []
        person_index_in_frame = 0

        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = model.names[cls_id]
            if class_name != "person":
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            bbox = (x1, y1, x2, y2)

            person_index_in_frame += 1
            feat = extract_appearance_feature(image, bbox)

            detections.append({
                "bbox": bbox,
                "feat": feat,
                "confidence": conf,
                "person_id_in_frame": person_index_in_frame,
                "track_id": None,
                "global_id": None,
                "match_score": None,
                "global_match_score": None,
            })

        assigned_track_ids = set()
        assigned_det_indices = set()

        # ---------- Local track-to-detection matching ----------
        # Build a cost matrix comparing existing tracks to new detections.
        # The Hungarian algorithm finds the optimal 1-to-1 assignment
        # that minimizes total matching cost.
        num_tracks = len(active_tracks)
        num_dets = len(detections)

        if num_tracks > 0 and num_dets > 0:
            cost_matrix = np.full((num_tracks, num_dets), 1e6, dtype=np.float32)

            for ti, track in enumerate(active_tracks):
                for di, det in enumerate(detections):
                    iou = compute_iou(track.bbox, det["bbox"])
                    dist = normalized_predicted_distance(track, det["bbox"], frame_w, frame_h)

                    # Gating: discard impossible matches early
                    # - too far apart (distance too large)
                    # - no overlap AND still not close enough
                    if dist > 0.80:
                        continue
                    if iou < 0.001 and dist > 0.20:
                        continue

                    score = match_score(track, det["bbox"], det["feat"], frame_w, frame_h)
                    cost_matrix[ti, di] = score

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for ti, di in zip(row_ind, col_ind):
                score = float(cost_matrix[ti, di])

                # reject impossible / weak matches
                if score >= 1e6 or score > MAX_MATCH_SCORE:
                    continue
                if ti in assigned_track_ids or di in assigned_det_indices:
                    continue

                track = active_tracks[ti]
                det = detections[di]

                new_cx, new_cy = bbox_center(det["bbox"])

                # Update velocity using exponential smoothing to reduce jitter
                # and improve prediction stability across frames
                measured_vx = new_cx - track.cx
                measured_vy = new_cy - track.cy

                track.vx = 0.7 * track.vx + 0.3 * measured_vx
                track.vy = 0.7 * track.vy + 0.3 * measured_vy

                # update state
                track.cx = new_cx
                track.cy = new_cy
                track.bbox = det["bbox"]
                track.appearance = det["feat"]
                track.last_frame_index = frame_index
                track.misses = 0

                det["track_id"] = track.track_id
                det["global_id"] = track.global_id
                det["match_score"] = round(score, 4)

                assigned_track_ids.add(ti)
                assigned_det_indices.add(di)

        # ---------- Create new tracks ----------
        # Any detection not matched to an existing track becomes a new track.
        # Before assigning a new global ID, attempt cross-camera matching
        # using appearance similarity.
        for di, det in enumerate(detections):
            if di in assigned_det_indices:
                continue

            if det["confidence"] < 0.4:
                continue

            global_id, global_score, next_global_id = assign_global_id(
                det_feat=det["feat"],
                camera=camera,
                day=day,
                time_str=time_str,
                frame_index=frame_index,
                global_profiles=global_profiles,
                next_global_id=next_global_id,
                threshold=GLOBAL_MATCH_THRESHOLD,
            )
            
            det["track_id"] = next_track_id
            det["global_id"] = global_id
            det["match_score"] = None
            det["global_match_score"] = global_score

            cx, cy = bbox_center(det["bbox"])

            active_tracks.append(
                Track(
                    track_id=next_track_id,
                    global_id=global_id,
                    bbox=det["bbox"],
                    appearance=det["feat"],
                    last_frame_index=frame_index,
                    cx=cx,
                    cy=cy,
                    vx=0.0,
                    vy=0.0,
                    misses=0,
                )
            )
            next_track_id += 1

        # ---------- Handle unmatched tracks ----------
        # Tracks not matched in this frame are marked as missed.
        # Tracks exceeding max_misses are removed.
        for ti, track in enumerate(active_tracks):
            if ti not in assigned_track_ids:
                if track.last_frame_index != frame_index:
                    track.misses += 1

        active_tracks = [t for t in active_tracks if t.misses <= max_misses]

        # ---------- Save detections ----------
        # Convert detections into structured rows for CSV output,
        # including spatial features, IDs, and matching scores.
        for det in detections:
            if det["track_id"] is None:
                continue

            x1, y1, x2, y2 = det["bbox"]
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            center_x = x1 + bbox_width / 2.0
            center_y = y1 + bbox_height / 2.0

            rows.append({
                "camera": camera,
                "day": day,
                "frame_file": image_path.name,
                "frame_path": str(image_path),
                "timestamp": time_str.replace("-", ":"),
                "person_id_in_frame": det["person_id_in_frame"],
                "track_id": det["track_id"],
                "global_id": det["global_id"],
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "bbox_width": round(bbox_width, 2),
                "bbox_height": round(bbox_height, 2),
                "bbox_area": round(bbox_area, 2),
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "confidence": round(det["confidence"], 4),
                "match_score": det["match_score"],
                "global_match_score": det["global_match_score"],
            })

    return rows


def write_grouped_csvs(rows: list[dict], detections_dir: Path) -> None:
    """
    Save results grouped by camera and day.

    Output structure:
    detections/
        camera_1/
            day1_detections.csv
        camera_2/
            day1_detections.csv
    """
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

# -------------------------------
# Data structures
# -------------------------------

@dataclass
class Track:
    """
    Represents a single tracked object within a camera stream.

    Stores:
    - spatial state (bbox, center)
    - motion (velocity)
    - appearance feature
    - tracking + global IDs
    - last seen frame
    """
    track_id: int
    global_id: int
    bbox: tuple[float, float, float, float]
    appearance: np.ndarray
    last_frame_index: int
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    misses: int = 0

@dataclass
class GlobalProfile:
    """
    Represents a cross-camera identity.

    Used to link the same person across different cameras
    using appearance similarity.
    """
    global_id: int
    camera: str
    day: str
    appearance: np.ndarray
    last_frame_index: int
    last_timestamp: str

def assign_global_id(
    det_feat: np.ndarray,
    camera: str,
    day: str,
    time_str: str,
    frame_index: int,
    global_profiles: list[GlobalProfile],
    next_global_id: int,
    threshold: float,
) -> tuple[int, float | None, int]:
    """
    Assign or create a global identity for a detection.

    Logic:
    - Compare detection appearance to existing profiles
    - Only match across different cameras
    - If similarity is high → reuse global_id
    - Otherwise → create new identity

    Also updates profile appearance over time.
    """

    best_profile = None
    best_score = float("inf")

    for profile in global_profiles:
        # only compare across different cameras
        if profile.camera == camera:
            continue

        # optional: keep same day only
        if profile.day != day:
            continue

        score = appearance_distance(profile.appearance, det_feat)

        if score < best_score:
            best_score = score
            best_profile = profile

    if best_profile is not None and best_score <= threshold:
        # update profile appearance slowly
        best_profile.appearance = 0.8 * best_profile.appearance + 0.2 * det_feat
        best_profile.camera = camera
        best_profile.day = day
        best_profile.last_frame_index = frame_index
        best_profile.last_timestamp = time_str

        return best_profile.global_id, round(best_score, 4), next_global_id

    new_profile = GlobalProfile(
        global_id=next_global_id,
        camera=camera,
        day=day,
        appearance=det_feat,
        last_frame_index=frame_index,
        last_timestamp=time_str,
    )

    global_profiles.append(new_profile)

    return next_global_id, None, next_global_id + 1

# -------------------------------
# Matching + similarity functions
# -------------------------------

def match_score(track: Track, det_bbox, det_feat, frame_w, frame_h) -> float:
    """
    Combined cost function for track matching.

    Uses weighted combination of:
    - motion distance (predicted vs actual)
    - size difference
    - IoU overlap
    - appearance similarity

    Lower score = better match
    """
    dist = normalized_predicted_distance(track, det_bbox, frame_w, frame_h)
    size_diff = size_difference(track.bbox, det_bbox)
    iou = compute_iou(track.bbox, det_bbox)
    app_dist = appearance_distance(track.appearance, det_feat)

    score = (
        0.40 * dist +
        0.15 * size_diff +
        0.25 * (1.0 - iou) +
        0.20 * app_dist
    )
    return score

def compute_iou(box_a, box_b) -> float:
    """
    Intersection over Union between two boxes.
    Measures spatial overlap.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def extract_appearance_feature(image, bbox):
    """
    Extract simple appearance descriptor.

    Method:
    - Crop bounding box
    - Convert to HSV
    - Compute histograms (upper/lower body)
    - Normalize and concatenate

    Result:
    - Lightweight feature vector for identity matching
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(64, dtype=np.float32)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(64, dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    crop_h = hsv.shape[0]
    upper = hsv[: max(1, crop_h // 2), :]
    lower = hsv[max(1, crop_h // 2):, :]

    hist_upper = cv2.calcHist([upper], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
    hist_lower = cv2.calcHist([lower], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()

    hist_upper = hist_upper / (np.linalg.norm(hist_upper) + 1e-8)
    hist_lower = hist_lower / (np.linalg.norm(hist_lower) + 1e-8)

    feat = np.concatenate([hist_upper, hist_lower]).astype(np.float32)
    return feat


def appearance_distance(feat_a, feat_b) -> float:
    """
    Euclidean distance between appearance vectors.
    """
    return float(np.linalg.norm(feat_a - feat_b))


def normalized_center_distance(box_a, box_b, frame_w, frame_h) -> float:
    """
    Distance between box centers normalized by frame size.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    dx = (acx - bcx) / max(frame_w, 1)
    dy = (acy - bcy) / max(frame_h, 1)
    return float(math.sqrt(dx * dx + dy * dy))


def bbox_center(box):
    """
    Compute center of bounding box.
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def predicted_center(track: Track):
    """
    Predict next position using simple linear motion model.
    """
    return track.cx + 1.2 * track.vx, track.cy + 1.2 * track.vy


def normalized_predicted_distance(track: Track, det_bbox, frame_w, frame_h) -> float:
    """
    Distance between predicted track position and detection.
    """
    pred_cx, pred_cy = predicted_center(track)
    det_cx, det_cy = bbox_center(det_bbox)

    dx = (pred_cx - det_cx) / max(frame_w, 1)
    dy = (pred_cy - det_cy) / max(frame_h, 1)
    return float(math.sqrt(dx * dx + dy * dy))


def size_difference(box_a, box_b) -> float:
    """
    Relative size difference between two bounding boxes.
    """
    aw = max(1.0, box_a[2] - box_a[0])
    ah = max(1.0, box_a[3] - box_a[1])
    bw = max(1.0, box_b[2] - box_b[0])
    bh = max(1.0, box_b[3] - box_b[1])

    w_diff = abs(aw - bw) / max(aw, bw)
    h_diff = abs(ah - bh) / max(ah, bh)
    return float((w_diff + h_diff) / 2.0)

def main() -> None:
    """
    Entry point:
    1. Parse arguments
    2. Load YOLO model
    3. Collect frames
    4. Run detection + tracking pipeline
    5. Save results
    """
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