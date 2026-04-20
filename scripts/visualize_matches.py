#!/usr/bin/env python3
"""
Visualize candidate matches with bounding boxes.
"""

import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np


def parse_detection_id(detection_id):
    """Parse detection_id like 'balcony_day1_balcony_day1_18-00-42.jpg_1'"""
    parts = detection_id.split('_')
    if len(parts) < 4:
        raise ValueError("Invalid detection_id format: {}".format(detection_id))

    # Handle camera names with underscores
    camera_parts = []
    day_part = None
    frame_part = None
    person_id = None

    i = 0
    while i < len(parts):
        if parts[i].startswith('day'):
            day_part = parts[i]
            break
        camera_parts.append(parts[i])
        i += 1

    if day_part is None:
        raise ValueError("Could not find day in detection_id: {}".format(detection_id))

    camera = '_'.join(camera_parts)
    day = day_part

    # Find the frame file (ends with .jpg) and person_id
    remaining = parts[i+1:]
    frame_parts = []
    for j, part in enumerate(remaining):
        if part.endswith('.jpg'):
            frame_parts.append(part)
            person_id = int(remaining[j+1]) if j+1 < len(remaining) else 1
            break
        frame_parts.append(part)

    frame_file = '_'.join(frame_parts)

    return camera, day, frame_file, person_id


def load_detection_data(detections_dir):
    """Load all detection CSVs into a single DataFrame."""
    all_detections = []

    for csv_file in detections_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_detections.append(df)
        except Exception as e:
            print("[WARN] Failed to load {}: {}".format(csv_file, e))

    if not all_detections:
        raise ValueError("No detection files found")

    return pd.concat(all_detections, ignore_index=True)


def find_detection_row(detections_df, camera, day, frame_file, person_id):
    """Find the detection row for a specific detection."""
    mask = (
        (detections_df['camera'] == camera) &
        (detections_df['day'] == day) &
        (detections_df['frame_file'] == frame_file) &
        (detections_df['person_id_in_frame'] == person_id)
    )

    matches = detections_df[mask]
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        print("[WARN] Multiple matches found for {}_{}_{}_{}, using first".format(camera, day, frame_file, person_id))
        return matches.iloc[0]
    else:
        return matches.iloc[0]


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on an image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def create_match_visualization(image_a, image_b, detection_a, detection_b, match_info):
    """Create a side-by-side visualization of two matched detections."""

    # Get bounding boxes
    bbox_a = (detection_a['x1'], detection_a['y1'], detection_a['x2'], detection_a['y2'])
    bbox_b = (detection_b['x1'], detection_b['y1'], detection_b['x2'], detection_b['y2'])

    # Draw bounding boxes
    image_a = draw_bounding_box(image_a.copy(), bbox_a, color=(0, 255, 0))  # Green
    image_b = draw_bounding_box(image_b.copy(), bbox_b, color=(0, 255, 0))  # Green

    # Resize images to same height for side-by-side display
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    max_height = max(height_a, height_b)

    # Resize maintaining aspect ratio
    if height_a != max_height:
        scale_a = max_height / height_a
        new_width_a = int(width_a * scale_a)
        image_a = cv2.resize(image_a, (new_width_a, max_height))

    if height_b != max_height:
        scale_b = max_height / height_b
        new_width_b = int(width_b * scale_b)
        image_b = cv2.resize(image_b, (new_width_b, max_height))

    # Create combined image
    combined_width = image_a.shape[1] + image_b.shape[1]
    combined = np.zeros((max_height, combined_width, 3), dtype=np.uint8)

    # Place images side by side
    combined[:, :image_a.shape[1]] = image_a
    combined[:, image_a.shape[1]:] = image_b

    # Add separator line
    cv2.line(combined, (image_a.shape[1], 0), (image_a.shape[1], max_height),
             (255, 255, 255), 2)

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1

    # Left side info
    y_offset = 30
    cv2.putText(combined, "Camera: {}".format(detection_a['camera']), (10, y_offset),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, "Time: {}".format(detection_a['timestamp']), (10, y_offset + 25),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, "Person ID: {}".format(detection_a['person_id_in_frame']), (10, y_offset + 50),
                font, font_scale, (255, 255, 255), font_thickness)

    # Right side info
    right_x = image_a.shape[1] + 10
    cv2.putText(combined, "Camera: {}".format(detection_b['camera']), (right_x, y_offset),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, "Time: {}".format(detection_b['timestamp']), (right_x, y_offset + 25),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, "Person ID: {}".format(detection_b['person_id_in_frame']), (right_x, y_offset + 50),
                font, font_scale, (255, 255, 255), font_thickness)

    # Match scores at bottom
    bottom_y = max_height - 60
    cv2.putText(combined, "Face Similarity: {:.3f}".format(match_info['face_similarity']),
                (10, bottom_y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(combined, "Time Similarity: {:.3f}".format(match_info['time_similarity']),
                (10, bottom_y + 25), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(combined, "Combined Score: {:.3f}".format(match_info['combined_score']),
                (10, bottom_y + 50), font, font_scale, (0, 255, 255), font_thickness + 1)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Visualize candidate matches with bounding boxes on paired images."
    )
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=Path("data/summaries/candidate_links_with_faces.csv"),
        help="Path to candidate matches CSV file."
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("data/detections"),
        help="Path to detections directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/visuals/match_visualizations"),
        help="Output directory for visualization images."
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=10,
        help="Maximum number of match visualizations to create."
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum combined score to visualize."
    )

    args = parser.parse_args()

    # Load data
    print("[INFO] Loading matches CSV...")
    matches_df = pd.read_csv(args.matches_csv)
    print("[INFO] Loaded {} matches".format(len(matches_df)))

    print("[INFO] Loading detection data...")
    detections_df = load_detection_data(args.detections_dir)
    print("[INFO] Loaded {} detections".format(len(detections_df)))

    # Filter matches by score
    high_score_matches = matches_df[matches_df['combined_score'] >= args.min_score]
    print("[INFO] Found {} matches with score >= {}".format(len(high_score_matches), args.min_score))

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process matches
    processed = 0
    for idx, match in high_score_matches.iterrows():
        if processed >= args.max_visualizations:
            break

        try:
            # Parse detection IDs
            camera_a, day_a, frame_a, person_a = parse_detection_id(match['detection_id_a'])
            camera_b, day_b, frame_b, person_b = parse_detection_id(match['detection_id_b'])

            # Find detection rows
            det_a = find_detection_row(detections_df, camera_a, day_a, frame_a, person_a)
            det_b = find_detection_row(detections_df, camera_b, day_b, frame_b, person_b)

            if det_a is None or det_b is None:
                print("[WARN] Could not find detection data for match {}".format(idx))
                continue

            # Load images
            image_path_a = Path(det_a['frame_path'])
            image_path_b = Path(det_b['frame_path'])

            if not image_path_a.exists():
                print("[WARN] Image not found: {}".format(image_path_a))
                continue
            if not image_path_b.exists():
                print("[WARN] Image not found: {}".format(image_path_b))
                continue

            image_a = cv2.imread(str(image_path_a))
            image_b = cv2.imread(str(image_path_b))

            if image_a is None or image_b is None:
                print("[WARN] Could not load images for match {}".format(idx))
                continue

            # Create visualization
            visualization = create_match_visualization(image_a, image_b, det_a, det_b, match)

            # Save visualization
            output_filename = "match_{:04d}_score_{:.3f}.jpg".format(processed, match['combined_score'])
            output_path = args.output_dir / output_filename
            cv2.imwrite(str(output_path), visualization)

            processed += 1
            if processed % 5 == 0:
                print("[INFO] Processed {} visualizations...".format(processed))

        except Exception as e:
            print("[WARN] Error processing match {}: {}".format(idx, e))
            continue

    print("[DONE] Created {} match visualizations in {}".format(processed, args.output_dir))


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
Visualize candidate matches by showing paired images with bounding boxes.
"""

import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from typing import Tuple, Optional


def parse_detection_id(detection_id: str) -> Tuple[str, str, str, int]:
    """Parse detection_id like 'balcony_day1_balcony_day1_18-00-42.jpg_1'"""
    parts = detection_id.split('_')
    if len(parts) < 4:
        raise ValueError("Invalid detection_id format: {}".format(detection_id))

    # Handle camera names with underscores
    camera_parts = []
    day_part = None
    frame_part = None
    person_id = None

    i = 0
    while i < len(parts):
        if parts[i].startswith('day'):
            day_part = parts[i]
            break
        camera_parts.append(parts[i])
        i += 1

    if day_part is None:
        raise ValueError("Could not find day in detection_id: {}".format(detection_id))

    camera = '_'.join(camera_parts)
    day = day_part

    # Find the frame file (ends with .jpg) and person_id
    remaining = parts[i+1:]
    frame_parts = []
    for j, part in enumerate(remaining):
        if part.endswith('.jpg'):
            frame_parts.append(part)
            person_id = int(remaining[j+1]) if j+1 < len(remaining) else 1
            break
        frame_parts.append(part)

    frame_file = '_'.join(frame_parts)

    return camera, day, frame_file, person_id


def load_detection_data(detections_dir: Path) -> pd.DataFrame:
    """Load all detection CSVs into a single DataFrame."""
    all_detections = []

    for csv_file in detections_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_detections.append(df)
        except Exception as e:
            print("[WARN] Failed to load {}: {}".format(csv_file, e))

    if not all_detections:
        raise ValueError("No detection files found")

    return pd.concat(all_detections, ignore_index=True)


def find_detection_row(detections_df: pd.DataFrame, camera: str, day: str,
                      frame_file: str, person_id: int) -> Optional[pd.Series]:
    """Find the detection row for a specific detection."""
    mask = (
        (detections_df['camera'] == camera) &
        (detections_df['day'] == day) &
        (detections_df['frame_file'] == frame_file) &
        (detections_df['person_id_in_frame'] == person_id)
    )

    matches = detections_df[mask]
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        print(f"[WARN] Multiple matches found for {camera}_{day}_{frame_file}_{person_id}, using first")
        return matches.iloc[0]
    else:
        return matches.iloc[0]


def draw_bounding_box(image: np.ndarray, bbox: Tuple[float, float, float, float],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """Draw a bounding box on an image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def create_match_visualization(image_a: np.ndarray, image_b: np.ndarray,
                              detection_a: pd.Series, detection_b: pd.Series,
                              match_info: pd.Series) -> np.ndarray:
    """Create a side-by-side visualization of two matched detections."""

    # Get bounding boxes
    bbox_a = (detection_a['x1'], detection_a['y1'], detection_a['x2'], detection_a['y2'])
    bbox_b = (detection_b['x1'], detection_b['y1'], detection_b['x2'], detection_b['y2'])

    # Draw bounding boxes
    image_a = draw_bounding_box(image_a.copy(), bbox_a, color=(0, 255, 0))  # Green
    image_b = draw_bounding_box(image_b.copy(), bbox_b, color=(0, 255, 0))  # Green

    # Resize images to same height for side-by-side display
    height_a, width_a = image_a.shape[:2]
    height_b, width_b = image_b.shape[:2]
    max_height = max(height_a, height_b)

    # Resize maintaining aspect ratio
    if height_a != max_height:
        scale_a = max_height / height_a
        new_width_a = int(width_a * scale_a)
        image_a = cv2.resize(image_a, (new_width_a, max_height))

    if height_b != max_height:
        scale_b = max_height / height_b
        new_width_b = int(width_b * scale_b)
        image_b = cv2.resize(image_b, (new_width_b, max_height))

    # Create combined image
    combined_width = image_a.shape[1] + image_b.shape[1]
    combined = np.zeros((max_height, combined_width, 3), dtype=np.uint8)

    # Place images side by side
    combined[:, :image_a.shape[1]] = image_a
    combined[:, image_a.shape[1]:] = image_b

    # Add separator line
    cv2.line(combined, (image_a.shape[1], 0), (image_a.shape[1], max_height),
             (255, 255, 255), 2)

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1

    # Left side info
    y_offset = 30
    cv2.putText(combined, f"Camera: {detection_a['camera']}", (10, y_offset),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, f"Time: {detection_a['timestamp']}", (10, y_offset + 25),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, f"Person ID: {detection_a['person_id_in_frame']}", (10, y_offset + 50),
                font, font_scale, (255, 255, 255), font_thickness)

    # Right side info
    right_x = image_a.shape[1] + 10
    cv2.putText(combined, f"Camera: {detection_b['camera']}", (right_x, y_offset),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, f"Time: {detection_b['timestamp']}", (right_x, y_offset + 25),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined, f"Person ID: {detection_b['person_id_in_frame']}", (right_x, y_offset + 50),
                font, font_scale, (255, 255, 255), font_thickness)

    # Match scores at bottom
    bottom_y = max_height - 60
    cv2.putText(combined, f"Face Similarity: {match_info['face_similarity']:.3f}",
                (10, bottom_y), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(combined, f"Time Similarity: {match_info['time_similarity']:.3f}",
                (10, bottom_y + 25), font, font_scale, (255, 255, 0), font_thickness)
    cv2.putText(combined, f"Combined Score: {match_info['combined_score']:.3f}",
                (10, bottom_y + 50), font, font_scale, (0, 255, 255), font_thickness + 1)

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Visualize candidate matches with bounding boxes on paired images."
    )
    parser.add_argument(
        "--matches-csv",
        type=Path,
        default=Path("data/summaries/candidate_links_with_faces.csv"),
        help="Path to candidate matches CSV file."
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("data/detections"),
        help="Path to detections directory."
    )
    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        default=Path("data/raw_frames"),
        help="Path to raw frames directory."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/visuals/match_visualizations"),
        help="Output directory for visualization images."
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=50,
        help="Maximum number of match visualizations to create."
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.7,
        help="Minimum combined score to visualize."
    )

    args = parser.parse_args()

    # Load data
    print("[INFO] Loading matches CSV...")
    matches_df = pd.read_csv(args.matches_csv)
    print(f"[INFO] Loaded {len(matches_df)} matches")

    print("[INFO] Loading detection data...")
    detections_df = load_detection_data(args.detections_dir)
    print(f"[INFO] Loaded {len(detections_df)} detections")

    # Filter matches by score
    high_score_matches = matches_df[matches_df['combined_score'] >= args.min_score]
    print(f"[INFO] Found {len(high_score_matches)} matches with score >= {args.min_score}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process matches
    processed = 0
    for idx, match in high_score_matches.iterrows():
        if processed >= args.max_visualizations:
            break

        try:
            # Parse detection IDs
            camera_a, day_a, frame_a, person_a = parse_detection_id(match['detection_id_a'])
            camera_b, day_b, frame_b, person_b = parse_detection_id(match['detection_id_b'])

            # Find detection rows
            det_a = find_detection_row(detections_df, camera_a, day_a, frame_a, person_a)
            det_b = find_detection_row(detections_df, camera_b, day_b, frame_b, person_b)

            if det_a is None or det_b is None:
                print(f"[WARN] Could not find detection data for match {idx}")
                continue

            # Load images
            image_path_a = Path(det_a['frame_path'])
            image_path_b = Path(det_b['frame_path'])

            if not image_path_a.exists():
                print(f"[WARN] Image not found: {image_path_a}")
                continue
            if not image_path_b.exists():
                print(f"[WARN] Image not found: {image_path_b}")
                continue

            image_a = cv2.imread(str(image_path_a))
            image_b = cv2.imread(str(image_path_b))

            if image_a is None or image_b is None:
                print(f"[WARN] Could not load images for match {idx}")
                continue

            # Create visualization
            visualization = create_match_visualization(image_a, image_b, det_a, det_b, match)

            # Save visualization
            output_filename = "match_{:04d}_score_{:.3f}.jpg".format(processed, match['combined_score'])
            output_path = args.output_dir / output_filename
            cv2.imwrite(str(output_path), visualization)

            processed += 1
            if processed % 10 == 0:
                print("[INFO] Processed {} visualizations...".format(processed))

        except Exception as e:
            print(f"[WARN] Error processing match {idx}: {e}")
            continue

    print("[DONE] Created {} match visualizations in {}".format(processed, args.output_dir))


if __name__ == "__main__":
    main()