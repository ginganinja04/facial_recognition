#!/usr/bin/env python3
"""
Visualize person paths with bounding boxes across multiple frames.
"""

import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np


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


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2, label=None):
    """Draw a bounding box on an image."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), font, font_scale, (255, 255, 255), font_thickness)

    return image


def create_path_visualization(path_details, output_path, max_frames_per_row=3):
    """Create a visualization of a person path across multiple frames."""

    if not path_details:
        print("[WARN] No path details provided")
        return

    # Load all images
    images = []
    valid_details = []

    for detail in path_details:
        image_path = Path(detail['frame_path'])
        if image_path.exists():
            image = cv2.imread(str(image_path))
            if image is not None:
                images.append(image)
                valid_details.append(detail)
            else:
                print("[WARN] Could not load image: {}".format(image_path))
        else:
            print("[WARN] Image not found: {}".format(image_path))

    if not images:
        print("[WARN] No valid images found for path")
        return

    # Update path_details to only include valid ones
    path_details = valid_details

    # Resize all images to the same dimensions
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    target_height = min(heights)  # Use smallest height
    target_width = min(widths)   # Use smallest width to avoid distortion

    resized_images = []
    for img in images:
        resized = cv2.resize(img, (target_width, target_height))
        resized_images.append(resized)

    # Draw bounding boxes on each image
    for i, (img, detail) in enumerate(zip(resized_images, path_details)):
        bbox = (detail['x1'], detail['y1'], detail['x2'], detail['y2'])
        label = "Frame {}".format(i + 1)
        resized_images[i] = draw_bounding_box(img, bbox, color=(0, 255, 0), thickness=2, label=label)

    # Arrange images vertically (simpler approach)
    if len(resized_images) == 1:
        final_image = resized_images[0]
    else:
        # Stack images vertically
        final_image = np.concatenate(resized_images, axis=0)

    # Add path information as text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    y_offset = 30

    # Path summary
    cameras = list(set([d['camera'] for d in path_details]))
    days = list(set([d['day'] for d in path_details]))
    timestamps = [d['timestamp'] for d in path_details]

    cv2.putText(final_image, "Person Path: {} frames".format(len(path_details)), (10, y_offset),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(final_image, "Cameras: {}".format(', '.join(cameras)), (10, y_offset + 35),
                font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(final_image, "Time: {} -> {}".format(timestamps[0], timestamps[-1]), (10, y_offset + 70),
                font, font_scale, (255, 255, 255), font_thickness)

    # Save the visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_image)
    print("[INFO] Saved path visualization to {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize person paths across multiple frames."
    )
    parser.add_argument(
        "--paths-csv",
        type=Path,
        default=Path("data/summaries/person_paths.csv"),
        help="Path to person paths CSV file."
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
        default=Path("data/visuals"),
        help="Output directory for path visualizations."
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=10,
        help="Maximum number of path visualizations to create."
    )
    parser.add_argument(
        "--min-path-length",
        type=int,
        default=3,
        help="Minimum path length to visualize."
    )

    args = parser.parse_args()

    # Load data
    print("[INFO] Loading paths CSV...")
    paths_df = pd.read_csv(args.paths_csv)
    print("[INFO] Loaded {} paths".format(len(paths_df)))

    print("[INFO] Loading detection data...")
    detections_df = load_detection_data(args.detections_dir)
    print("[INFO] Loaded {} detections".format(len(detections_df)))

    # Filter paths by length
    long_paths = paths_df[paths_df['path_length'] >= args.min_path_length]
    print("[INFO] Found {} paths with length >= {}".format(len(long_paths), args.min_path_length))

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process paths
    processed = 0
    for idx, path_row in long_paths.iterrows():
        if processed >= args.max_visualizations:
            break

        try:
            # Parse path details
            detection_ids = path_row['detection_ids'].split('|')
            cameras = path_row['cameras'].split('|')
            days = path_row['days'].split('|')
            frame_files = path_row['frame_files'].split('|')
            timestamps = path_row['timestamps'].split('|')

            path_details = []
            for i, detection_id in enumerate(detection_ids):
                # Parse detection ID: camera_day_frame_file_person_id
                parts = detection_id.split('_')
                if len(parts) >= 4:
                    camera = cameras[i] if i < len(cameras) else parts[0]
                    day = days[i] if i < len(days) else parts[1]
                    frame_file = frame_files[i] if i < len(frame_files) else '_'.join(parts[2:-1])
                    person_id = int(parts[-1])

                    # Find detection details
                    det_row = find_detection_row(detections_df, camera, day, frame_file, person_id)
                    if det_row is not None:
                        path_details.append({
                            'detection_id': detection_id,
                            'camera': camera,
                            'day': day,
                            'timestamp': timestamps[i] if i < len(timestamps) else det_row['timestamp'],
                            'frame_file': frame_file,
                            'person_id': person_id,
                            'x1': det_row['x1'],
                            'y1': det_row['y1'],
                            'x2': det_row['x2'],
                            'y2': det_row['y2'],
                            'frame_path': det_row['frame_path']
                        })

            if len(path_details) >= args.min_path_length:
                # Create visualization
                output_filename = "path_{:04d}_length_{}_score_{:.3f}.jpg".format(
                    processed, len(path_details), path_row['total_score'])
                output_path = args.output_dir / output_filename

                create_path_visualization(path_details, output_path)
                processed += 1

                if processed % 5 == 0:
                    print("[INFO] Processed {} path visualizations...".format(processed))

        except Exception as e:
            print("[WARN] Error processing path {}: {}".format(idx, e))
            continue

    print("[DONE] Created {} path visualizations in {}".format(processed, args.output_dir))


if __name__ == "__main__":
    main()