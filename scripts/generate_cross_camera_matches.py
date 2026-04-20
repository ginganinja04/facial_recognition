#!/usr/bin/env python3
"""
Generate cross-camera matches using time and size similarities.
This is a Python implementation that can work without face embeddings.
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np


def parse_timestamp(timestamp_str):
    """Parse timestamp string like '18:00:42' into minutes since midnight."""
    try:
        time_obj = datetime.strptime(timestamp_str, '%H:%M:%S')
        return time_obj.hour * 60 + time_obj.minute + time_obj.second / 60.0
    except ValueError:
        try:
            time_obj = datetime.strptime(timestamp_str, '%H:%M')
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            return 0


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


def generate_cross_camera_matches(detections_df, max_time_diff=30, min_combined_score=0.5):
    """Generate matches between different cameras using time and size similarities."""
    matches = []

    # Group detections by camera+day
    camera_groups = {}
    for _, det in detections_df.iterrows():
        key = "{}_{}".format(det['camera'], det['day'])
        if key not in camera_groups:
            camera_groups[key] = []
        camera_groups[key].append(det)

    print("[INFO] Found {} camera-day groups".format(len(camera_groups)))

    # Compare between different camera groups
    camera_keys = list(camera_groups.keys())
    total_comparisons = 0

    for i in range(len(camera_keys)):
        for j in range(i + 1, len(camera_keys)):
            cam_a_key = camera_keys[i]
            cam_b_key = camera_keys[j]

            cam_a_dets = camera_groups[cam_a_key]
            cam_b_dets = camera_groups[cam_b_key]

            print("[INFO] Comparing {} ({} detections) vs {} ({} detections)".format(
                cam_a_key, len(cam_a_dets), cam_b_key, len(cam_b_dets)))

            # Create time-sorted lists for efficient comparison
            cam_a_sorted = sorted(cam_a_dets, key=lambda x: parse_timestamp(x['timestamp']))
            cam_b_sorted = sorted(cam_b_dets, key=lambda x: parse_timestamp(x['timestamp']))

            # Compare detections within time window
            matches_found = 0
            for det_a in cam_a_sorted:
                time_a = parse_timestamp(det_a['timestamp'])

                for det_b in cam_b_sorted:
                    time_b = parse_timestamp(det_b['timestamp'])
                    time_diff = abs(time_a - time_b)

                    if time_diff > max_time_diff:
                        if time_b > time_a + max_time_diff:
                            break  # b is too late, skip remaining
                        continue  # b is too early, check next

                    # Calculate similarities
                    time_similarity = max(0.0, 1.0 - time_diff / max_time_diff)

                    # Size similarity (if bbox_area exists)
                    if 'bbox_area' in det_a and 'bbox_area' in det_b:
                        area_a = det_a['bbox_area']
                        area_b = det_b['bbox_area']
                        if area_a > 0 and area_b > 0:
                            size_diff = abs(area_a - area_b)
                            max_area = max(area_a, area_b)
                            size_similarity = max(0.0, 1.0 - size_diff / max_area)
                        else:
                            size_similarity = 0.5  # neutral
                    else:
                        size_similarity = 0.5

                    # Face similarity (placeholder - will be 0 until face recognition works)
                    face_similarity = 0.0

                    # Combined score (without face for now)
                    combined_score = 0.3 * time_similarity + 0.4 * size_similarity + 0.3 * face_similarity

                    if combined_score >= min_combined_score:
                        detection_id_a = "{}_{}_{}_{}".format(
                            det_a['camera'], det_a['day'], det_a['frame_file'], det_a['person_id_in_frame'])
                        detection_id_b = "{}_{}_{}_{}".format(
                            det_b['camera'], det_b['day'], det_b['frame_file'], det_b['person_id_in_frame'])

                        match = {
                            'detection_id_a': detection_id_a,
                            'detection_id_b': detection_id_b,
                            'camera_a': det_a['camera'],
                            'camera_b': det_b['camera'],
                            'day_a': det_a['day'],
                            'day_b': det_b['day'],
                            'timestamp_a': det_a['timestamp'],
                            'timestamp_b': det_b['timestamp'],
                            'face_similarity': face_similarity,
                            'time_similarity': time_similarity,
                            'size_similarity': size_similarity,
                            'combined_score': combined_score,
                            'time_diff_minutes': time_diff,
                            'size_diff': abs(det_a.get('bbox_area', 0) - det_b.get('bbox_area', 0))
                        }
                        matches.append(match)
                        matches_found += 1

            print("[INFO] Found {} matches between {} and {}".format(matches_found, cam_a_key, cam_b_key))
            total_comparisons += matches_found

    print("[INFO] Total cross-camera matches found: {}".format(total_comparisons))
    return matches


def save_matches_to_csv(matches, output_file):
    """Save matches to CSV file."""
    df = pd.DataFrame(matches)
    df.to_csv(output_file, index=False)
    print("[INFO] Saved {} matches to {}".format(len(matches), output_file))


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-camera matches using time and size similarities."
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("data/detections"),
        help="Path to detections directory."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/summaries/cross_camera_matches.csv"),
        help="Output CSV file for cross-camera matches."
    )
    parser.add_argument(
        "--max-time-diff",
        type=int,
        default=30,
        help="Maximum time difference in minutes for matches."
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Minimum combined score for matches."
    )

    args = parser.parse_args()

    # Load detection data
    print("[INFO] Loading detection data...")
    detections_df = load_detection_data(args.detections_dir)
    print("[INFO] Loaded {} detections".format(len(detections_df)))

    # Generate cross-camera matches
    print("[INFO] Generating cross-camera matches...")
    matches = generate_cross_camera_matches(
        detections_df,
        max_time_diff=args.max_time_diff,
        min_combined_score=args.min_score
    )

    # Save results
    save_matches_to_csv(matches, args.output_file)

    # Show statistics
    if matches:
        df = pd.DataFrame(matches)
        print("\n[STATISTICS]")
        print("Total matches: {}".format(len(matches)))
        print("Average combined score: {:.3f}".format(df['combined_score'].mean()))
        print("Average time difference: {:.1f} minutes".format(df['time_diff_minutes'].mean()))
        print("Camera pairs:")
        camera_pairs = df[['camera_a', 'camera_b']].drop_duplicates()
        for _, row in camera_pairs.iterrows():
            count = len(df[(df['camera_a'] == row['camera_a']) & (df['camera_b'] == row['camera_b'])])
            print("  {} <-> {}: {} matches".format(row['camera_a'], row['camera_b'], count))


if __name__ == "__main__":
    main()