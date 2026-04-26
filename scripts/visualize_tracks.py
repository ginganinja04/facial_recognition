"""
Purpose:
--------
This script overlays tracking results onto raw video frames.

It reads a CSV file containing detection + tracking outputs (bounding boxes,
track IDs, optional global matching info), draws those on the corresponding
frames, and saves annotated images to an output directory.

Pipeline Role:
--------------
This sits AFTER detection + tracking + (optional) cross-camera linking.

Input:
- CSV with detections/tracks (per-frame bounding boxes + IDs)
- Raw frames directory

Output:
- Annotated frames with bounding boxes + labels

Key Features:
-------------
- Draws bounding boxes for each detected person
- Colors boxes differently if cross-camera matches exist
- Displays track ID and optional global ID
- Writes one annotated image per frame
"""

from pathlib import Path
import argparse
import cv2
import pandas as pd


def main():
    """
    Entry point:
    1. Parse CLI arguments
    2. Load detection CSV
    3. Iterate through frames
    4. Draw bounding boxes + labels
    5. Save annotated frames
    """

    # -------------------------------
    # Step 1: Parse command-line args
    # -------------------------------
    parser = argparse.ArgumentParser(
        description="Visualize tracking results on frames."
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to detections CSV",
    )

    parser.add_argument(
        "--raw-frames-dir",
        type=Path,
        required=True,
        help="Directory containing raw frames",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for visualized frames",
    )

    args = parser.parse_args()

    csv_path = args.csv_path
    raw_frames_dir = args.raw_frames_dir
    out_dir = args.out_dir

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Step 2: Load detections/tracks
    # -------------------------------
    df = pd.read_csv(csv_path)

    # -------------------------------
    # Step 3: Process frame-by-frame
    # -------------------------------
    # Group detections by frame so we can draw all boxes per image
    for frame_file, group in df.groupby("frame_file"):

        # Load corresponding image
        image_path = raw_frames_dir / frame_file
        image = cv2.imread(str(image_path))

        # Skip if frame is missing or unreadable
        if image is None:
            print(f"[WARN] Could not read {image_path}")
            continue

        # -------------------------------
        # Step 4: Draw detections/tracks
        # -------------------------------
        for _, row in group.iterrows():

            # Extract bounding box + track info
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            track_id = int(row["track_id"])

            # Determine whether this detection is linked across cameras
            # (based on presence of global_match_score)
            is_cross_camera_match = (
                "global_match_score" in row
                and pd.notna(row["global_match_score"])
            )

            # Color coding:
            # - Red = cross-camera match
            # - Green = regular detection
            box_color = (0, 0, 255) if is_cross_camera_match else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            # -------------------------------
            # Step 5: Construct label
            # -------------------------------
            # Optional global ID (if cross-camera linking was done)
            global_id = int(row["global_id"]) if "global_id" in row and pd.notna(row["global_id"]) else -1

            if is_cross_camera_match:
                label = f"ID {track_id} G{global_id} XCAM"
            else:
                label = f"ID {track_id} G{global_id}"

            # Compute text box size
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Position label above bounding box (with minimum y offset)
            text_x = x1
            text_y = max(20, y1 - 8)

            # Draw label background (for readability)
            cv2.rectangle(
                image,
                (text_x, text_y - text_h - 6),
                (text_x + text_w + 6, text_y + baseline - 2),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (text_x + 3, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # -------------------------------
        # Step 6: Save annotated frame
        # -------------------------------
        out_path = out_dir / frame_file
        cv2.imwrite(str(out_path), image)

        print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()