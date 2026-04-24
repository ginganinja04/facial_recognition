from pathlib import Path
import argparse
import cv2
import pandas as pd


def main():
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

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for frame_file, group in df.groupby("frame_file"):
        image_path = raw_frames_dir / frame_file
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] Could not read {image_path}")
            continue

        for _, row in group.iterrows():
            x1 = int(row["x1"])
            y1 = int(row["y1"])
            x2 = int(row["x2"])
            y2 = int(row["y2"])
            track_id = int(row["track_id"])

            is_cross_camera_match = (
                "global_match_score" in row
                and pd.notna(row["global_match_score"])
            )
            
            # OpenCV uses BGR, so red is (0, 0, 255)
            box_color = (0, 0, 255) if is_cross_camera_match else (0, 255, 0)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            # label
            global_id = int(row["global_id"]) if "global_id" in row and pd.notna(row["global_id"]) else -1

            if is_cross_camera_match:
                label = f"ID {track_id} G{global_id} XCAM"
            else:
                label = f"ID {track_id} G{global_id}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            text_x = x1
            text_y = max(20, y1 - 8)

            cv2.rectangle(
                image,
                (text_x, text_y - text_h - 6),
                (text_x + text_w + 6, text_y + baseline - 2),
                (0, 255, 0),
                -1,
            )

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

        out_path = out_dir / frame_file
        cv2.imwrite(str(out_path), image)
        print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()