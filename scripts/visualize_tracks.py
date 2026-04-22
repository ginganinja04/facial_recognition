from pathlib import Path
import cv2
import pandas as pd

CSV_PATH = "mini_demo/data/detections/street_view/day1_detections.csv"
RAW_FRAMES_DIR = Path("mini_demo/data/raw_frames/street_view/day1")
OUT_DIR = Path("mini_demo/data/tracks_visualized/street_view/day1")

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for frame_file, group in df.groupby("frame_file"):
    image_path = RAW_FRAMES_DIR / frame_file
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

        # box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # label background
        label = f"ID {track_id}"
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

        # label text
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

    out_path = OUT_DIR / frame_file
    cv2.imwrite(str(out_path), image)
    print(f"[OK] Wrote {out_path}")