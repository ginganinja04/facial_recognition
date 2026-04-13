from pathlib import Path
from datetime import datetime, timedelta

# CONFIG (edit this once)
ROOT = Path("data/raw_frames")
SECONDS_BETWEEN_FRAMES = 3

# Set start times manually per folder if needed
# Format: ("camera", "dayX"): "HH:MM:SS"
START_TIMES = {
    ("balcony", "day1 (04-08)"): "18:00:00",
    ("balcony", "day2 (04-09)"): "18:00:00",
    ("balcony", "day3 (04-10)"): "18:00:00",
    ("balcony", "day4 (04-11)"): "18:00:00",

    ("inside_bar", "day1 (04-08)"): "18:00:00",
    ("inside_bar", "day2 (04-09)"): "18:00:00",
    ("inside_bar", "day3 (04-10)"): "18:00:00",
    ("inside_bar", "day4 (04-11)"): "18:00:00",


    ("street_view", "day1 (04-08)"): "18:00:00",
    ("street_view", "day2 (04-09)"): "18:00:00",
    ("street_view", "day3 (04-10)"): "18:00:00",
    ("street_view", "day4 (04-11)"): "18:00:00",

    ("bar_stage", "day1 (04-08)"): "18:00:00",
    ("bar_stage", "day2 (04-09)"): "18:00:00",
    ("bar_stage", "day3 (04-10)"): "18:00:00",
    ("bar_stage", "day4 (04-11)"): "18:00:00"
}

def rename_folder(camera_path: Path):
    camera = camera_path.name

    for day_folder in camera_path.iterdir():
        if not day_folder.is_dir():
            continue

        day = day_folder.name

        key = (camera, day)
        if key not in START_TIMES:
            print(f"⚠️ Missing start time for {key}, skipping")
            continue

        start_time = datetime.strptime(START_TIMES[key], "%H:%M:%S")

        files = sorted(day_folder.glob("*.jpg"))

        print(f"\nProcessing {camera}/{day} ({len(files)} files)")

        for i, file in enumerate(files):
            timestamp = start_time + timedelta(seconds=i * SECONDS_BETWEEN_FRAMES)
            time_str = timestamp.strftime("%H-%M-%S")

            new_name = f"{camera}_{day.split()[0]}_{time_str}.jpg"
            new_path = file.parent / new_name

            file.rename(new_path)

        print(f"✅ Done {camera}/{day}")


def main():
    for camera_path in ROOT.iterdir():
        if camera_path.is_dir():
            rename_folder(camera_path)


if __name__ == "__main__":
    main()