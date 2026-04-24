# facial_recognition
cyber identity project using facial recognition on live webcam footage

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Run person detection with face extraction:
```bash
python3 scripts/detect_people.py --raw-frames-dir data/raw_frames --detections-dir data/detections
```

2. Visualize Bounding Boxes and IDs:
```bash
python3 scripts/visualize_tracks.py \
  --csv-path data/detections/balcony/day1_detections.csv \
  --raw-frames-dir data/raw_frames/balcony \
  --out-dir data/tracks_visualized/balcony &&
python3 scripts/visualize_tracks.py \
  --csv-path data/detections/bar_stage/day1_detections.csv \
  --raw-frames-dir data/raw_frames/bar_stage \
  --out-dir data/tracks_visualized/bar_stage &&
python3 scripts/visualize_tracks.py \
  --csv-path data/detections/inside_bar/day1_detections.csv \
  --raw-frames-dir data/raw_frames/inside_bar \
  --out-dir data/tracks_visualized/inside_bar &&
python3 scripts/visualize_tracks.py \
  --csv-path data/detections/street_view/day1_detections.csv \
  --raw-frames-dir data/raw_frames/street_view \
  --out-dir data/tracks_visualized/street_view
```


