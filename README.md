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
or for smaller demo
```bash
python3 scripts/detect_people.py --raw-frames-dir mini_demo/data/raw_frames --detections-dir mini_demo/data/detections
```


