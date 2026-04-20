# facial_recognition
cyber identity project using facial recognition on live webcam footage

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

data/raw_frames/ contains the original extracted PNG frames for each camera and day.   
data/detections/ contains YOLO person detection outputs in CSV format.   
data/summaries/ contains merged datasets and profile analysis outputs.   
data/visuals/ contains plots, heatmaps, annotated images, and match visualizations.

## Improved Person Re-identification

This project now includes face recognition for more accurate person matching across frames:

### Detection Phase (detect_people.py)
- Uses YOLOv8 for person detection
- Extracts 128-dimensional face embeddings using face_recognition library
- Stores embeddings in detection CSV files

### Linking Phase (link_candidates_with_faces.py)
- Combines face similarity with temporal and size-based matching
- Uses weighted scoring: 40% face similarity, 30% time proximity, 30% size similarity
- Outputs candidate links with confidence scores

### Usage

1. Run person detection with face extraction:
```bash
python3 scripts/detect_people.py --raw-frames-dir data/raw_frames --detections-dir data/detections
```

2. Compile the C++ candidate linking program:
```bash
g++ -std=c++17 -O2 scripts/link_candidates_with_faces.cpp -o scripts/link_candidates_with_faces
```

3. Link candidates using the compiled C++ matcher:
```bash
./scripts/link_candidates_with_faces --detections-dir data/detections --output-file data/summaries/candidate_links_with_faces.csv
```

4. Build person paths by chaining matches:
```bash
g++ -std=c++17 -O2 scripts/build_person_paths.cpp -o scripts/build_person_paths
./scripts/build_person_paths --matches-csv data/summaries/candidate_links_with_faces.csv --output-file data/summaries/person_paths.csv --min-confidence 0.45 --max-neighbors 8 --max-start-nodes 20000
```

- `--min-confidence`: omit lower-confidence detections that are likely false positives
- `--require-face`: only include detections that have valid face extraction metadata
- `--max-neighbors`: limit the number of outgoing candidate edges per detection to speed path search
- `--max-start-nodes`: only search the top-scoring start nodes when building person paths

The C++ builder automatically deduplicates identical detection sequences and prints an estimated remaining time while constructing the graph and finding paths.

5. Visualize person paths across multiple frames:
```bash
python3 scripts/visualize_person_paths.py --paths-csv data/summaries/person_paths.csv --output-dir data/visuals/person_paths
```

### Person Path Tracking

This system now builds complete paths of individuals moving through multiple frames:

#### Path Building (build_person_paths.py)
- **Graph-based Matching**: Treats detections as nodes and matches as edges
- **Path Finding**: Uses depth-first search to find chains of matches
- **Multi-frame Paths**: Creates sequences of up to 6 consecutive frames showing the same person
- **Cross-camera Preference**: Prioritizes paths that span different cameras (when available)
- **Scoring**: Ranks paths by length, camera diversity, and match quality

#### Path Visualization (visualize_person_paths.py)
- **Multi-frame Display**: Shows person paths as vertical stacks of frames
- **Bounding Boxes**: Highlights the tracked individual in each frame
- **Metadata Overlay**: Displays camera, time, and path information
- **Path Sequences**: Visualizes how individuals move through the surveillance area

### Current Limitations & Future Improvements

- **Face Recognition**: Currently disabled due to library installation issues
- **Cross-camera Matching**: Limited by lack of face embeddings for inter-camera comparison
- **Real-time Processing**: Batch processing only; not optimized for live video streams

#### To Enable Full Cross-Camera Tracking:
1. Install face recognition dependencies:
```bash
pip install face-recognition
```
2. Re-run detection to extract face embeddings:
```bash
python3 scripts/detect_people.py --raw-frames-dir data/raw_frames --detections-dir data/detections
```
3. Re-run matching with cross-camera enabled:
```bash
g++ -std=c++17 -O2 scripts/link_candidates_with_faces.cpp -o scripts/link_candidates_with_faces
./scripts/link_candidates_with_faces --detections-dir data/detections --output-file data/summaries/candidate_links_with_faces_cross_camera.csv
```   
