# facial_recognition
cyber identity project using facial recognition on live webcam footage

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

- `data/raw_frames/` - Original extracted PNG frames organized by camera and day
- `data/detections/` - YOLO person detection outputs in CSV format (camera, timestamp, bounding box, confidence, face embedding)
- `data/summaries/` - Processed results:
  - `candidate_links_with_faces.csv` - Pairwise person matches with similarity scores
  - `person_paths.csv` - Multi-frame person tracking paths
- `data/visuals/person_paths/` - Multi-frame visualizations of tracked individuals

## Improved Person Re-identification

This project now includes face recognition for more accurate person matching across frames:

### Detection Phase (detect_people.py)
- Uses YOLOv8 for person detection
- Extracts 128-dimensional face embeddings using face_recognition library
- Stores embeddings in detection CSV files

### Linking Phase (link_candidates_with_faces.cpp)
- High-performance C++ implementation for pairwise person matching
- Combines face similarity with temporal and size-based matching
- Uses weighted scoring: 40% face similarity, 30% time proximity, 30% size similarity
- Within-camera and cross-camera matching with consecutive frame handling
- Outputs candidate links ranked by combined confidence score

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

#### Path Building (build_person_paths.cpp)
- **High-Performance C++ Implementation**: Optimized graph processing with built-in deduplication
- **Directed Graph Construction**: Time-ordered edges eliminate redundant path exploration
- **Multi-frame Paths**: Creates sequences of up to 6 consecutive frames showing the same person
- **Cross-camera Preference**: Prioritizes paths that span different cameras (when available)
- **Scoring**: Ranks paths by length, camera diversity, and match quality

#### Path Visualization (visualize_person_paths.py)
- **Multi-frame Display**: Shows person paths as vertical stacks of frames
- **Bounding Boxes**: Highlights the tracked individual in each frame
- **Metadata Overlay**: Displays camera, time, and path information
- **Path Sequences**: Visualizes how individuals move through the surveillance area

### Performance Features

- **Neighbor Pruning**: `--max-neighbors 8` keeps top-scoring candidates per detection node
- **Start Node Filtering**: `--max-start-nodes 20000` focuses search on high-potential paths
- **Consecutive Frame Handling**: Special matching logic for duplicate/near-identical frames
- **Built-in Deduplication**: Removes duplicate detection sequences, reducing output by 70%+
- **Time-Forward Traversal**: Directed DFS prevents exploring backward in time

### Limitations & Future Work

- **Real-time Processing**: Currently batch-only; not optimized for live video streams
- **Multi-GPU Support**: Could parallelize detection and matching phases across devices
- **Temporal Gap Handling**: May miss individuals who leave frame and return after long gaps
- **Occlusion Handling**: Limited robustness when individuals are partially occluded   

//#### Current Example
//![Candidate Path 001](https://raw.githubusercontent.com/ginganinja04/facial_recognition/main/data/visuals/candidate_path_001_visualization.png) 