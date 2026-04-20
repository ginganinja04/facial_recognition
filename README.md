# facial_recognition
cyber identity project using facial recognition on live webcam footage

<<<<<<< HEAD
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
=======
data/raw_frames/ contains the original extracted PNG frames for each camera and day.   
data/detections/ contains YOLO person detection outputs in CSV format.   
data/summaries/ contains merged datasets and profile analysis outputs.   
data/visuals/ contains plots, heatmaps, and annotated images used in the final report.   

## Pipeline

### 1. Frame Collection
Collect frames from multiple public camera viewpoints and store them under:
- data/raw_frames/<camera>/<day folder>/

Current cameras:
- bar_stage
- inside_bar
- street_view
- balcony

Files are named like:
- bar_stage_day3_19-35-06.jpg
- inside_bar_day2_18-24-42.jpg

Each frame encodes:
- camera
- day
- timestamp

---

### 2. Person Detection
Run detect_people.py on the raw frames.
```
python3 scripts/merge_detections.py
```

Input:
- data/raw_frames/...

Output:
0 Per-camera/day detection CSVs in data/detections/

Each row represents one detected person and includes:
- camera
- day
- frame filename
- timestamp
- bounding box coordinates
- center_x / center_y
- bbox area
- confidence

---

### 3. Detection Merge + Summaries
Run the merge script to combine all detection CSVs.
```
python3 scripts/merge_detections.py
```

**Outputs (in `data/summaries/`):**
- all_detections.csv
- people_counts_by_frame.csv
- camera_daily_summary.csv

This provides:
- one unified detection table
- frame-level person counts
- daily summaries per camera

---

### 4. Basic Plots
```python3 scripts/make_plots.py```
Generate visual summaries such as:
- people count over time
- daily totals
- heatmaps

**Saved in:**
- data/visuals/

This stage supports descriptive analysis:
- camera activity levels
- spatial clustering of detections
- temporal patterns of presence

---

### 5. Profile Construction
Run `build_profiles.py` on `all_detections.csv`.
```
python3 scripts/build_profiles.py \
  --input data/summaries/all_detections.csv \
  --output data/summaries/pseudonymous_profiles.csv \
  --raw-frames-dir data/raw_frames \
  --spatial-bin-size 80 \
  --size-quantiles 4 \
  --time-bucket-minutes 3 \
  --min-confidence 0.50
```

This improved version:
- filters low-confidence detections
- uses smaller time buckets
- uses spatial bins
- uses size groupings
- computes simple appearance features from cropped detections:
  - mean hue
  - mean saturation
  - mean value
  - aspect ratio

Profiles are grouped using:
- camera
- day
- zone (spatial bin)
- size
- time bucket
- appearance label

**Output:**
- data/summaries/pseudonymous_profiles.csv
> This stage converts detections into recurring pseudonymous profiles.

---

### 6. Candidate Profile Linking
Run the C++ linker:
```g++ -O3 -std=c++17 link_candidate_profiles.cpp -o link_candidate_profiles_cpp```

```bash
./link_candidate_profiles_cpp \
  --input data/summaries/pseudonymous_profiles.csv \
  --links-output data/summaries/candidate_profile_links.csv \
  --groups-output data/summaries/candidate_profile_groups.csv \
  --min-detection-count 12 \
  --min-unique-frames 6 \
  --max-time-diff-min 8 \
  --max-size-diff 1
```
Input:
- pseudonymous_profiles.csv

Profiles are filtered by:
- minimum detection count
- minimum unique frame count

Profiles are compared using constraints:
- same day
- limited time difference
- limited size difference

Outputs:
- data/summaries/candidate_profile_links.csv
- data/summaries/candidate_profile_groups.csv
> This stage links profiles across cameras into candidate correspondences.

---

### 7. Candidate Path Building
Run build_candidate_paths.py.
```
python3 scripts/build_candidate_paths.py \
  --input data/summaries/candidate_profile_links.csv \
  --output data/summaries/candidate_paths.csv \
  --summary-output data/summaries/candidate_path_summary.csv \
  --min-score 0.85
```

Input:
- candidate_profile_links.csv
> Links are filtered by score and combined into short paths.

Outputs:
- data/summaries/candidate_paths.csv
- data/summaries/candidate_path_summary.csv

Typical paths:
- 2-step paths
- same short time window
- cross-camera matches

Example:
- candidate_path_001, day2, bar_stage -> inside_bar,
- 18:21-18:23 -> 18:21-18:23, 2, 0.8681
> This stage converts pairwise links into candidate cross-camera paths.

---

### 8. Path Visualization
Run visualize_candidate_path.py.
```
python3 scripts/visualize_candidate_path.py \
  --path-id candidate_path_001 \
  --candidate-paths data/summaries/candidate_paths.csv \
  --profiles data/summaries/pseudonymous_profiles.csv \
  --all-detections data/summaries/all_detections.csv \
  --raw-frames-dir data/raw_frames \
  --output data/visuals/candidate_path_001_visualization.png
```

Inputs:
- candidate_paths.csv
- pseudonymous_profiles.csv
- all_detections.csv

For a selected path:
- find path steps
- select representative detections
- locate corresponding raw frames
- draw bounding boxes
- combine into a side-by-side image

Output:
- data/visuals/<path_id>_visualization.png
> This stage enables qualitative inspection of candidate paths.

#### Correct Example
![Candidate Path 001](https://raw.githubusercontent.com/ginganinja04/facial_recognition/main/data/visuals/candidate_path_001_visualization.png) 

---

>>>>>>> dcf8094f8200a0e56c016685328727176d5f0001
