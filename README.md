# facial_recognition
cyber indentity project using facial recognition on live webcam footage

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

