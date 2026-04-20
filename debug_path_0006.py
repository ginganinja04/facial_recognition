#!/usr/bin/env python3
"""Debug script to analyze why path 0006 incorrectly merges different people."""

import pandas as pd
from pathlib import Path
import numpy as np

# Load data
paths_df = pd.read_csv('data/summaries/person_paths.csv')
matches_df = pd.read_csv('data/summaries/candidate_links_with_faces.csv')
dets = pd.concat([pd.read_csv(f) for f in Path('data/detections').rglob('*.csv')], ignore_index=True)

# Get path 0006
path_0006 = paths_df[paths_df['path_id'] == 6].iloc[0]
print("Path 0006 details:")
print(f"  Length: {path_0006['path_length']}")
print(f"  Camera changes: {path_0006['camera_changes']}")
print(f"  Average score: {path_0006['average_score']:.6f}")
print(f"  Total score: {path_0006['total_score']:.6f}")
print()

# Parse the detection IDs in the path
detection_ids = path_0006['detection_ids'].split('|')
print(f"Detection IDs in path ({len(detection_ids)} total):")
for i, det_id in enumerate(detection_ids):
    print(f"  {i}: {det_id}")
    # Extract person IDs
    parts = det_id.split('_')
    person_id = int(parts[-1])
    camera = parts[0]
    print(f"     -> Person {person_id} in {camera}")
print()

# Find the matches that chain these detections together
print("Matches connecting these detections:")
for i in range(len(detection_ids) - 1):
    id_a = detection_ids[i]
    id_b = detection_ids[i + 1]
    
    # Find the match
    match = matches_df[
        ((matches_df['detection_id_a'] == id_a) & (matches_df['detection_id_b'] == id_b)) |
        ((matches_df['detection_id_a'] == id_b) & (matches_df['detection_id_b'] == id_a))
    ]
    
    if len(match) > 0:
        match = match.iloc[0]
        print(f"\n  Link {i}: {id_a} → {id_b}")
        print(f"    Face similarity: {match['face_similarity']:.6f}")
        print(f"    Time similarity: {match['time_similarity']:.6f}")
        print(f"    Size similarity: {match['size_similarity']:.6f}")
        print(f"    Combined score: {match['combined_score']:.6f}")
        print(f"    Time diff: {match['time_diff_minutes']} minutes")
        print(f"    Size diff: {match['size_diff']:.2f}")
        
        # Extract person IDs
        parts_a = id_a.split('_')
        parts_b = id_b.split('_')
        person_a = int(parts_a[-1])
        person_b = int(parts_b[-1])
        
        if person_a != person_b:
            print(f"    ⚠️  WARNING: Different people! ({person_a} → {person_b})")
    else:
        print(f"\n  Link {i}: {id_a} → {id_b}")
        print(f"    ⚠️  NO MATCH FOUND!")

print("\n" + "="*80)
print("Analysis of problematic links:")
print("="*80)

# Check which person IDs appear
person_ids = set()
for det_id in detection_ids:
    parts = det_id.split('_')
    person_id = int(parts[-1])
    person_ids.add(person_id)

print(f"Unique person IDs in path: {sorted(person_ids)}")
print()

# Identify where people change
print("Person ID transitions:")
for i in range(len(detection_ids)):
    parts = detection_ids[i].split('_')
    person_id = int(parts[-1])
    camera = parts[0]
    timestamp = detection_ids[i].split('_')[-4]  # Approximate
    print(f"  Frame {i}: Person {person_id} ({camera})")
    
    if i > 0:
        parts_prev = detection_ids[i-1].split('_')
        prev_person = int(parts_prev[-1])
        if person_id != prev_person:
            print(f"    -> PERSON CHANGE from {prev_person} to {person_id}")
