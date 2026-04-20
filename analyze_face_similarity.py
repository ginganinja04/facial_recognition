#!/usr/bin/env python3
"""Analyze face similarity distribution to find better threshold."""

import pandas as pd
import numpy as np

matches = pd.read_csv('data/summaries/candidate_links_with_faces.csv')

print("="*80)
print("FACE SIMILARITY DISTRIBUTION ANALYSIS")
print("="*80)
print()

# Get statistics
face_sims = matches['face_similarity'].values
print(f"Total matches: {len(matches)}")
print(f"Face similarity statistics:")
print(f"  Min: {face_sims.min():.6f}")
print(f"  Max: {face_sims.max():.6f}")
print(f"  Mean: {face_sims.mean():.6f}")
print(f"  Median: {np.median(face_sims):.6f}")
print(f"  Std Dev: {face_sims.std():.6f}")
print()

# Distribution breakdown
thresholds = [0.80, 0.85, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
print("Matches above various thresholds:")
for t in thresholds:
    count = len(matches[matches['face_similarity'] >= t])
    pct = 100 * count / len(matches)
    print(f"  >= {t:.2f}: {count:7d} ({pct:5.1f}%)")

print()
print("="*80)
print("ANALYSIS BY CAMERA PAIR TYPE")
print("="*80)

# Separate same-camera and cross-camera
same_camera = matches[matches['camera_a'] == matches['camera_b']]
cross_camera = matches[matches['camera_a'] != matches['camera_b']]

print(f"\nSame-camera matches: {len(same_camera)} ({100*len(same_camera)//len(matches)}%)")
print(f"  Face similarity: min={same_camera['face_similarity'].min():.4f}, "
      f"mean={same_camera['face_similarity'].mean():.4f}, "
      f"max={same_camera['face_similarity'].max():.4f}")

print(f"\nCross-camera matches: {len(cross_camera)} ({100*len(cross_camera)//len(matches)}%)")
print(f"  Face similarity: min={cross_camera['face_similarity'].min():.4f}, "
      f"mean={cross_camera['face_similarity'].mean():.4f}, "
      f"max={cross_camera['face_similarity'].max():.4f}")

print()
print("="*80)
print("RECOMMENDATION")
print("="*80)
print()
print("Current threshold: 0.95 (too aggressive)")
print()
print("Suggested threshold:")
print("  - Same-camera: 0.90 (allows 90% + high time/size)")
print("  - Cross-camera: 0.92 (allows 92% - medium confidence with high time/size)")
print()
print("This balances:")
print("  ✓ Rejecting clearly different people (0.88 range)")
print("  ✓ Accepting high-confidence same-person matches")
print("  ✓ Enough candidates for good path building")
