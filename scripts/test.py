import pandas as pd

df = pd.read_csv("mini_demo/data/detections/street_view/day1_detections.csv")

print(df[["frame_file", "person_id_in_frame", "track_id", "match_score"]])

print("\nCounts per track:")
print(df["track_id"].value_counts().sort_index())