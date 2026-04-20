import pandas as pd
from pathlib import Path

matches = pd.read_csv('data/summaries/candidate_links_with_faces.csv')
dets = pd.concat([pd.read_csv(f) for f in Path('data/detections').rglob('*.csv')], ignore_index=True)
print('matches', len(matches), 'dets', len(dets))


def parse_detection_id(detection_id):
    parts = detection_id.split('_')
    camera = parts[0]
    day = parts[1]
    frame_file = '_'.join(parts[2:-1])
    person_id = int(parts[-1])
    return camera, day, frame_file, person_id

valid = 0
missing = 0
low_conf = 0
for _, row in matches.iterrows():
    camera_a, day_a, frame_a, pid_a = parse_detection_id(row['detection_id_a'])
    camera_b, day_b, frame_b, pid_b = parse_detection_id(row['detection_id_b'])
    det_a = dets[(dets['camera'] == camera_a) & (dets['day'] == day_a) & (dets['frame_file'] == frame_a) & (dets['person_id_in_frame'] == pid_a)]
    det_b = dets[(dets['camera'] == camera_b) & (dets['day'] == day_b) & (dets['frame_file'] == frame_b) & (dets['person_id_in_frame'] == pid_b)]
    if len(det_a) == 0 or len(det_b) == 0:
        missing += 1
        continue
    det_a = det_a.iloc[0]
    det_b = det_b.iloc[0]
    if det_a['confidence'] < 0.45 or det_b['confidence'] < 0.45:
        low_conf += 1
        continue
    valid += 1

print('valid', valid, 'missing', missing, 'low_conf', low_conf)
