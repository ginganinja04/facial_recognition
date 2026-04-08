#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage:"
  echo "  $0 <camera_name> <day_label> <duration> <seconds_between_frames> <stream_url>"
  echo
  echo "Example:"
  echo '  ./scripts/capture_frames.sh balcony day1 02:00:00 3 "https://www.youtube.com/watch?v=..."'
  exit 1
fi

CAMERA_NAME="$1"
DAY_LABEL="$2"
DURATION="$3"
SECONDS_BETWEEN_FRAMES="$4"
STREAM_URL="$5"

PROJECT_ROOT="/mnt/c/Users/Maria/spring26_projects/FACIAL_RECOGNITION"
OUT_DIR="${PROJECT_ROOT}/data/raw_frames/${CAMERA_NAME}/${DAY_LABEL}"

mkdir -p "$OUT_DIR"

START_TIME="$(date +%H-%M-%S)"
START_TIME_FILE="$(date +%H:%M:%S)"
DATE_FILE="$(date +%F)"

echo "Saving raw frames to: $OUT_DIR"
echo "Camera: $CAMERA_NAME"
echo "Day: $DAY_LABEL"
echo "Start time: $START_TIME_FILE"

ffmpeg -y \
  -i "$STREAM_URL" \
  -t "$DURATION" \
  -vf "fps=1/${SECONDS_BETWEEN_FRAMES}" \
  "${OUT_DIR}/frame_%06d.png"

cat > "${OUT_DIR}/capture_metadata.txt" <<EOF
camera_name=${CAMERA_NAME}
day_label=${DAY_LABEL}
capture_date=${DATE_FILE}
capture_start_time=${START_TIME_FILE}
seconds_between_frames=${SECONDS_BETWEEN_FRAMES}
duration=${DURATION}
stream_url=${STREAM_URL}
EOF

echo "Capture complete."
echo "Metadata written to ${OUT_DIR}/capture_metadata.txt"
echo "Next: run rename_standardize_frames.py"