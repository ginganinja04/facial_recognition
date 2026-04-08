#!/usr/bin/env bash
set -euo pipefail

YOUTUBE_URL="https://www.youtube.com/watch?v=rnNPl27Arpk" ## new orleans live   
OUT_BASE="/mnt/c/Users/Maria/spring26_projects/facial_recognition/frames/new_orleans/balcony_view/"
SECONDS_BETWEEN_FRAMES=3
DURATION="02:00:00"

DATE_DIR="$(date +%F)"
TIME_DIR="$(date +%H-%M-%S)"
OUT_DIR="${OUT_BASE}/${DATE_DIR}/${TIME_DIR}"

mkdir -p "$OUT_DIR"
echo "Saving frames to $OUT_DIR"

STREAM_URL="$(yt-dlp -g -f "bv*/b" "$YOUTUBE_URL" | head -n 1)" ## asks yt-dlp for the direct media URL of the best video stream

ffmpeg -y \
  -i "$STREAM_URL" \
  -t "$DURATION" \
  -vf "fps=1/${SECONDS_BETWEEN_FRAMES}" \
  -q:v 3 \
  "${OUT_DIR}/frame_%06d.jpg"

echo "Capture complete"