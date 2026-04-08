#!/usr/bin/env bash
set -euo pipefail

PAGE_URL="https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=catsmeowkaraoke"
STREAM_URL="https://videos-3.earthcam.com/fecnetwork/4281.flv/chunklist_w511007498.m3u8?t=YOUR_TOKEN_HERE&td=YOUR_TIMESTAMP_HERE"

OUT_BASE="/mnt/c/Users/Maria/spring26_projects/facial_recognition/frames/new_orleans/bar_stage"
##OUT_BASE="../maria_test"
SECONDS_BETWEEN_FRAMES=3
DURATION="03:00:00"

USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
HEADERS=$'Referer: '"$PAGE_URL"$'\r\nUser-Agent: '"$USER_AGENT"$'\r\n'

DATE_DIR="$(date +%F)"
TIME_DIR="$(date +%H-%M-%S)"
OUT_DIR="${OUT_BASE}/${DATE_DIR}/${TIME_DIR}"

mkdir -p "$OUT_DIR"
echo "Saving frames to $OUT_DIR"

ffmpeg -y \
  -headers "$HEADERS" \
  -i "$STREAM_URL" \
  -t "$DURATION" \
  -vf "fps=1/${SECONDS_BETWEEN_FRAMES}" \
  -q:v 3 \
  "${OUT_DIR}/frame_%06d.jpg"

echo "Capture complete"