#!/usr/bin/env bash
set -euo pipefail

SEQUENCE="${1:-SNMOT-155}"
MAX_FRAMES="${MAX_FRAMES:-300}"
MODEL_PATH="${MODEL_PATH:-../models/best_yolo26x_gsr_light.pt}"
VIDEO_PATH="${VIDEO_PATH:-../input_videos/soccernet/${SEQUENCE}.mp4}"
OUTPUT_PATH="${OUTPUT_PATH:-output_videos/${SEQUENCE}_ft.mp4}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts/${SEQUENCE}_ft}"
ROSTER_PATH="${ROSTER_PATH:-configs/roster.example.json}"

python3 -m ft.cli run \
  --config configs/default.yaml \
  --video-path "$VIDEO_PATH" \
  --model-path "$MODEL_PATH" \
  --output-path "$OUTPUT_PATH" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --roster-path "$ROSTER_PATH" \
  --max-frames "$MAX_FRAMES"

