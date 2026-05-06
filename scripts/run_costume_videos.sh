#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-costume-video}"
MAX_FRAMES="${MAX_FRAMES:-1800}"
MODEL_PATH="${MODEL_PATH:-best_yolo26x_gsr_light.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output_videos/costume-video}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-artifacts/costume-video}"
ROSTER_PATH="${ROSTER_PATH:-configs/roster.example.json}"
LIMIT="${LIMIT:-}"

ARGS=(
  python3 -m ft.cli run
  --config configs/default.yaml
  --input-dir "$INPUT_DIR"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --artifacts-root "$ARTIFACTS_ROOT"
  --roster-path "$ROSTER_PATH"
  --max-frames "$MAX_FRAMES"
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

"${ARGS[@]}"
