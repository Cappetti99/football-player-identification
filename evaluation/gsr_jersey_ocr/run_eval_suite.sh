#!/usr/bin/env bash
set -euo pipefail

# Controlled OCR-only evaluation on SoccerNet-GSR ground-truth boxes.
# Override these variables from the shell to scale the run up/down.
GSR_DIR="${GSR_DIR:-/media/data-lie/cappetti/dataset/SoccerNet-GSR}"
OUT_ROOT="${OUT_ROOT:-evaluation_outputs/gsr_jersey_ocr}"
LOG_DIR="${LOG_DIR:-logs/evaluation}"
SPLIT="${SPLIT:-val}"
MAX_SEQUENCES="${MAX_SEQUENCES:-2}"
MAX_TRACKLETS="${MAX_TRACKLETS:-100}"
CROPS_PER_TRACKLET="${CROPS_PER_TRACKLET:-20}"
TEMPLATE_MIN_SCORE="${TEMPLATE_MIN_SCORE:-0.62}"
TEMPLATE_WEIGHT="${TEMPLATE_WEIGHT:-0.03}"
TEMPLATE_MAX_CANDIDATES="${TEMPLATE_MAX_CANDIDATES:-4}"
RUN_MMOCR="${RUN_MMOCR:-false}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-football-tracking}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
WANDB_LOG_ARTIFACTS="${WANDB_LOG_ARTIFACTS:-true}"
WANDB_ALERT_ON_FINISH="${WANDB_ALERT_ON_FINISH:-true}"
WANDB_ALERT_ON_FAILURE="${WANDB_ALERT_ON_FAILURE:-true}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

lower_bool() {
  printf "%s" "$1" | tr '[:upper:]' '[:lower:]'
}

wandb_args=()
case "$(lower_bool "$WANDB")" in
  1|true|yes|on)
    wandb_args+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-init-timeout "$WANDB_INIT_TIMEOUT")
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args+=(--wandb-entity "$WANDB_ENTITY")
    fi
    if [[ "$(lower_bool "$WANDB_LOG_ARTIFACTS")" =~ ^(0|false|no|off)$ ]]; then
      wandb_args+=(--no-wandb-log-artifacts)
    fi
    if [[ "$(lower_bool "$WANDB_ALERT_ON_FINISH")" =~ ^(0|false|no|off)$ ]]; then
      wandb_args+=(--no-wandb-alert-on-finish)
    fi
    if [[ "$(lower_bool "$WANDB_ALERT_ON_FAILURE")" =~ ^(0|false|no|off)$ ]]; then
      wandb_args+=(--no-wandb-alert-on-failure)
    fi
    ;;
esac

run_eval() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  local log_path="$LOG_DIR/${name}.log"

  echo "==== $name"
  echo "output: $out_dir"
  echo "log:    $log_path"

  python3 evaluation/gsr_jersey_ocr/run_eval.py \
    --gsr-dir "$GSR_DIR" \
    --output-dir "$out_dir" \
    --split "$SPLIT" \
    --max-sequences "$MAX_SEQUENCES" \
    --max-tracklets "$MAX_TRACKLETS" \
    --crops-per-tracklet "$CROPS_PER_TRACKLET" \
    --template-min-score "$TEMPLATE_MIN_SCORE" \
    --template-weight "$TEMPLATE_WEIGHT" \
    --template-max-candidates "$TEMPLATE_MAX_CANDIDATES" \
    --wandb-name "$name" \
    "${wandb_args[@]}" \
    "$@" 2>&1 | tee "$log_path"

  echo "metrics:"
  python3 -m json.tool "$out_dir/metrics.json"
}

run_eval "gsr_${SPLIT}_easyocr_${MAX_SEQUENCES}s_${MAX_TRACKLETS}t" \
  --backend easyocr \
  --easyocr-gpu

run_eval "gsr_${SPLIT}_easyocr_template_${MAX_SEQUENCES}s_${MAX_TRACKLETS}t" \
  --backend easyocr \
  --easyocr-gpu \
  --template-matching \
  --template-font-image docs/numberFont.jpg

case "$(lower_bool "$RUN_MMOCR")" in
  1|true|yes|on)
    run_eval "gsr_${SPLIT}_mmocr_${MAX_SEQUENCES}s_${MAX_TRACKLETS}t" \
      --backend mmocr

    run_eval "gsr_${SPLIT}_mmocr_rec_${MAX_SEQUENCES}s_${MAX_TRACKLETS}t" \
      --backend mmocr_rec

    run_eval "gsr_${SPLIT}_mmocr_easyocr_${MAX_SEQUENCES}s_${MAX_TRACKLETS}t" \
      --backend mmocr_easyocr \
      --easyocr-gpu
    ;;
  *)
    echo "==== skipping mmocr suite; set RUN_MMOCR=true after installing mmocr"
    ;;
esac
