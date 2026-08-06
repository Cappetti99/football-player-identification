#!/usr/bin/env bash
# Final evaluation on the sampled SoccerNet-GSR test split, unattended.
#
# Runs, in order: TVCalib on the 12 sampled test sequences, the pipeline under
# both jersey decision-policy arms, the GS-HOTA aggregation, and the oracle
# ablation. Every phase writes its own log, and a phase that fails stops the
# ones that depend on it rather than producing numbers from partial input.
#
# Protocol frozen in
# evaluation/jersey_heldout_manifests/gsr_test_final_v1_preregistration.md.
# The sample, its seed and the selection procedure are recorded in the manifest.
#
#   nohup bash scripts/run_test_final_v1.sh > /home/cappetti/FT/logs/test_final_v1/driver.log 2>&1 &

set -u

FT=/home/cappetti/FT
TVCALIB_REPO=/home/cappetti/tvcalib
TVCALIB_CKPT="${TVCALIB_REPO}/data/segment_localization/train_59.pt"
GSR=/media/data-lie/cappetti/dataset/SoccerNet-GSR

MANIFEST=evaluation/detection_tracking_manifests/test_final_v1.json
TVROOT=evaluation_outputs/tvcalib_gsr_test_final_v1
ARTIFACTS=artifacts/gs_hota_test_final
OUTPUTS=output_videos/gs_hota_test_final
EVALROOT=evaluation_outputs/gs_hota_test_final
MODEL=best_yolo26x_gsr_light.pt

SEQUENCES="SNGS-120 SNGS-125 SNGS-127 SNGS-131 SNGS-133 SNGS-135 SNGS-137 SNGS-138 SNGS-139 SNGS-149 SNGS-192 SNGS-193"

LOGDIR="${FT}/logs/test_final_v1"
mkdir -p "${LOGDIR}"

cd "${FT}" || exit 1
source ~/miniconda3/etc/profile.d/conda.sh

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

say "start | manifest=${MANIFEST}"
say "sequences: ${SEQUENCES}"

# --- preflight ------------------------------------------------------------
# Fail before spending a night on a run that cannot finish.
fail=0
[ -f "${MANIFEST}" ]      || { say "MISSING manifest: ${MANIFEST}"; fail=1; }
[ -f "${MODEL}" ]         || { say "MISSING detector: ${MODEL}"; fail=1; }
[ -f "${TVCALIB_CKPT}" ]  || { say "MISSING TVCalib checkpoint: ${TVCALIB_CKPT}"; fail=1; }
[ -d "${TVCALIB_REPO}" ]  || { say "MISSING TVCalib repo: ${TVCALIB_REPO}"; fail=1; }
# Check that the environments actually import what they need, not merely that
# a directory with the right name exists: a half-created env (conda packages
# installed, pip step failed) passes a name check and then dies on the first
# real call, after the preflight has already declared everything fine.
conda activate tvcalib-ft 2>/dev/null && \
  PYTHONPATH="${TVCALIB_REPO}" python3 -c "
from tvcalib.module import TVCalibModule
from sn_segmentation.src.custom_extremities import generate_class_synthesis
import pytorch_lightning, kornia, torch
assert torch.cuda.is_available(), 'CUDA not visible'
" 2>/dev/null || { say "env tvcalib-ft missing or incomplete (imports fail)"; fail=1; }
conda deactivate 2>/dev/null

conda activate jersey-yolo-ocr 2>/dev/null && \
  python3 -c "import torch, ultralytics; assert torch.cuda.is_available()" 2>/dev/null \
  || { say "env jersey-yolo-ocr missing or incomplete (imports fail)"; fail=1; }
conda deactivate 2>/dev/null
free_gb=$(df -BG --output=avail "${FT}" | tail -1 | tr -dc '0-9')
say "free disk: ${free_gb} GB"
[ "${free_gb}" -ge 20 ] || { say "LOW DISK: need ~20 GB for 24 runs of crops and artifacts"; fail=1; }
if [ "${fail}" -ne 0 ]; then say "preflight failed, nothing was run"; exit 1; fi
say "preflight ok"

# --- phase 1: TVCalib -----------------------------------------------------
say "=== 1/5 TVCalib on 12 test sequences (env: tvcalib-ft) ==="
conda activate tvcalib-ft
PYTHONPATH="${TVCALIB_REPO}" python3 scripts/run_tvcalib_gsr.py \
  --gsr-dir "${GSR}" \
  --split test \
  --sequences ${SEQUENCES} \
  --output-root "${TVROOT}" \
  --checkpoint "${TVCALIB_CKPT}" \
  --device cuda \
  --samples 30 \
  --optim-steps 1000 \
  --seed 10 \
  --resume \
  > "${LOGDIR}/1_tvcalib.log" 2>&1
tv_status=$?
conda deactivate
if [ "${tv_status}" -ne 0 ]; then
  say "TVCalib FAILED (exit ${tv_status}); see ${LOGDIR}/1_tvcalib.log"
  say "GS-HOTA needs metric pitch positions, so the arms are not started."
  exit 1
fi

missing=0
for s in ${SEQUENCES}; do
  [ -f "${TVROOT}/${s}/per_sample_output.json" ] || { say "TVCalib output missing for ${s}"; missing=1; }
done
if [ "${missing}" -ne 0 ]; then say "incomplete TVCalib output, stopping"; exit 1; fi
say "TVCalib complete"

# --- phase 2 and 3: the two policy arms -----------------------------------
conda activate jersey-yolo-ocr
arm_status=0
for arm in a b; do
  say "=== $([ "${arm}" = a ] && echo 2 || echo 3)/5 pipeline arm ${arm^^} (env: jersey-yolo-ocr) ==="
  PYTHONPATH="${FT}" python3 scripts/run_gsr_gs_hota_benchmark.py \
    --manifest "${MANIFEST}" \
    --arm "${arm}" \
    --model-path "${MODEL}" \
    --tvcalib-root "${TVROOT}" \
    --artifacts-root "${ARTIFACTS}" \
    --outputs-root "${OUTPUTS}" \
    --evaluation-root "${EVALROOT}" \
    --max-frames 750 \
    --resume \
    --allow-test \
    --wandb \
    > "${LOGDIR}/$([ "${arm}" = a ] && echo 2 || echo 3)_arm_${arm}.log" 2>&1
  status=$?
  if [ "${status}" -ne 0 ]; then
    say "arm ${arm^^} FAILED (exit ${status})"
    arm_status=1
  else
    say "arm ${arm^^} complete"
  fi
done

if [ "${arm_status}" -ne 0 ]; then
  say "at least one arm failed; aggregation would compare incomplete blocks, stopping"
  exit 1
fi

# --- phase 4: aggregate GS-HOTA, both arms --------------------------------
say "=== 4/5 aggregate GS-HOTA across sequences ==="
PYTHONPATH="${FT}" python3 scripts/aggregate_gsr_gs_hota_benchmark.py \
  --manifest "${MANIFEST}" \
  --arm-a-root "${EVALROOT}/arm_a" \
  --arm-b-root "${EVALROOT}/arm_b" \
  --output-dir "${EVALROOT}/aggregate" \
  > "${LOGDIR}/4_aggregate.log" 2>&1
say "aggregate exit $?"

# --- phase 5: oracle ablation (no GPU, pure re-analysis) ------------------
say "=== 5/5 oracle ablation on the test block ==="
PYTHONPATH="${FT}" python3 scripts/aggregate_gs_hota_ablation.py \
  --manifest "${MANIFEST}" \
  --artifacts-root "${ARTIFACTS}" \
  --arms a b \
  --output-dir evaluation_outputs/gs_hota_ablation/test_final \
  --allow-test \
  > "${LOGDIR}/5_ablation.log" 2>&1
say "ablation exit $?"

say "=== DONE ==="
say "aggregate : ${EVALROOT}/aggregate/aggregate.json"
say "ablation  : evaluation_outputs/gs_hota_ablation/test_final/tables.tex"
say "logs      : ${LOGDIR}"
