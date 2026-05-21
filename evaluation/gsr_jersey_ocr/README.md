# SoccerNet-GSR Jersey OCR Evaluation

This folder contains a controlled evaluation for jersey-number recognition on
SoccerNet-GSR. It intentionally bypasses YOLO, ByteTrack and Hungarian
assignment by using the ground-truth GSR bounding boxes and track IDs.

The goal is to measure the OCR module itself before evaluating the full
end-to-end pipeline.

## What It Evaluates

- Ground truth: `Labels-GameState.json` annotations with `attributes.jersey_number`.
- Prediction: `ft.features.jersey_ocr.JerseyOCR` over crops extracted from GT boxes.
- Unit of evaluation: one SoccerNet-GSR tracklet.

The script reports:

- `coverage`: predicted tracklets / evaluated tracklets.
- `accuracy_assigned`: correct predictions among predicted tracklets.
- `accuracy_tracklet`: correct predictions over all evaluated tracklets.
- confusion rows for wrong assigned numbers.

## Quick Run

Requirements: run this from the project environment used for the main pipeline
(`conda activate tesi`), with OpenCV and the selected OCR backend installed.

```bash
cd /home/cappetti/FT
conda activate tesi

python3 evaluation/gsr_jersey_ocr/run_eval.py \
  --gsr-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
  --output-dir evaluation_outputs/gsr_jersey_ocr/easyocr_template_val_smoke \
  --split val \
  --max-sequences 2 \
  --max-tracklets 100 \
  --backend easyocr \
  --easyocr-gpu \
  --template-matching \
  --template-font-image docs/numberFont.jpg \
  --template-weight 0.03
```

For a larger run, remove `--max-sequences` and `--max-tracklets`.

## Outputs

```text
evaluation_outputs/gsr_jersey_ocr/<run_name>/
  config.json
  metrics.json
  predictions.csv
  confusion.csv
  threshold_sweep.csv
  ocr_diagnostics.json
  crops/
```

`predictions.csv` is the main file for analysis. It contains GT jersey,
predicted jersey, confidence, votes, sequence, role and team.

`threshold_sweep.csv` re-scores the saved predictions under stricter confidence,
head-confidence, margin and vote gates. Use it to choose the reliability
thresholds for identity assignment without rerunning OCR.

## Weights & Biases

`run_eval.py` supports W&B logging:

```bash
python3 evaluation/gsr_jersey_ocr/run_eval.py ... \
  --wandb \
  --wandb-project football-tracking \
  --wandb-name gsr-val-easyocr
```

The suite enables W&B by default. Override `WANDB=false` to disable it, or set
`WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_LOG_ARTIFACTS`,
`WANDB_ALERT_ON_FINISH` and `WANDB_ALERT_ON_FAILURE` from the shell.

The MMOCR evaluation is skipped by default because it requires the OpenMMLab
stack. Set `RUN_MMOCR=true` only after installing MMOCR in the environment. The
suite then runs three OCR variants:

- `mmocr`: DBNet text detection + SAR recognition, following the SoccerNet
  Game State baseline.
- `mmocr_rec`: SAR recognition on the same fixed jersey crop variants used by
  EasyOCR, without MMOCR text detection.
- `mmocr_easyocr`: MMOCR and EasyOCR proposals combined in the same tracklet
  voting stage, so their contribution can be compared against each backend
  alone.

Template parameters can be controlled with `TEMPLATE_MIN_SCORE`,
`TEMPLATE_WEIGHT` and `TEMPLATE_MAX_CANDIDATES`.

## Notes

- This is not an end-to-end tracking metric. It uses GT boxes to isolate OCR.
- Do not use final video metadata for OCR evaluation: after identity assignment,
  `jersey_number` may already be roster-corrected.
- Use this before testing GS-HOTA or bbox-IoU matching on the full pipeline.
