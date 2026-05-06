# Football Player Identification

Modular football broadcast-video pipeline for thesis experiments on Game State Reconstruction and player identification.

The goal is not only to detect and track players, but to assign each stable tracklet to a real player identity using weak evidence: team, jersey number OCR, pitch position, visual descriptors, role priors, goalkeeper/referee cues and trajectory continuity.

## Pipeline

```text
video
  -> YOLO detector
  -> ByteTrack multi-object tracking
  -> tracklet linking into display_track_id
  -> pitch transform / calibration
  -> team assignment
  -> five semantic groups
  -> crop and metadata export
  -> jersey OCR with candidate distribution
  -> roster-aware filtering
  -> Hungarian player identification
  -> annotated video and diagnostics
```

The central object is `display_track_id`. Raw tracker IDs may fragment; `display_track_id` is the stable unit used for OCR aggregation, identity assignment and diagnostics.

## Semantic Groups

`team_id` remains the two real football teams used by identity assignment. A separate `semantic_group` is exported for visualization and analysis:

```text
1 team1_players
2 team2_players
3 team1_goalkeeper
4 team2_goalkeeper
5 referees
```

## Main Features

- YOLO + ByteTrack tracking for players, goalkeepers, referees and ball.
- Tracklet linking to reduce raw tracker fragmentation.
- Manual or automatic fallback pitch transform.
- Conservative team assignment from torso colour.
- Referee colour diagnostics using a restricted palette: yellow, light blue, black and red.
- Jersey OCR with multi-pass crop sampling and OCR candidate distributions.
- Roster-aware OCR: numbers outside the team roster can be degraded, while roster-compatible alternatives can be promoted.
- Hungarian assignment from tracklets to roster players.
- Metadata exports for debugging every identity decision.
- Optional Weights & Biases logging and alerts.

## Repository Layout

```text
ft/                         Python package
  calibration/              pitch homography and automatic fallback
  features/                 team, referee, OCR, visual descriptors, groups
  identity/                 roster parsing and Hungarian assignment
  linking/                  tracklet linking
  tracking/                 YOLO + ByteTrack wrapper
  visualization/            annotated video overlay
configs/                    default config and example roster/calibration
scripts/                    training and run helpers
tests/                      lightweight regression tests
costume-video/              local custom videos, ignored by git
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

For OCR:

```bash
python3 -m pip install easyocr
```

On the SSH server with Conda:

```bash
cd /home/cappetti/FT
conda activate tesi
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## Inputs

You need:

- a video file;
- a YOLO model, for example `best_yolo26x_gsr_light.pt`;
- optionally, a roster JSON for real player identification.

Model weights and videos are intentionally ignored by git.

## Roster JSON

The roster is a list of players:

```json
[
  {
    "player_id": "team1_10",
    "name": "Example Player",
    "team_id": 1,
    "jersey_number": 10,
    "role": "player",
    "position_prior": [46.0, 24.0],
    "visual_embedding": null,
    "metadata": {
      "team": "Team 1",
      "role_hint": "striker",
      "starter": true
    }
  }
]
```

Rules:

- `team_id` is `1` or `2`.
- `jersey_number` must be between `1` and `99`.
- the same team cannot contain two players with the same jersey number.
- `role` can be `player`, `goalkeeper` or another descriptive value such as `substitute`.
- `position_prior` is optional and uses a 105 x 68 metre pitch coordinate system.

## Running One Video

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --video-path costume-video/Roma-Verona/Roma-Verona.mp4 \
  --model-path best_yolo26x_gsr_light.pt \
  --output-path output_videos/Roma-Verona_ft.mp4 \
  --artifacts-dir artifacts/Roma-Verona_ft \
  --roster-path costume-video/Roma-Verona/Roma-Verona.json \
  --max-frames 1800
```

## Running A Folder Of Custom Videos

Put videos in `costume-video/`, then run:

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --input-dir costume-video \
  --model-path best_yolo26x_gsr_light.pt \
  --output-dir output_videos/costume-video \
  --artifacts-root artifacts/costume-video \
  --roster-path configs/roster.example.json \
  --max-frames 1800
```

Shortcut:

```bash
MAX_FRAMES=1800 ./scripts/run_costume_videos.sh
```

## Weights & Biases

Do not store API keys in the repository. Log in once on the server:

```bash
wandb login
```

Then add W&B flags to a run:

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --video-path costume-video/Int-Ata/Int-Ata.mp4 \
  --model-path best_yolo26x_gsr_light.pt \
  --output-path output_videos/Int-Ata_ft.mp4 \
  --artifacts-dir artifacts/Int-Ata_ft \
  --roster-path costume-video/Int-Ata/Int-Ata.json \
  --max-frames 3600 \
  --wandb \
  --wandb-project football-tracking \
  --wandb-name Int-Ata
```

FT logs run metrics, metadata artifacts and W&B alerts on success or failure. Video upload is disabled by default because output videos can be large.

## Outputs

For a run named `artifacts/Roma-Verona_ft`, the most useful files are:

```text
metadata/*_tracklets.json
metadata/*_tracklets.csv
metadata/*_tracklet_summaries.csv
metadata/*_candidate_scores.csv
metadata/*_identity_assignments.json
metadata/*_jersey_ocr.json
metadata/*_referee_colour.json
crops/<video_id>/
```

`candidate_scores.csv` is the main debugging file: it shows the cost components for each tracklet-player candidate pair.

## Testing

```bash
PYTHONPATH=. python3 tests/test_identity.py
python3 -m ft.cli --help
python3 -m ft.cli run --help
```

## Training YOLO On SoccerNet-GSR

If the converted YOLO dataset exists:

```bash
python3 scripts/train_yolo_gsr.py \
  --data-yaml /media/data-lie/cappetti/dataset/soccernet_gsr_yolo/data.yaml \
  --base-model yolo26x.pt \
  --epochs 80 \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

This trains the detector only. Identity quality still depends on downstream features: team assignment, crop quality, jersey OCR, pitch transform, roster quality and the Hungarian cost matrix.
