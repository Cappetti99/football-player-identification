# Football Player Identification

Broadcast-video football tracking and semantic player identification pipeline for thesis experiments on **Game State Reconstruction**.

The project does not stop at object detection or multi-object tracking. After producing player positions and tracklets, it tries to reconstruct the match state by assigning semantic information to every stable tracklet:

- team;
- player/referee/goalkeeper role;
- jersey number;
- real roster identity when evidence is strong enough;
- `unknown` when the evidence is weak or contradictory.

The central design choice is conservative identification: a missing identity is better than a wrong identity.

## What The Pipeline Does

```text
video
  -> YOLO detector
  -> ByteTrack tracking
  -> optional tracklet linking
  -> pitch transform / calibration fallback
  -> team assignment
  -> referee and goalkeeper colour cues from roster metadata
  -> semantic groups
  -> crop and metadata export
  -> jersey OCR with EasyOCR / MMOCR fallback and template matching
  -> roster-aware OCR filtering
  -> Hungarian tracklet-to-player assignment
  -> hard identity constraints
  -> annotated video and diagnostics
```

The output is an annotated video plus JSON/CSV artifacts that explain why each tracklet was, or was not, assigned to a real player.

## Semantic Groups

The pipeline separates the two real teams from richer semantic groups.

`team_id` is used for identity assignment:

```text
1 = team 1
2 = team 2
```

`semantic_group_id` is used for visualization and diagnostics:

```text
1 = team1 players
2 = team2 players
3 = team1 goalkeeper
4 = team2 goalkeeper
5 = referees
```

## Current Capabilities

- YOLO + ByteTrack tracking for `person` and `ball` detectors.
- Experimental StrongSORT-style tracker for comparison, not the recommended default.
- Tracklet linking with gates on temporal overlap, distance, team consistency and visual appearance.
- Team assignment from visual crop colours, with per-frame team evidence for detecting ID switches.
- Referee detection from roster-provided kit colours, for example `yellow` or `light_blue`.
- Goalkeeper detection from roster-provided kit colours, with optional team correction when the colour evidence is strong.
- Jersey OCR with:
  - multi-pass crop sampling;
  - EasyOCR backend;
  - optional MMOCR backend with EasyOCR fallback;
  - jersey font template matching using `docs/numberFont.jpg`;
  - crop-level aggregation before tracklet voting.
- Roster-aware OCR filtering:
  - removes or degrades jersey numbers that do not exist in the roster for that team;
  - can promote a valid roster candidate from the OCR distribution.
- Hungarian assignment from tracklets to roster players.
- Identity constraints:
  - no duplicate `player_id` in the same frame;
  - no duplicate `(team_id, jersey_number)` in the same frame;
  - no jersey number outside the team roster;
  - goalkeeper-only jersey numbers are cleared from non-goalkeepers;
  - non-goalkeeper jersey numbers are cleared from goalkeeper tracklets;
  - persistent per-frame team conflicts can split a contaminated `display_track_id`.
- W&B logging for runs and metadata artifacts.

## Repository Layout

```text
ft/
  calibration/              pitch transform and automatic fallback calibration
  export/                   CSV, JSON, crop and metadata artifact export
  features/                 team, referee, goalkeeper, OCR and visual features
  identity/                 roster parsing, Hungarian assignment and constraints
  linking/                  tracklet linking
  tracking/                 YOLO + ByteTrack and experimental StrongSORT wrapper
  utils/                    video IO and W&B helpers
  visualization/            overlay rendering

configs/
  default.yaml              main end-to-end configuration
  bytetrack_tracking_debug.yaml
  strongsort.yaml
  strongsort_tracking_debug.yaml

docs/
  numberFont.jpg            jersey-number font reference for template matching

scripts/
  train_yolo_gsr_full.py    SoccerNet-GSR conversion and YOLO training
  run_costume_videos.sh     helper for custom videos

tests/
  test_identity.py          lightweight regression tests
```

Local videos, outputs, W&B folders, model weights and generated artifacts are intentionally ignored by git.

## Installation

Basic editable install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

OCR extras:

```bash
python3 -m pip install easyocr pytesseract
```

On the thesis server:

```bash
cd /home/cappetti/FT
conda activate tesi
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

MMOCR is optional. If unavailable, the configured `mmocr_easyocr` mode falls back to EasyOCR where possible.

## Required Inputs

For a full run you need:

- a broadcast video;
- a YOLO detector checkpoint;
- optionally a roster JSON.

Expected custom-video structure:

```text
costume-video/<MatchName>/<MatchName>.mp4
costume-video/<MatchName>/<MatchName>.json
```

Example:

```text
costume-video/Roma-Verona/Roma-Verona.mp4
costume-video/Roma-Verona/Roma-Verona.json
```

`costume-video/` is ignored by git except for its placeholder files.

## Roster Format

The roster is a list of players and match officials.

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
      "role_hint": "striker"
    }
  },
  {
    "player_id": "team1_gk_01",
    "name": "Goalkeeper",
    "team_id": 1,
    "jersey_number": 1,
    "role": "goalkeeper",
    "position_prior": [8.0, 34.0],
    "visual_embedding": null,
    "metadata": {
      "team": "Team 1",
      "kit_color": "black"
    }
  },
  {
    "player_id": "referee_yellow",
    "name": "Referee",
    "team_id": null,
    "jersey_number": null,
    "role": "referee",
    "position_prior": null,
    "visual_embedding": null,
    "metadata": {
      "kit_color": "yellow"
    }
  }
]
```

Rules:

- `team_id` should be `1` or `2` for players.
- `team_id` should be `null` for referees.
- `jersey_number` must be between `1` and `99` when present.
- each team should contain each jersey number at most once.
- `role` is usually `player`, `goalkeeper`, `substitute` or `referee`.
- `position_prior` is optional and uses a 105 x 68 pitch coordinate system.

Supported named kit colours include:

```text
black
yellow
fluorescent_yellow
orange
red
blue
light_blue
```

Hex colours such as `#00aaff` are also accepted.

## Running One Video

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --video-path costume-video/Roma-Verona/Roma-Verona.mp4 \
  --model-path runs/detect/runs/ft_yolo_gsr/yolo26s_gsr_person_ball_768_e202/weights/best.pt \
  --output-path output_videos/costume-video/Roma-Verona_ft.mp4 \
  --artifacts-dir artifacts/costume-video/Roma-Verona_ft \
  --roster-path costume-video/Roma-Verona/Roma-Verona.json \
  --max-frames 3600
```

With W&B:

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --video-path costume-video/Inter-Atalanta/Inter-Atalanta.mp4 \
  --model-path runs/detect/runs/ft_yolo_gsr/yolo26x_gsr_person_ball_768_e20/weights/best.pt \
  --output-path output_videos/costume-video/Inter-Atalanta_yolo26x_ft.mp4 \
  --artifacts-dir artifacts/costume-video/Inter-Atalanta_yolo26x_ft \
  --roster-path costume-video/Inter-Atalanta/Inter-Atalanta.json \
  --max-frames 3600 \
  --wandb \
  --wandb-project football-tracking \
  --wandb-name Inter-Atalanta-yolo26x-ft
```

## Running A Folder

```bash
python3 -m ft.cli run \
  --config configs/default.yaml \
  --input-dir costume-video \
  --model-path runs/detect/runs/ft_yolo_gsr/yolo26x_gsr_person_ball_768_e20/weights/best.pt \
  --output-dir output_videos/costume-video \
  --artifacts-root artifacts/costume-video \
  --max-frames 3600 \
  --limit 3
```

The batch mode discovers video files directly inside the input directory. For nested per-match folders, run the per-video command or use a shell wrapper.

## Main Outputs

For a run under `artifacts/costume-video/<run_name>/metadata/`:

```text
*_tracklets.json                 final per-frame metadata
*_tracklets.csv                  final per-frame table
*_tracklet_summaries.csv         one row per display_track_id
*_candidate_scores.csv           Hungarian assignment candidate costs
*_identity_assignments.json      final tracklet identity assignments
*_jersey_ocr.json                OCR detections, votes and candidates
*_constraints.json               identity-constraint diagnostics
*_linking.json                   tracklet-linking diagnostics
*_referee_colour.json            referee colour diagnostics
*_goalkeeper_colour.json         goalkeeper colour diagnostics
```

Useful high-level diagnostics:

```bash
jq '{
  frame_team_conflict_count,
  display_track_split_count,
  duplicate_player_frame_count,
  remaining_duplicate_team_jersey_count,
  remaining_duplicate_player_id_count,
  goalkeeper_only_jersey_count,
  goalkeeper_invalid_jersey_count
}' artifacts/costume-video/<run_name>/metadata/<video_id>_constraints.json
```

## Training YOLO On SoccerNet-GSR

The training helper can convert SoccerNet-GSR into a YOLO dataset and train one of three label modes:

- `person_ball`: players, goalkeepers and referees are merged into `person`, plus `ball`;
- `person_only`: only the merged person class;
- `four_class`: `ball`, `goalkeeper`, `player`, `referee`.

Example, current person/ball setup:

```bash
python3 -u scripts/train_yolo_gsr_full.py \
  --gsr-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
  --output-dir /media/data-lie/cappetti/dataset/soccernet_gsr_yolo_person_ball \
  --mode person_ball \
  --base-model yolo26x.pt \
  --epochs 20 \
  --imgsz 768 \
  --batch 2 \
  --device 0 \
  --workers 4 \
  --project runs/ft_yolo_gsr \
  --name yolo26x_gsr_person_ball_768_e20
```

The detector is only the first stage. Better detection can improve crops and tracking continuity, but real player identification still depends on OCR, roster filtering, team consistency, role cues and assignment constraints.

## Testing

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 tests/test_identity.py
python3 -m ft.cli --help
python3 -m ft.cli run --help
```

## Current Limitations

- Broadcast cuts and camera changes can break temporal continuity.
- ByteTrack IDs are not true player identities.
- Long videos should be evaluated by action segment or scene cut when possible.
- The StrongSORT wrapper is experimental and currently not the recommended tracker.
- OCR remains sensitive to crop quality, pose, motion blur and occlusion.
- Pitch calibration has an automatic fallback but is not a full metric field-keypoint model.

## Thesis Direction

The project is aimed at evaluating weak-signal identity reconstruction in football broadcast video:

- robust person detection and tracking;
- team and role semantics;
- jersey-number OCR;
- roster-aware filtering;
- visual and trajectory cues;
- conservative assignment to real players;
- explicit diagnostics for every failure mode.

This makes the repository useful both as an experimental pipeline and as a source of artifacts for thesis analysis.
