# Evaluation Results

This file records the main quantitative results used during the thesis
experiments. The SoccerNet-GSR jersey OCR runs are controlled evaluations with
ground-truth bounding boxes and jersey numbers. The custom-video runs are
pipeline diagnostics without manual ground truth, so they should be read as
coverage and consistency checks rather than accuracy measurements.

## SoccerNet-GSR Jersey OCR

Dataset/task:

- split: `val`
- source labels: `Labels-GameState.json`
- evaluated object: one GSR tracklet with a jersey-number annotation
- crops: ground-truth GSR boxes, up to 20 crops per tracklet
- metric definitions:
  - `coverage`: assigned tracklets / evaluated tracklets
  - `accuracy_assigned`: correct assigned tracklets / assigned tracklets
  - `accuracy_tracklet`: correct assigned tracklets / all evaluated tracklets

### 10 Sequences

| Run | Tracklets | Assigned | Coverage | Accuracy Assigned | Accuracy Tracklet |
| --- | ---: | ---: | ---: | ---: | ---: |
| EasyOCR | 157 | 105 | 66.88% | 51.43% | 34.39% |
| EasyOCR + template | 157 | 105 | 66.88% | 51.43% | 34.39% |
| MMOCR | 157 | 17 | 10.83% | 88.24% | 9.55% |
| MMOCR + EasyOCR | 157 | 105 | 66.88% | 54.29% | 36.31% |

Best balanced threshold sweep:

| Run | Confidence | Head Conf. | Margin | Votes | Assigned | Coverage | Accuracy Assigned | Accuracy Tracklet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EasyOCR | 0.20 | 0.55 | 0.05 | 2 | 86 | 54.78% | 61.63% | 33.76% |
| MMOCR | 0.20 | 0.55 | 0.00 | 2 | 16 | 10.19% | 87.50% | 8.92% |
| MMOCR + EasyOCR | 0.20 | 0.55 | 0.05 | 2 | 87 | 55.41% | 64.37% | 35.67% |

### 50 Sequences

| Run | Tracklets | Assigned | Coverage | Accuracy Assigned | Accuracy Tracklet |
| --- | ---: | ---: | ---: | ---: | ---: |
| EasyOCR | 830 | 574 | 69.16% | 37.11% | 25.66% |
| MMOCR + EasyOCR | 830 | 583 | 70.24% | 43.05% | 30.24% |

Best balanced threshold sweep:

| Run | Confidence | Head Conf. | Margin | Votes | Assigned | Coverage | Accuracy Assigned | Accuracy Tracklet |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EasyOCR | 0.20 | 0.55 | 0.00 | 2 | 444 | 53.49% | 44.14% | 23.61% |
| MMOCR + EasyOCR | 0.20 | 0.55 | 0.00 | 2 | 448 | 53.98% | 49.55% | 26.75% |

Interpretation:

- MMOCR alone is high precision but low coverage.
- Combining MMOCR and EasyOCR improves both assigned accuracy and tracklet
  accuracy while preserving EasyOCR's coverage.
- The single-font template matcher did not improve the controlled evaluation
  when kept at low weight. Earlier stronger weighting over-predicted `1`, so
  template matching is disabled by default for the current experiments.

## Custom Video Diagnostics

These runs do not have manual ground truth. The values below describe metadata
coverage and identity-assignment behavior.

### Inter-Atalanta

| Run | Backend | Frames | Rows | Unique IDs | Jersey Rows | Assigned Rows | OCR Tracklets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Inter-Atalanta_yolo26x_semantic_v10_gksingleton_3600f` | EasyOCR + template | 294 | 3745 | 18 | 2880 | 2190 | 16 |
| `Inter-Atalanta_yolo26x_semantic_v17_mmocr_easyocr_3600f` | MMOCR + EasyOCR | 294 | 3745 | 18 | 3165 | 2190 | 15 |

Both runs kept duplicate identity constraints clean:

- `duplicate_player_frame_count`: 0
- `remaining_duplicate_team_jersey_count`: 0
- `remaining_duplicate_player_id_count`: 0

The MMOCR + EasyOCR run preserved the same final assigned identities while
raising jersey-row coverage. The goalkeeper was assigned through
`goalkeeper_roster_singleton`; OCR evidence for the goalkeeper number remained
weak, so overlay configuration can choose whether to display roster-derived
numbers after identity assignment.

### Monza-Milan

| Run | Backend | Frames | Rows | Unique IDs | Jersey Rows | Assigned Rows | OCR Tracklets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Monza-Milan_yolo26x_semantic_v15_conservativejersey_3600f` | EasyOCR fallback | 217 | 4103 | 25 | 2630 | 1342 | 18 |
| `Monza-Milan_yolo26x_semantic_v18_mmocr_easyocr_overlayid_3600f` | MMOCR + EasyOCR | 217 | 4103 | 25 | 3231 | 1741 | 19 |

The Monza-Milan run is the clearest custom-video improvement observed so far:

- `jersey_rows`: +601
- `assigned_rows`: +399
- duplicate identity constraints remained clean

## Current Recommendation

Use `mmocr_easyocr` for experiments where the OpenMMLab stack is available.
Use EasyOCR fallback when running in the lighter `tesi` environment. Keep
template matching disabled unless testing a known video-specific font.
