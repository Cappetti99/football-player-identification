# Additional Thesis Metrics

Metrics added on top of the existing detection/tracking/jersey benchmarks
(`docs/gsr_detection_tracking_benchmark.md`, `docs/evaluation_results.md`),
each with the exact artifact and command it was derived from. Every number
here has been verified against a real artifact on the remote machine
(`cappetti@lie:~/FT`); none is estimated.

## MOTA

**Value:** MOTA = 0.872 (12-sequence GSR validation pilot, `display_*`
surface, i.e. after FT's tracklet linker).

**Source artifact:**
```
evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1_aggregate/aggregate.json
```
Same file already used for the tracking-bottleneck slide (DetA/AssA/HOTA).
`macro.display_mota.mean` was already computed by
`scripts/aggregate_gsr_detection_tracking_benchmark.py` (metric key
`display_mota` -> `tracking.mota`, computed by
`ft/evaluation/gsr_detection_tracking.py::tracking_summary` as the standard
CLEAR MOT formula `1 - (FN + FP + id_switches) / GT`); it had never been
pulled into a slide before.

**Command used:**
```bash
python scripts/plot_jersey_gsr_tracking_bottleneck.py \
  --aggregate evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1_aggregate/aggregate.json \
  --output fig_tracking_bottleneck_v2.png
```
(`scripts/plot_jersey_gsr_tracking_bottleneck.py` extended to read/plot
`display_mota` alongside DetA/AssA/HOTA.)

**Full readout for context:**
```
DetA=0.710  AssA=0.517  HOTA=0.602  MOTA=0.872
recall@0.50=95.8%  AP50=0.951  AP75=0.619
ID switches=1298  fragmentations=1941
```

**Why it matters for the thesis narrative:** MOTA is dominated by detection
quality (FP/FN) with only a linear penalty for identity switches, so it stays
high (0.872) even where association is weak. It masks exactly the problem
HOTA exposes: AssA=0.517 and the 19-point DetA-to-HOTA gap show that
maintaining identity over time is the real bottleneck, not finding players. A
single surface-level metric (MOTA) would have hidden the problem that a
finer one (HOTA) reveals.

## HOTA — which detection is matched to which

Not a new number, a clarification of the existing HOTA computation in
`ft/evaluation/gsr_detection_tracking.py::hota_summary`, since the matching
rule is not the same as MOTA's.

MOTA/tracking_summary uses per-frame CLEAR MOT matching: continuity from the
previous frame's assignment is preferred, then a per-frame Hungarian
assignment on raw bbox IoU, gated at `threshold` (0.50).

HOTA uses TrackEval's own global-alignment procedure instead:

1. For every candidate (GT track, predicted track) pair, sum a per-frame IoU
   term across **the whole sequence** into a `global_alignment` score — how
   consistently the two tracks overlap over time, not just in one frame.
2. For each of 19 alpha thresholds (0.05 to 0.95, step 0.05), run a per-frame
   Hungarian assignment maximizing `global_alignment x frame_IoU`, keeping
   only pairs with `frame_IoU >= alpha`.
3. `DetA_alpha`, `AssA_alpha` are computed from those matches;
   `HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)`; the reported HOTA is the mean
   over the 19 alphas.

Consequence: a detection can be assigned to a track other than the one with
the highest IoU in that specific frame, if a different track has better
long-run temporal consistency with it. This is deliberate — it is what makes
HOTA sensitive to identity consistency rather than just per-frame overlap.

## Pose estimator: Missing vs. Failure

**Source artifact (the same run already cited as "763/865 reliable" in the
thesis deck):**
```
evaluation_outputs/jersey_number_region/frozen_region_coverage_gap_v3/pose_orientation_v1/crop_pose_orientation.csv
```
Checkpoint: `yolov8x-pose.pt` (generic, zero training on GSR/football).
Confirmed via `crop_pose_orientation_summary.json` (`pose_checkpoint:
yolov8x-pose.pt`, `pose_detected: 763/865`) before use — a second run in the
same directory tree (`pose_orientation_v2_yolo26`, checkpoint
`yolo26x-pose.pt`, 808/865) exists but is **not** the one already cited and
was not used here.

Previously reported as one number ("763/865 reliable", i.e. 12% "unreliable").
That 12% actually splits into two structurally different failure modes,
visible in the row-level CSV (`pose_detected`, `orientation` columns) but
never separated in the summary before today:

| | count | % of 865 |
|---|---|---|
| reliable (used in the slide) | 751 | 86.8% |
| **missing** — pose model finds no person at all in the crop | 102 | 11.8% |
| **failure** — person found, shoulder keypoints too low-confidence to call an orientation | 12 | 1.4% |

**Definitions:**
- `missing`: `pose_detected == False` (no keypoints returned for this crop at all).
- `failure`: `pose_detected == True` and `orientation == "undetermined"` (a
  person was found, but `left_shoulder_conf`/`right_shoulder_conf` did not
  both clear `keypoint_confidence_threshold`, 0.5).

**Derivation (no re-inference needed — computed from the existing per-row CSV):**
```python
import csv

def truthy(value):
    return str(value).strip().lower() == "true"

path = "evaluation_outputs/jersey_number_region/frozen_region_coverage_gap_v3/pose_orientation_v1/crop_pose_orientation.csv"
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))

total = len(rows)
pose_detected = sum(1 for row in rows if truthy(row["pose_detected"]))
missing = total - pose_detected
failure = sum(1 for row in rows if truthy(row["pose_detected"]) and row["orientation"] == "undetermined")
reliable = pose_detected - failure
```
Same logic is now also computed by `scripts/audit_crop_pose_orientation.py`'s
own summary (fields `missing_count`/`missing_rate`, `failure_count`/
`failure_rate`, `reliable_count`/`reliable_rate`), for any future re-run of
that script — no need to re-derive it manually next time.

**Why it matters:** almost all of the unreliability (11.8 of 13.2 points) is
`missing`, not `failure`. The bottleneck is not "the shoulders are ambiguous"
— it is that the pose model's own person detector frequently fails to find
a person at all in these crops, most likely because of crop size/quality
(consistent with the region-crop resolution issue found independently today
via the CTC super-resolution test, though that test showed upscaling alone
does not fix this kind of failure).

## GS-HOTA

From Somers et al., "SoccerNet Game State Reconstruction: End-to-End Athlete
Tracking and Identification on a Minimap" (CVPR24 CVSports workshop,
arXiv:2404.11335).

```
Sim_GS-HOTA(P, G) = LocSim(P, G) x IdSim(P, G)

LocSim(P, G) = exp( ln(0.05) * ||P - G||_2^2 / tau^2 ),  tau = 5 meters
IdSim(P, G)  = 1 if team, role AND jersey number all match, else 0
               (attributes absent in G, e.g. a referee's jersey number, are ignored)
```

`Sim_GS-HOTA` replaces bounding-box IoU inside the same HOTA procedure
described above (global alignment, 19 `alpha` thresholds, `sqrt(DetA*AssA)`).
Requires pitch positions in meters (not bounding boxes) for both GT and
predictions, plus the final team/role/jersey-number decision per subject.

### Implementation

`ft/evaluation/gsr_detection_tracking.py::gs_hota_similarity` /
`gs_hota_summary`, mirroring `hota_summary`'s structure with
`gs_hota_similarity` in place of `bbox_iou`. Wired into
`scripts/evaluate_ft_gsr.py` (`add_gs_hota`), which already loads
`gt_position_pitch`/`pred_position_pitch` and the role/team/jersey attribute
pairs for its existing team/role/jersey accuracy sections — no new data
loading needed. 10 unit tests in `tests/test_gs_hota.py` (identical
position+attributes -> 1.0; any wrong attribute -> 0.0; a missing GT
attribute, e.g. a referee's jersey, is ignored rather than penalized; the
tau=5m boundary matches the paper's Fig. 2 exactly: 0.05 at distance=tau;
identity swaps hurt AssA, not DetA/LocA).

One integration fix was needed: GT team is `"left"`/`"right"`
(`normalize_team`), predictions carry FT's own numeric `team_id`
(`normalize_pred_team`) — not directly comparable. `scripts/evaluate_ft_gsr.py`
already computes a `left`/`right` <-> `team_id` mapping from matched pairs
for its own team-accuracy report (`choose_team_mapping`); GS-HOTA reuses the
same mapping, applied to every predicted row (not just IoU-matched ones,
since GS-HOTA scores unmatched rows too).

### First real measurement (SNGS-082, 750 frames, arm A vs arm B)

Verified no regression first: re-ran the pre-existing
`gsr_valid_pilot12_baseline_v1` artifact for SNGS-082 through the updated
evaluator -- HOTA/DetA/AssA/MOTA came back byte-identical to the values
already in the frozen `aggregate.json` (0.572973 / 0.728009 / 0.451225 /
0.929542). GS-HOTA was exactly 0.0 on that run, for two compounding reasons,
not a bug: (1) that benchmark profile runs jersey OCR disabled
(`pred_jersey` is null on every row), so IdSim fails wherever GT has a
jersey number; (2) that profile uses the automatic field-quad calibration
fallback (~29m mean pitch error), far beyond `tau=5m`, which collapses
LocSim to numerically indistinguishable from 0 regardless of IdSim.

To get a real number, both problems needed fixing at once: a pipeline run
with jersey OCR **and** TVCalib **and** the region-CTC decision policy all
enabled together -- something no existing artifact combined, since they
were developed and validated in separate experiment tracks (the 12-sequence
detection/tracking pilot used TVCalib without jersey OCR; the jersey policy
held-out experiment used jersey OCR without TVCalib).

Also found and fixed in passing: `configs/gsr_jersey_pipeline_baseline_apply_v1.yaml`
(the pre-registered arm A / conservative SAR-primary profile) did not
declare `decision_policy` explicitly -- it inherited whatever
`ft/config.py`/`configs/default.yaml` currently defaults to, which has since
changed (region CTC promoted ahead of SAR as the pipeline default). Without
an explicit override, arm A would have silently resolved to the same policy
as arm B, collapsing the A/B comparison. Fixed by declaring
`sources: [jersey_ocr_primary, jersey_region_ctc]` explicitly in that file,
independent of whatever the global default says.

New configs, both layered on the existing arm A/B pipeline-apply profiles
plus TVCalib (`evaluation_outputs/tvcalib_gsr_val_10s/SNGS-082/per_sample_output.json`,
already computed for the 12-sequence pilot):
- `configs/gsr_jersey_pipeline_baseline_apply_tvcalib_v1.yaml` (arm A: SAR primary)
- `configs/gsr_jersey_pipeline_ctc_primary_apply_tvcalib_v1.yaml` (arm B: CTC primary)

Run via `ft.cli run --video-path input_videos/soccernet_gsr/SNGS-082.mp4
--max-frames 750` (no roster, matching the GSR benchmark convention), then
evaluated with `scripts/evaluate_ft_gsr.py` against SNGS-082's
`Labels-GameState.json`.

| | arm A (SAR primary) | arm B (CTC primary) |
|---|---|---|
| HOTA / DetA / AssA / MOTA | 0.572973 / 0.728009 / 0.451225 / 0.929542 | identical (same detection/tracking) |
| pitch error (mean / median) | 1.434m / 1.121m | identical (same TVCalib) |
| jersey coverage (tracklet-level, visible) | 70% | 90% |
| jersey accuracy (tracklet-level, all-visible) | 20% | 65% |
| **GS-HOTA** | **0.1826** | **0.3301** |
| GS-DetA | 0.0807 | 0.2405 |
| GS-AssA | 0.4130 | 0.4531 |
| GS-LocA | 0.8608 | 0.8548 |

GS-HOTA nearly doubles from arm A to arm B, driven mostly by GS-DetA (~3x):
with IdSim requiring team+role+jersey to all be correct, a wrong jersey
number turns an otherwise-correct tracklet into a GS-HOTA false positive,
not just a jersey-accuracy error -- which is exactly the end-to-end
sensitivity GS-HOTA was designed to have. Pitch error (1.43m mean) matches
the ~1.10m median already cited for TVCalib in the thesis deck, confirming
calibration is working as expected; HOTA/DetA/AssA/MOTA staying
byte-identical between arms confirms TVCalib and the jersey decision policy
do not leak into detection/tracking.

**Caveat on this single-sequence check:** SNGS-082 belongs to the
12-sequence detection/tracking pilot (the one with TVCalib already computed,
`evaluation_outputs/tvcalib_gsr_val_10s/`) -- it is **not** one of the 7
sequences in the jersey-policy held-out block
(`evaluation/jersey_heldout_manifests/jersey_policy_heldout_v1.json`, SNGS-089/
091/092/093/094/095/096). Those are two different sequence sets from two
different experiment tracks; the aggregate result below is a comparison on
the TVCalib pilot 12, not an extension of the pre-registered jersey held-out
result.

### Aggregated result, all 12 TVCalib-pilot sequences

All 12 sequences in the pilot already had TVCalib computed, so the same arm
A/B comparison was run on all of them (not just SNGS-082), via
`scripts/run_gsr_gs_hota_benchmark.py` (per-sequence pipeline run, jersey
OCR + region CTC + TVCalib, one arm at a time) and
`scripts/aggregate_gsr_gs_hota_benchmark.py` (macro mean/median/bootstrap CI
per arm, plus a paired per-sequence delta B-A, bootstrapped over sequences
-- same style as the jersey held-out policy experiment's paired comparison).

| | arm A (SAR primary) | arm B (CTC primary) | Δ (B − A), 95% CI |
|---|---|---|---|
| GS-HOTA | 0.2097 | 0.2870 | **+0.0772**, [0.0476, 0.1067] |
| GS-DetA | 0.1168 | 0.1997 | **+0.0829**, [0.0419, 0.1293] |
| GS-AssA | 0.4184 | 0.4542 | **+0.0358**, [0.0174, 0.0557] |
| GS-LocA | 0.8258 | 0.8350 | +0.0092, [0.0017, 0.0190] |

All four 95% bootstrap CIs on the paired delta exclude zero: unlike the
single-sequence SNGS-082 check (where GS-HOTA can swing by two orders of
magnitude between sequences purely from calibration variance, per the
paper's own Fig. 5), this is a real, reproducible effect across 12
sequences, not sequence-level noise. GS-DetA remains the dominant driver, as
on SNGS-082 alone.

Context only, not a rigorous comparison (different pipeline, different
sequences -- see the caveats on comparability discussed when this section
was first written): arm A's mean GS-HOTA (0.2097) lands close to the
paper's own `Full Baseline: 22.26` (0.2226, Table 2) for their GSR-Baseline
pipeline (YOLOv8+StrongSORT+PRTreID+MMOCR+TVCalib). Reassuring as an order-
of-magnitude sanity check, nothing more.

Artifacts: `evaluation_outputs/gs_hota_benchmark/{arm_a,arm_b}/<sequence>/summary.json`
per sequence, `evaluation_outputs/gs_hota_benchmark/aggregate/{per_sequence.csv,aggregate.json}`
for the aggregate.
