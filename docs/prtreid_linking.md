# FT PRTReID tracklet linking

The PRTReID linker is an additive second pass. The legacy linker runs first and
remains the baseline. PRTReID can merge additional display tracklets but cannot
undo legacy links or assign team, role, jersey, or roster identity.
The deep prototypes are isolated from downstream identity evidence: after the
second linker, FT restores the baseline `hsv_lab_gradient` visual descriptor.
Therefore a run with zero accepted PRTReID links must preserve baseline identity
metrics.

## Candidate run

```bash
cd /home/cappetti/FT

RUN=Int-Ata_identity_evidence_v1_prtreid_linking_cutsensitive_1200f
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 -m ft.cli run \
  --config configs/default_realvideo_identity_evidence_v1_prtreid_linking_cutsensitive_resetbytetrack.yaml \
  --video-path costume-video/Int-Ata/Int-Ata.mp4 \
  --model-path best_yolo26x_gsr_light.pt \
  --output-path output_videos/costume-video/${RUN}.mp4 \
  --artifacts-dir artifacts/costume-video/${RUN} \
  --roster-path costume-video/Int-Ata/Int-Ata.json \
  --max-frames 1200
```

Start with 50–200 frames before the 1200-frame run. The candidate config keeps
cross-scene links disabled while still exporting cross-scene candidates for
manual calibration.

## Guided pair audit

```bash
python3 scripts/build_prtreid_pair_audit.py \
  --run "$RUN" \
  --video-id Int-Ata \
  --max-pairs 40
```

Review the generated contact sheets and edit `labels.csv` with `same`,
`different`, or `uncertain`. Repeat on `Inter-Juve`; use `Inter-Atalanta` when
that run is unavailable.

```bash
python3 scripts/calibrate_prtreid_linker.py \
  evaluation_outputs/prtreid_pair_audit/RUN_INT_ATA/labels.csv \
  evaluation_outputs/prtreid_pair_audit/RUN_SECOND_VIDEO/labels.csv \
  --output-dir evaluation_outputs/prtreid_linker_calibration/v1
```

The generated YAML enables a policy only when its accepted labeled pairs have
zero false positives and at least one true positive. The overall pretrained
gate additionally requires 20% recall and three correct links.

## Ground-truth fine-tuning

Fine-tuning is allowed only from SoccerNet-GSR ground truth. FT identity output
and pair-audit labels are validation data, not training labels.

```bash
python3 scripts/export_prtreid_dataset.py \
  --gsr-dir /path/to/SoccerNet-GSR \
  --output-dir datasets/prtreid_ft_v1

python3 scripts/train_prtreid.py \
  --dataset-dir datasets/prtreid_ft_v1 \
  --output-dir models/reid/prtreid_ft_linking_v1 \
  --initial-weights models/reid/prtreid-soccernet-baseline.pth.tar \
  --epochs 20
```

The exporter enforces sequence-disjoint train and validation splits. Training
uses identity-only loss and same-team batches. Checkpoints are ranked by
same-team recall at zero observed false positives, then rank-1 and mAP.

## Promotion gate

- scene cuts remain `[294, 517, 715, 925, 1052]` on the 1200-frame `Int-Ata` run;
- duplicate identities remain zero;
- no manually labeled false merge is accepted;
- `assigned_rows >= 5986` and `unknown_rows <= 9133`;
- no regression on the second video;
- cross-scene linking stays disabled unless it adds a correct bridge with no
  labeled error.

## Scoped Int-Ata profile

`configs/prtreid_linking_conservative_int_ata_v1.yaml` is the validated,
opt-in profile for the 1200-frame Int-Ata clip. It accepts the audited bridges
`69 -> 94` and `100 -> 111`, preserves 5,426 direct assignments and 9,693
unknown rows from the matched propagation-disabled control, reduces display
IDs from 85 to 83, preserves the five expected cuts, and leaves every
duplicate counter at zero.

The historical 5,986 assignment figure included 560 identity-propagated rows,
despite the original audit reporting zero propagated rows. The audit now also
checks `identity_evidence.status`, and supports `--assigned-floor 5426
--unknown-ceiling 9693` for a like-for-like propagation-disabled comparison.
Inter-Juve and Inter-Atalanta contained no cuts in the evaluated clips, so this
profile must not be treated as a generally calibrated cross-scene policy.
