# Pre-registration: final evaluation on the SoccerNet-GSR test split

Written and frozen **before any evaluation on the test split was executed**
for this experiment. Its purpose is to fix, in advance, what will be run, what
will be reported, and what will be concluded under every possible outcome, so
that looking at the test numbers is reporting rather than selection.

## 1. Prior disclosure

The GSR test split is **not pristine**. It was used once before, to compare
two checkpoints of the region CTC recogniser differing only in SJN-210k
pretraining (774 tracklets; accuracy on assigned 41.2% -> 73.1%). That result
is reported in this thesis. The present evaluation is therefore the **second**
use of the test split, and is disclosed as such.

## 2. What has already been decided, and where

The system configuration evaluated here is **already fixed**, and was chosen
on validation data, not on test:

* The jersey decision policy promotes the region CTC ahead of the primary SAR
  OCR (`decision_policy.jersey_number.sources: [jersey_region_ctc,
  jersey_ocr_primary]`). This is arm B.
* The choice was made on a pre-registered, sequence-disjoint held-out block of
  7 GSR `valid` sequences
  (`evaluation/jersey_heldout_manifests/jersey_policy_heldout_v1.json`), with
  both arms declared before any joint result was observed. Arm B won on the
  offline surface (+0.378 accuracy, 95% CI [0.275, 0.496], exact binomial
  p = 8.4e-12) and on the pipeline surface (+0.298, 95% CI [0.208, 0.374],
  exact McNemar p = 1.9e-8).
* It was independently confirmed end-to-end on a disjoint 12-sequence `valid`
  pilot with GS-HOTA (+0.0772, 95% CI [0.0476, 0.1067]).

No architectural or configuration change will be made after this document is
frozen. The evaluation below is measurement only.

## 3. Sequence sample

Drawn from the 49 test sequences by `random.Random(20260803).sample` over the
sorted sequence identifiers, size 12, recorded in
`evaluation/detection_tracking_manifests/test_final_v1.json` together with the
seed, the pool size and the selection procedure. Drawn **before** any test
sequence was evaluated.

Selected: SNGS-120, SNGS-125, SNGS-127, SNGS-131, SNGS-133, SNGS-135,
SNGS-137, SNGS-138, SNGS-139, SNGS-149, SNGS-192, SNGS-193.

A subset rather than the full split is used purely for compute reasons
(the full 49 sequences would require roughly 53 GPU-hours for both arms).
The subset is random and fixed in advance, so it remains an unbiased sample
of the test distribution.

## 4. What will be run

Both arms, on all 12 sampled sequences, 750 frames each, TVCalib enabled:

* arm A: `configs/gsr_jersey_pipeline_baseline_apply_v1.yaml` (SAR primary)
* arm B: `configs/gsr_jersey_pipeline_ctc_primary_apply_v1.yaml` (CTC primary)

Evaluated with `scripts/evaluate_ft_gsr.py` and aggregated with
`scripts/aggregate_gsr_gs_hota_benchmark.py`.

## 5. What will be reported

For each arm: GS-HOTA (with GS-DetA, GS-AssA, GS-LocA), HOTA, DetA, AssA,
MOTA, jersey coverage and accuracy at tracklet level, and pitch position
error. Macro means over the 12 sequences with 95% sequence-level bootstrap
confidence intervals, plus the paired per-sequence delta between arms.

All measured values will be reported, including any that are unfavourable.
No metric will be selected, dropped or re-weighted after the numbers are seen.

## 6. What will be concluded, under each outcome

**Binding commitments, fixed before the numbers are known.**

* **If test confirms arm B > arm A.** The promotion stands. The test result is
  reported as independent confirmation on a disjoint, previously unused sample.

* **If test shows no significant difference.** The promotion still stands: it
  was decided on validation, where the effect was large and significant on two
  independent surfaces. The absence of a significant difference on a
  12-sequence sample will be reported as such, and attributed explicitly to
  limited statistical power rather than presented as agreement.

* **If test contradicts, i.e. arm A > arm B.** The promotion **still stands**,
  because the decision was made and pre-registered on validation. The
  contradiction will be reported prominently, not minimised, and discussed as
  the most interesting finding of the evaluation: a disagreement between
  disjoint blocks on an effect that was robust elsewhere. No configuration
  will be changed in response, and no additional test evaluation will be run
  to "check" the result.

* **Under every outcome**, the number of test evaluations remains one. This
  document will not be revised after the results are seen.

## 7. Out of scope

Named-player identity assignment (roster filter and Hungarian) is not measured
here: GS-HOTA scores position, team, role and jersey number only. Out-of-domain
broadcast video is not part of this evaluation and its conclusions remain those
already reported on custom footage.
