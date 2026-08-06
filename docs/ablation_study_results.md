# Oracle Ablation Study: Where the Error Lives

Reference document for the results chapter. Every number here was produced by
a real run on the remote machine and is reproducible with the commands given;
none is estimated. Companion to `docs/additional_thesis_metrics.md`, which
covers the metric definitions (MOTA, HOTA, GS-HOTA, pose failure/missing).

## 1. Question and method

Which architecture stage is responsible for the end-to-end error, and by how
much? Following the ablation methodology of Somers et al. (CVPR24, Table 2),
each module is replaced in turn by a ground-truth oracle and GS-HOTA is
recomputed. Two complementary directions are reported per module:

* **marginal gain** (`oracle_<module>`) -- baseline plus that one module made
  perfect. What would we gain by fixing this alone?
* **isolated cost** (`oracle_all_but_<module>`) -- every other attribute made
  perfect, this one left real. What does it still cost when nothing else is
  broken? Equal to `oracle_all_attributes - oracle_all_but_<module>`.

Modules and the GS-HOTA component each controls:

| module | controls | GS-HOTA component |
|---|---|---|
| calibration | pitch position | LocSim (Gaussian, tau = 5 m) |
| team / role / jersey | identity attributes | IdSim (all-or-nothing) |
| detection | which subjects exist | DetA (FP/FN) |
| tracking | how they link over time | AssA |

**Surface.** SoccerNet-GSR `valid`, the 12-sequence detection/tracking pilot
(`evaluation/detection_tracking_manifests/valid_pilot12_v1.json`), 750 frames
each. The GSR `test` split is deliberately **not** used: an ablation exists to
decide where to invest effort, which is exactly the kind of decision that
consumes a held-out surface. `aggregate_gs_hota_ablation.py` refuses the test
split without an explicit `--allow-test`.

**Cost.** No inference, no GPU: the ablation is pure re-analysis of the
`tracklets.csv` artifacts already produced by the GS-HOTA benchmark runs.

### Reproduce

```bash
# single sequence
PYTHONPATH=/home/cappetti/FT python3 scripts/ablate_gs_hota_oracles.py \
  --labels /media/data-lie/cappetti/dataset/SoccerNet-GSR/valid/SNGS-082/Labels-GameState.json \
  --tracklets artifacts/gs_hota_check/SNGS-082_arm_b_tvcalib/metadata/SNGS-082_tracklets.csv \
  --output-dir evaluation_outputs/gs_hota_ablation/SNGS-082_arm_b

# all 12 sequences, both policy arms, with bootstrap CIs and LaTeX tables
PYTHONPATH=/home/cappetti/FT python3 scripts/aggregate_gs_hota_ablation.py \
  --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \
  --artifacts-root artifacts/gs_hota_benchmark \
  --arms a b \
  --output-dir evaluation_outputs/gs_hota_ablation/pilot12
```

Artifacts: `evaluation_outputs/gs_hota_ablation/pilot12/{aggregate.json,
per_sequence.csv, attribution.csv, tables.tex}`. `tables.tex` holds two
booktabs tables ready to `\input`. Tests: `tests/test_gs_hota_ablation.py`
(20 tests).

## 2. Two methodological defects found and fixed

Both were caught by the harness's own checks, not by inspection. Worth keeping
in the thesis: they are evidence the measurement instrument was validated
before its output was trusted.

### 2.1 Attribute rows were measured on the wrong surface

The first version measured `oracle_all_but_<module>` with detection **also**
under oracle. That injects ground-truth boxes the real system never detected,
carrying no attributes, so one false negative becomes a false-negative /
false-positive pair and the affected players are penalised twice.

Symptom: `oracle_detection_tracking` scored **below** baseline (-0.0104 on
SNGS-082), i.e. perfect detection appeared to make the system worse, which is
impossible. Contamination reached every attribute row: team and role came out
with *negative* isolated costs, which is meaningless.

Fix: attribute rows are computed on the **real** detection/tracking surface,
making them directly comparable to `oracle_all_attributes`. Pinned by
`test_attribute_rows_stay_on_the_real_detection_surface`.

The `oracle_detection_tracking` row itself remains a non-clean counterfactual
-- producing real attributes for boxes that were never detected would require
running the recognizers on crops that were never extracted, which no offline
substitution can do. It is excluded from the reported tables and the
detection/tracking cost is read from `oracle_all_attributes` instead.

### 2.2 Annotations without pitch coordinates depressed the sanity check

`full_oracle` (every module oracle) must score exactly 1.0: predictions are
the ground truth. It did not, on 4 of 12 sequences.

The cause is in the dataset, not the pipeline: some annotations carry no
`bbox_pitch`, so they have no position. GS-HOTA's similarity is defined on
pitch coordinates, so such a subject can never match anything -- not even a
perfect oracle -- and costs both a false negative and a false positive.

The mechanism was confirmed quantitatively. With `n` annotations of which `m`
lack a position, DetA = (n-m)/(n+m) ~= 1 - 2m/n, i.e. the deficit is exactly
twice the missing fraction:

| sequence | missing / total | observed deficit | 2 x missing fraction |
|---|---|---|---|
| SNGS-025 | 55 / 12744 (0.431%) | 0.856% | 0.863% |
| SNGS-055 | 8 / 11412 (0.070%) | 0.138% | 0.140% |
| SNGS-088 | 4 / 13135 (0.030%) | 0.061% | 0.061% |
| SNGS-051 | 1 / 11743 (0.009%) | 0.017% | 0.017% |
| SNGS-036, SNGS-082 | 0 | 0 | 0 |

An earlier hypothesis -- that subjects sharing the same (team, role, jersey)
tuple are intrinsically unidentifiable, as the paper itself notes in Sec. 4.2
-- was **tested and rejected**: zero such pairs within tau on any sequence,
including the failing ones.

Fix: `drop_unscoreable_subjects` removes those annotations *and* whatever
detection matched them, since dropping only the annotation would turn a
correct detection into a false positive and punish the system for a dataset
gap. Dropped counts are reported per sequence. `full_oracle` is now exactly
1.0000 on all 24 arm/sequence combinations.

Residual inconsistency, deliberate and negligible: the ablation baseline
excludes these annotations while the headline GS-HOTA from
`scripts/evaluate_ft_gsr.py` keeps them. Measured difference: **0.0002**
(0.2099 vs 0.2097 for arm A; 0.2871 vs 0.2870 for arm B), against an
arm-to-arm effect of 0.0772.

## 3. Main result: GS-HOTA under oracle substitution

12 sequences, macro mean, 95% sequence-level bootstrap CI. Arm A = SAR
primary (conservative), arm B = region CTC primary.

| configuration | arm A | arm B |
|---|---|---|
| baseline (no oracle) | 0.2099 [0.136, 0.295] | 0.2871 [0.194, 0.390] |
| + oracle calibration | 0.2832 [0.177, 0.396] | 0.3916 [0.264, 0.529] |
| + oracle team | 0.2371 [0.160, 0.318] | 0.3099 [0.214, 0.411] |
| + oracle role | 0.2370 [0.172, 0.312] | 0.3087 [0.222, 0.406] |
| + oracle jersey | 0.3819 [0.295, 0.475] | 0.3846 [0.298, 0.477] |
| all attributes but calibration | 0.5350 [0.480, 0.587] | 0.5374 [0.481, 0.590] |
| all attributes but team | 0.5593 [0.453, 0.670] | 0.5624 [0.456, 0.673] |
| all attributes but role | 0.5793 [0.461, 0.695] | 0.5824 [0.464, 0.698] |
| all attributes but jersey | 0.3638 [0.278, 0.459] | 0.4620 [0.353, 0.586] |
| all attributes oracle | 0.7404 [0.692, 0.794] | 0.7434 [0.695, 0.798] |
| full oracle (sanity) | 1.0000 | 1.0000 |

## 4. The arm difference is the jersey, and nothing else

The decisive pattern is not the absolute values but the **gap between arms**
in each configuration:

| configuration | jersey source | gap (B - A) |
|---|---|---|
| + oracle calibration | real | **+0.1084** |
| all attributes but jersey | real | **+0.0982** |
| baseline | real | **+0.0772** |
| + oracle team | real | **+0.0728** |
| + oracle role | real | **+0.0717** |
| + oracle jersey | oracle | +0.0027 |
| all attributes but calibration | oracle | +0.0024 |
| all attributes but team | oracle | +0.0031 |
| all attributes but role | oracle | +0.0031 |
| all attributes oracle | oracle | +0.0030 |
| full oracle | oracle | 0.0000 |

The gap is 0.072-0.108 wherever the jersey is real and 0.002-0.003 wherever it
is oracle: a factor of about 30. Substituting the true jersey number makes the
two arms indistinguishable, because at that point they *are* the same system
-- same detector, same tracker, same TVCalib, same team and role stages. They
differ in one configuration line: which source decides the number.

This doubles as an internal validation of the harness. The isolated costs of
the modules the arms share come out identical to within 0.3% (detection/
tracking 0.2596 vs 0.2566, calibration 0.2054 vs 0.2060, team 0.1812 vs
0.1809, role 0.1612 vs 0.1610). Had the ablation attributed a real difference
to, say, calibration between two arms running the same TVCalib, that would
have signalled a defect.

The residual ~0.003 is not exactly zero because the oracle is applied to pairs
matched by **IoU**, while GS-HOTA matches on **position and attributes**:
predictions that failed IoU matching keep their real jersey (which differs
between arms) and can still match in GS-HOTA space.

## 5. Module attribution, and a correction

Isolated cost in GS-HOTA points, arm B, 12 sequences:

| module | isolated cost | 95% CI |
|---|---|---|
| jersey | 0.2814 | [0.203, 0.358] |
| detection / tracking | 0.2566 | [0.205, 0.309] |
| calibration | 0.2060 | [0.178, 0.231] |
| team | 0.1809 | [0.103, 0.262] |
| role | 0.1610 | [0.082, 0.244] |

**All five modules fall between 0.16 and 0.28. There is no single dominant
bottleneck.** The jersey is the most expensive, but only marginally, and its
CI overlaps every other module's.

This corrects a conclusion drawn earlier from SNGS-082 alone, where team cost
0.0291 and role 0.0070 and both looked negligible. Across 12 sequences they
cost 0.181 and 0.161 -- six and twenty-three times more. SNGS-082 happened to
sit at the favourable extreme (96.2% team accuracy, 98.7% role accuracy), and
the wide CIs on team and role confirm the high sequence-to-sequence variance
that made one sequence misleading. This is the same lesson the paper's own
Fig. 5 teaches (GS-HOTA ranging from 0.23% to 49.69% across videos) and that
we hit independently when extending GS-HOTA from one sequence to twelve.

Costs are **not additive**: they sum to 1.086 against a total gap from 1.0 of
0.713, because GS-HOTA multiplies LocSim by IdSim and two modules failing on
the same subject overlap rather than sum.

## 6. Why arm B is more accurate, mechanistically

Three levels, each measured on a different and explicitly named block.

### 6.1 The gain splits into coverage and correction

Pipeline surface, 7 held-out sequences, 114 tracklets. The +34 correct
tracklets decompose as:

| mechanism | tracklets | what happens |
|---|---|---|
| coverage | +20 | SAR abstained, CTC answered correctly |
| correction | +17 | SAR answered **wrong**, CTC overrode it correctly |
| regression | -3 | SAR was right, CTC broke it |

(the 20 is derived: 37 documented recoveries minus 17 corrections)

An arithmetic check makes the correction component undeniable: emitted grows
63 -> 94 (**+31**) while correct grows 19 -> 53 (**+34**). If the CTC were
merely more talkative, correct could not grow faster than emitted. It does.

On the **offline** surface (same 7 sequences, ground-truth boxes and tracks)
the proportion inverts and the point sharpens: SAR 18 -> arm A 28 -> arm B 72.
The first +10 is pure fallback on abstentions; the following **+44 is entirely
correction** (46 tracklets won by the CTC against 2 won by the SAR, out of 48
discordant). Both arms assign the *same 95 tracklets*: coverage does not
change at all, only who answers.

### 6.2 The CTC's advantage is reading quality, not coverage

Measured in isolation on the GSR test block (774 tracklets, already spent),
comparing two CTC checkpoints differing **only** in SJN-210k pretraining:

| | GSR only | SJN -> GSR |
|---|---|---|
| region coverage | 61.0% | **61.0%** |
| tracklet coverage | 78.2% | **78.2%** |
| correct | 249 | 442 |
| wrong | 356 | 163 |
| accuracy on assigned | 41.2% | **73.1%** |

Coverage is identical to the decimal: the same crops, the same regions. Yet
correct nearly doubles and wrong drops by more than half. A pure reading-
accuracy gain at constant input -- which is what the CTC brings when it
overrides the SAR.

### 6.3 GS-HOTA amplifies jersey errors, because IdSim is multiplicative

`Sim = LocSim x IdSim`, and IdSim is 0 unless team, role **and** jersey all
match. A wrong number therefore does not merely lose the number: it makes the
subject unmatchable, costing a false negative and a false positive together.

The ablation shows the size of this effect directly: GS-DetA rises from
**0.2405** at baseline to **0.9393** with attributes under oracle (SNGS-082).
Roughly 70% of the apparent "missed detection" is not detection failing at all
-- it is correctly detected, correctly tracked players rendered unmatchable by
one wrong attribute.

The same signature appears in the arm comparison over 12 sequences:

| component | arm A | arm B | relative change |
|---|---|---|---|
| GS-DetA | 0.1168 | 0.1997 | **+71%** |
| GS-AssA | 0.4184 | 0.4542 | +8.6% |
| GS-LocA | 0.8258 | 0.8350 | +1.1% |

The gain is concentrated almost entirely in **DetA**, not association. Arm B
does not track better and does not calibrate better; it makes *identifiable*
subjects that were previously discarded over a wrong number.

**Caveat on GS-LocA.** The +1.1% is not a calibration improvement: both arms
run bit-identical TVCalib. LocA is averaged over matched pairs only, so more
matched subjects changes the population it is averaged over. A compositional
effect, not a quality one -- worth stating explicitly, since it otherwise
reads as a gain that does not exist.

## 7. Limitations

* **Evaluation-surface substitution, not a pipeline re-run.** Oracles are
  applied after inference, so downstream interactions are not modelled: a
  perfect team label here does not also change what the roster-aware filter
  would have done to the jersey during the run. The tables answer "what does
  this module's error cost at the output", not "what would the system do if
  this module were perfect". The latter requires injecting ground truth into
  the pipeline itself.
* **GS-HOTA does not score named identity.** Position, team, role and jersey
  only. The roster filter and Hungarian assignment to named players are
  downstream of everything measured here and are covered by
  `ft/evaluation/identity_benchmark.py` instead.
* **`oracle_detection_tracking` is not a clean counterfactual** (Sec. 2.1) and
  is excluded from the reported tables.
* **Valid split, not test.** These numbers describe the validation pilot. Any
  final claim on unseen data still requires the frozen test split, spent once.

## 8. Open items

* Decompose detection/tracking further. At `oracle_all_attributes`, GS-DetA is
  0.9393 while GS-AssA is 0.5683 (SNGS-082): with perfect attributes detection
  is nearly solved and **association is the binding constraint**. This aligns
  with the tracking-bottleneck result already in the thesis (AssA 0.517,
  19-point DetA-to-HOTA gap) but has not yet been separated into its own
  ablation row.
* Optionally apply `drop_unscoreable_subjects` inside
  `scripts/evaluate_ft_gsr.py` as well, so the headline GS-HOTA and the
  ablation baseline agree exactly. Costs one re-analysis, no inference; the
  current discrepancy is 0.0002.
