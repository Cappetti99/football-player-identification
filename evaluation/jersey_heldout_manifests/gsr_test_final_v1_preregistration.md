# Pre-registration: final evaluation on the SoccerNet-GSR test split

Written and frozen **before any evaluation on the sampled test sequences**.
Its purpose is to fix, in advance, what will be run, what will be reported,
and what will be concluded under every possible outcome, so that looking at
the test split is *reporting* and not *selection*.

## 1. Prior disclosure

The GSR test split is not untouched. It was used once already, for the
comparison between two CTC recognizer checkpoints differing only in SJN-210k
pretraining (774 tracklets; accuracy on assigned readings 41.2% vs 73.1%).
That result is reported in this thesis. The present evaluation is therefore a
**second** use of the test split, and is disclosed as such.

## 2. Sample

The full test split contains 49 sequences; evaluating all of them under two
policy arms is not feasible within the available compute budget. A subset was
drawn at random **before any test sequence was evaluated**:

- pool: 49 sequences, sorted by identifier
- procedure: `random.Random(seed).sample(sequences, 12)`
- seed: `20260803`
- size: 12, matching the size of the validation pilot

Selected: SNGS-120, SNGS-125, SNGS-127, SNGS-131, SNGS-133, SNGS-135,
SNGS-137, SNGS-138, SNGS-139, SNGS-149, SNGS-192, SNGS-193.

Recorded in `evaluation/detection_tracking_manifests/test_final_v1.json`,
which stores the seed, the pool size, the procedure and a SHA-256 of each
label file.

## 3. What will be run

Two configurations, differing in exactly one line of configuration --- which
source decides the jersey number --- with identical detection, tracking,
calibration, team and role stages:

- **Arm A**: `sources: [jersey_ocr_primary, jersey_region_ctc]` (SAR primary)
- **Arm B**: `sources: [jersey_region_ctc, jersey_ocr_primary]` (region CTC primary)

TVCalib is enabled for both, as required by GS-HOTA's metric tolerance.
750 frames per sequence, matching the validation protocol.

## 4. What will be reported

For each arm: GS-HOTA (with GS-DetA, GS-AssA, GS-LocA), HOTA, DetA, AssA,
MOTA, jersey coverage and accuracy at tracklet level, macro-averaged over the
12 sequences with 95% sequence-level bootstrap confidence intervals. Both arms
will be reported in full regardless of outcome.

## 5. The decision, fixed in advance

**Arm B is the promoted configuration.** This decision was made on the
validation surfaces and is not revisited here. The supporting evidence,
already obtained:

- pre-registered 7-sequence held-out block, offline surface: 46 tracklets won
  by B against 2 won by A among 48 discordant, exact binomial
  p = 8.4 x 10^-12;
- same block, pipeline surface: +37 recoveries against 3 regressions,
  exact McNemar p = 1.9 x 10^-8;
- 12-sequence validation pilot, GS-HOTA: +0.077, 95% CI [0.048, 0.107].

**Commitment.** The test result will not change which configuration is
promoted. If the test split confirms the validation ordering, it is reported
as an independent confirmation. If it contradicts it, the contradiction is
reported and discussed as a finding --- a disagreement between disjoint
sequence blocks on an effect that is robust elsewhere --- and arm B remains
the promoted configuration, because the promotion was decided on validation
under a pre-registered protocol. Neither outcome will be used to re-select.

## 6. What this evaluation cannot support

The 12 sequences are a random subset, not the whole test split: reported
values estimate performance on the test distribution with the stated
uncertainty, they are not the exact value over all 49 sequences. No
configuration other than the two declared above will be evaluated on these
sequences. No threshold, parameter or architectural choice will be tuned
after seeing these results.
