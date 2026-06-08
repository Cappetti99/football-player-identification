# Experiment Results Summary

This document summarizes the real-video experiments discussed during the OCR and identity-propagation work.
The metrics below come from `scripts/audit_realvideo_runs.py` outputs pasted from the remote machine.

## Key

- `assigned_rows`: player rows with a final `player_id`.
- `candidate_rows`: player rows with a candidate identity but no final assignment.
- `jersey_rows`: player rows with a final visible jersey number.
- `unknown_rows`: player rows without a final jersey number.
- `propagated_rows`: player rows filled by identity propagation.
- `gk_only`: rows rejected by goalkeeper-only roster constraints.

## Inter-Juve

| Run | Frames | Player rows | Display IDs | Assigned rows | Candidate rows | Jersey rows | Unknown rows | Main outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_500f` | 500 | 10,033 | 32 | 3,962 | 6,007 | 3,962 | 6,071 | Strong 500-frame baseline. Correctly recovers several stable numbers, but still misses many visible jerseys. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_morecrops_500f` | 500 | 10,033 | 32 | 4,067 | 5,902 | 4,067 | 5,966 | Small recall gain from more crops; not a structural fix. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_broadcastcontrast_500f` | 500 | 10,033 | 32 | 3,464 | 6,505 | 3,464 | 6,569 | Broadcast contrast hurt recall on the 500-frame sample. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_visualrich_500f` | 500 | 10,033 | 32 | 2,468 | 7,462 | 2,468 | 7,565 | Rich visual features did not help identity assignment in this configuration. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_full` | full | 37,728 | 72 | 6,547 | 31,010 | 7,653 | 30,075 | Main conservative baseline. Duplicate constraints stay clean, but recall is low. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_superres2_full` | full | 37,728 | 72 | 7,695 | 29,680 | 8,801 | 28,927 | Super-resolution increases recall but introduces visible mistakes, especially systematic single-digit noise. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_superres2_merge_partial_apply_full` | full | 37,728 | 72 | 9,241 | 31,010 | 10,347 | 27,381 | Merge raises recall, but visual quality was not as good as rollback auxiliary merge. |
| `Inter-Juve_realvideo_ocr_nogk_nopromote_rawgate020_rollbackaux_merge_partial_apply_full` | full | n/a | n/a | n/a | n/a | n/a | n/a | Visually best Inter-Juve result so far. Uses conservative baseline plus rollback/legacy auxiliary OCR as a filtered source. |

Notes:

- The conservative baseline is stable but leaves too many players unknown.
- The legacy/rollback auxiliary signal recovers useful numbers and works best when merged conservatively.
- Super-resolution can recover extra rows, but it also amplifies false positives.
- Manual identity corrections are intentionally excluded from the target workflow.

## Inter-Atalanta Short Clip

| Run | Frames | Player rows | Display IDs | Assigned rows | Candidate rows | Jersey rows | Unknown rows | Main outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Inter-Atalanta_baseline_rawgate020_500f` | 500 | 3,308 | 19 | 2,311 | 970 | 2,311 | 997 | Good short-clip baseline. Several correct numbers are recovered. |
| `Inter-Atalanta_aux_roi_promote_500f` | 500 | 3,308 | 19 | 1,513 | 1,768 | 2,000 | 1,308 | Auxiliary permissive run loses too many assignments and is not a good replacement baseline. |
| `Inter-Atalanta_baseline_rawgate020_musso_gkfallback_stopowner_500f` | 500 | n/a | n/a | n/a | n/a | n/a | n/a | Added Musso as Atalanta GK #1 and stopped greedy goalkeeper-only fallback when the alternate has a known owner conflict. |

Notes:

- Adding Juan Musso as Atalanta goalkeeper #1 exposed a systematic `#1` OCR issue on a non-goalkeeper tracklet.
- Greedy fallback from goalkeeper-only numbers caused cascades (`#1` -> `#4` -> `#6`), so owner-conflict alternates now stop instead of continuing to weaker candidates.

## Lazio-Juve

| Run | Frames | Player rows | Display IDs | Assigned rows | Candidate rows | Jersey rows | Unknown rows | Main outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Lazio-Juve_baseline_rawgate020_500f` | 500 | 7,441 | 38 | 2,548 | 4,633 | 2,548 | 4,893 | Baseline works partially, but team assignment is weak because kits are visually similar. |
| `Lazio-Juve_aux_roi_promote_500f` | 500 | 7,441 | 38 | 1,395 | 5,870 | 2,400 | 5,041 | Auxiliary run is worse as final output. It finds one novel `#6` candidate but does not fix the broader team/OCR issue. |

Notes:

- This video is mainly a team-assignment stress case: white-black versus white-light-blue kit colors make roster/team filtering less reliable.
- It is not the best target for validating OCR merge until team assignment is improved.

## Int-Ata Long Clip

| Run | Frames | Player rows | Display IDs | Identity tracklets | Assigned rows | Candidate rows | Propagated rows | Jersey rows | Unknown rows | gk_only | Main outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_1200f` | 1,200 | 15,200 | 89 | 75 | 6,017 | 2,188 | 156 | 6,081 | 9,119 | 1,869 | Main long-video baseline. Good stress test for multiple actions and camera changes. |
| `Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_qualityvote_1200f` | 1,200 | 15,200 | 89 | 75 | 5,176 | 3,468 | 0 | 6,172 | 9,028 | 1,586 | Crop-quality weighted voting increases candidates but hurts final identity assignment. Useful only as diagnostic/auxiliary. |
| `Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_1200f` | 1,200 | 15,200 | 89 | 75 | 6,097 | n/a | n/a | 6,161 | n/a | n/a | Propagation recovers some rows but accepted unsafe links without jersey match. |
| `Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_jerseymatch_1200f` | 1,200 | 15,200 | 89 | 75 | 6,613 | 1,996 | 752 | 6,677 | 8,523 | 1,869 | Jersey-match propagation improves metrics but visually links across actions and can involve goalkeeper identities. |
| `Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_1200f` | 1,200 | pending | pending | pending | pending | pending | pending | pending | pending | pending | Next validation run. Disables goalkeeper propagation, disables cut bridge, shortens temporal gap, and raises composite threshold. |

Notes:

- `Int-Ata` is the best current stress test because it includes multiple actions, camera cuts, re-entries, and fragmented tracklets.
- The propagation work exposed a key failure mode: appearance/scene-cut links can connect different actions or transfer a player identity onto a goalkeeper tracklet.
- The strict propagation config is expected to propagate fewer rows; zero or near-zero propagation is acceptable if it prevents unsafe identity transfer.

## Current Technical Conclusions

| Area | What works | Current weakness | Next direction |
|---|---|---|---|
| Detection/tracking | YOLO + ByteTrack gives usable short-term tracklets. | IDs fragment after players leave and re-enter the broadcast view. | Use conservative propagation only when team, jersey, role and temporal/spatial evidence agree. |
| Team assignment | Works well on Inter-Juve and Inter-Atalanta. | Fails on visually similar kits such as Lazio-Juve. | Improve team color modeling before using roster filters aggressively on those videos. |
| OCR baseline | Conservative rawgate/nopromote keeps duplicate constraints clean. | Too many visible numbers remain unknown. | Use auxiliary OCR as candidate evidence, not as unconditional final assignment. |
| Auxiliary OCR | Rollback/legacy auxiliary improves Inter-Juve visually when merged. | Super-resolution and permissive ROI can amplify false positives. | Keep merges filtered by roster uniqueness, display spread, support frames and conflicts. |
| Goalkeeper constraints | Stop-owner fallback prevents greedy cascades. | Many `goalkeeper_only_jersey` rows indicate systematic OCR confusion around `#1`. | Treat goalkeeper-only numbers as hard constraints unless there is a safe alternate without known-owner conflict. |
| Identity propagation | Can recover fragmented identities. | Unsafe when it relies on appearance or scene cuts without enough jersey/role evidence. | Require jersey match, block goalkeeper role mismatch, disable goalkeeper source propagation, and avoid cut bridge by default. |

## Metadata Needed To Rebuild This Table Automatically

To regenerate a complete table locally, copy only metadata from the remote machine. Video files, crops, cache directories and rendered MP4s are not needed.

Recommended remote paths:

```text
/home/cappetti/FT/artifacts/costume-video/*/metadata
```

Minimal files per run:

```text
<video>_tracklets.csv
<video>_constraints.json
<video>_jersey_ocr.json
<video>_identity_propagation.json
<video>_jersey_identity_linking.json
<video>_segment_jersey_candidates.csv
```

Once copied, run:

```bash
python3 scripts/audit_realvideo_runs.py \
  --video-id Int-Ata \
  --run Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_1200f \
  --run Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_qualityvote_1200f \
  --run Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_jerseymatch_1200f
```

