# Archived configs

Configs for experiments the project has explicitly closed with a rejected or
superseded verdict in `handoff.md`. Moved here so the active `configs/`
directory only shows profiles that are either operational or still open for
investigation. Nothing here changes pipeline behavior: these files are inert
unless someone points `--config` at them directly, and `base_config` chains
into/out of this directory were verified to still resolve correctly after the
move (no PyYAML dependency needed for that check — see git history of this
commit for the verification script used).

## tvcalib/

TVCalib pitch-based gating and re-ranking of the linker. Geometrically
accurate (median ~1.10 m, P90 ~2.71 m) but produced no measurable end-to-end
tracking benefit and one regression case; rejected from the operational path.
See handoff.md, "Aggiornamento 2026-07-14 - TVCalib valutata e scartata dal
percorso operativo". `calibration.tvcalib.enabled`, `linking.pitch_gate.enabled`,
and `linking.pitch_reranking.enabled` must stay `false` in any active profile.

## prtreid_superseded/

Early PRTReID-as-visual-backend ablation steps, superseded by the later
"additive linker" architecture (2026-06-23). Kept only as lineage context:

- `default_realvideo_identity_evidence_v1_prtreid.yaml` — first backend
  integration step.
- `..._prtreid_cutsensitive_resetbytetrack_colorteam.yaml` — color-team +
  propagation ablation step.
- `..._prtreid_colorteam_norole_cutsensitive_resetbytetrack.yaml` — role
  classifier disabled ablation step.
- `..._prtreid_linking_cutsensitive_resetbytetrack.yaml` — the flawed first
  linking candidate: it inherited `appearance_min_similarity: 0.55` from the
  old backend config, which also altered the legacy linker instead of staying
  additive. The corrected version
  (`configs/default_realvideo_identity_evidence_v1_prtreid_linking_additive_cutsensitive_resetbytetrack.yaml`,
  still active) inherits directly from the clean baseline instead.

The validated, still-active PRTReID profiles
(`prtreid_linking_conservative_int_ata_v1.yaml`,
`prtreid_linking_calibrated_int_ata_audit.yaml`,
`prtreid_identity_bridge_audit_int_ata.yaml`,
`prtreid_linking_additive_control_disabled.yaml`,
`gsr_prtreid_detection_audit_v1.yaml`) remain in `configs/`, not here.

## rejected/

- `number_region_cost_int_ata.yaml` — global Hungarian cost integration of
  number-region evidence. Explicitly scrapped ("Run costo globale -
  scartata"): a good local signal in one part of the match could steal or
  block identities in unrelated actions once all actions compete together in
  one global assignment. Superseded by `assignment_scope: scene_segment`
  (see `configs/number_region_cost_scene_reset_int_ata.yaml`, still active,
  which inherits from this archived file only for its base OCR/team/roster
  settings, not for the rejected global-cost behavior — the scene-reset
  config overrides `number_region_cost_enabled`/weights itself).
- `identity_constraints_a2_jersey_linker_int_ata.yaml` — conservative jersey
  identity linker gate (A2). Measured as a pure no-op on the frozen
  validation (0 accepted links, metrics identical to the A1 control); not
  promoted, not reopened without a new hypothesis.
