import pytest

from ft.core.emitters import (
    PRODUCER_JERSEY_PRIMARY,
    PRODUCER_JERSEY_REGION_CTC,
    PRODUCER_JERSEY_SECONDARY,
    jersey_assignment_evidence,
    jersey_region_ctc_evidence,
)
from ft.decision.jersey_policy import (
    DEFAULT_JERSEY_POLICY,
    normalize_policy,
    resolve_jersey_assignments,
)
from ft.pipeline import _apply_jersey


CONFIG_HASH = "cfg0"


def assignment(display_id=12, number=10, segment_index=None):
    return {
        "jersey_number": number,
        "confidence": 0.71,
        "head_confidence": 0.83,
        "winner_margin": 0.32,
        "winner_score_ratio": 5.25,
        "votes": 4,
        "candidates": [{"jersey_number": number, "confidence": 0.71, "votes": 4}],
        "raw_jersey_distribution": {str(number): 0.71},
        "jersey_distribution": {str(number): 0.9},
        "jersey_roster_mass": 0.9,
        "roster_filter": "applied",
        "full_body_sufficient": True,
        "display_track_id": display_id,
        "segment_index": segment_index,
        "segment_start_frame": None,
        "segment_end_frame": None,
    }


def ctc_diagnostics(display_id=12, number=95, confidence=0.42):
    return {
        "enabled": True,
        "standalone_assignments": {
            str(display_id): {
                "display_track_id": display_id,
                "jersey_number": number,
                "confidence": confidence,
                "winner_margin": 0.11,
                "recognized_frames": 3,
                "frames": [10, 40, 90],
                "top5": [number, 55, 5, 9, 59],
                "applied": False,
            }
        },
        "crops": [],
        "configuration": {"ctc_checkpoint_sha256": "11c635"},
    }


# --- the default must be a no-op -------------------------------------------

def test_default_policy_reproduces_the_primary_assignments():
    """Default sources = [primary]: the mapping must match the legacy one."""
    legacy = {12: assignment(12, 10), 7: assignment(7, 23)}
    evidence = jersey_assignment_evidence(legacy, CONFIG_HASH)

    resolved, diagnostics = resolve_jersey_assignments(evidence)

    assert set(resolved) == set(legacy)
    for key, value in legacy.items():
        for field, expected in value.items():
            assert resolved[key][field] == expected
    assert diagnostics["per_source"] == {PRODUCER_JERSEY_PRIMARY: 2}


def test_default_policy_produces_identical_rows_through_apply_jersey():
    legacy = {12: assignment(12, 10)}
    rows_legacy = [{"frame": 0, "track_id": 12, "display_track_id": 12, "track_group": "players"}]
    tracks_legacy = {"players": [{12: {"display_track_id": 12}}]}
    _apply_jersey(legacy, rows_legacy, tracks_legacy)

    resolved, _ = resolve_jersey_assignments(jersey_assignment_evidence(legacy, CONFIG_HASH))
    rows_policy = [{"frame": 0, "track_id": 12, "display_track_id": 12, "track_group": "players"}]
    tracks_policy = {"players": [{12: {"display_track_id": 12}}]}
    _apply_jersey(resolved, rows_policy, tracks_policy)

    assert rows_policy == rows_legacy
    # jersey_evidence carries the whole assignment, including the new
    # decision_source marker, so compare the applied fields instead.
    for field in ("jersey_number", "jersey_confidence", "jersey_votes", "jersey_distribution"):
        assert tracks_policy["players"][0][12][field] == tracks_legacy["players"][0][12][field]


def test_ctc_alone_is_never_selected_by_default():
    evidence = jersey_assignment_evidence({12: assignment(12, 10)}, CONFIG_HASH)
    evidence += jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)

    resolved, diagnostics = resolve_jersey_assignments(evidence)

    assert resolved[12]["jersey_number"] == 10
    assert diagnostics["unreachable_sources"] == []
    assert PRODUCER_JERSEY_REGION_CTC not in diagnostics["per_source"]


# --- promotion is a configuration change ------------------------------------

def test_promoting_ctc_is_a_source_list_change():
    evidence = jersey_assignment_evidence({12: assignment(12, 10)}, CONFIG_HASH)
    evidence += jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)

    resolved, diagnostics = resolve_jersey_assignments(
        evidence, {"sources": [PRODUCER_JERSEY_REGION_CTC, PRODUCER_JERSEY_PRIMARY]}
    )

    assert resolved[12]["jersey_number"] == 95
    assert resolved[12]["decision_source"] == PRODUCER_JERSEY_REGION_CTC
    assert diagnostics["per_source"] == {PRODUCER_JERSEY_REGION_CTC: 1}


def test_ctc_payload_is_mapped_explicitly_not_fabricated():
    evidence = jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)
    resolved, _ = resolve_jersey_assignments(evidence, {"sources": [PRODUCER_JERSEY_REGION_CTC]})

    entry = resolved[12]
    assert entry["votes"] == 3  # recognized_frames
    assert [c["jersey_number"] for c in entry["candidates"]] == [95, 55, 5, 9, 59]
    assert entry["decision_mapping"]
    # Fields with no upstream equivalent must be absent, not invented.
    assert "jersey_roster_mass" not in entry
    assert "head_confidence" not in entry


# --- abstention handling ----------------------------------------------------

def test_fallback_recovers_coverage_when_the_first_source_abstains():
    silent = {**assignment(12), "jersey_number": None}
    evidence = jersey_assignment_evidence({12: silent}, CONFIG_HASH)
    evidence += jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)

    resolved, diagnostics = resolve_jersey_assignments(
        evidence, {"sources": [PRODUCER_JERSEY_PRIMARY, PRODUCER_JERSEY_REGION_CTC]}
    )

    assert resolved[12]["jersey_number"] == 95
    assert diagnostics["decisions"][0]["reason"] == "fallback_after_abstain"


def test_fallback_never_overrides_a_source_that_decided():
    """The only transition against the baseline is unknown -> number."""
    evidence = jersey_assignment_evidence({12: assignment(12, 10)}, CONFIG_HASH)
    evidence += jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)

    resolved, _ = resolve_jersey_assignments(
        evidence, {"sources": [PRODUCER_JERSEY_PRIMARY, PRODUCER_JERSEY_REGION_CTC]}
    )

    assert resolved[12]["jersey_number"] == 10


def test_abstain_mode_disables_the_fallback():
    silent = {**assignment(12), "jersey_number": None}
    evidence = jersey_assignment_evidence({12: silent}, CONFIG_HASH)
    evidence += jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)

    resolved, diagnostics = resolve_jersey_assignments(
        evidence,
        {"sources": [PRODUCER_JERSEY_PRIMARY, PRODUCER_JERSEY_REGION_CTC], "on_abstain": "abstain"},
    )

    assert resolved == {}
    assert diagnostics["decisions"][0]["reason"] == "abstain_no_fallback"


def test_subject_with_no_configured_source_is_skipped():
    evidence = jersey_assignment_evidence(
        {12: assignment(12, 5)}, CONFIG_HASH, produced_by=PRODUCER_JERSEY_SECONDARY
    )
    resolved, diagnostics = resolve_jersey_assignments(evidence)

    assert resolved == {}
    assert diagnostics["unreachable_sources"] == [PRODUCER_JERSEY_PRIMARY]


# --- granularity and hygiene ------------------------------------------------

def test_crop_level_evidence_is_never_promoted():
    diagnostics_payload = ctc_diagnostics(12, 95)
    diagnostics_payload["crops"] = [
        {"display_track_id": 12, "frame": 40, "crop_path": "/c/1.jpg", "ctc_top1": 7,
         "ctc_top1_log_probability": -0.2}
    ]
    evidence = jersey_region_ctc_evidence(diagnostics_payload, CONFIG_HASH)

    resolved, _ = resolve_jersey_assignments(evidence, {"sources": [PRODUCER_JERSEY_REGION_CTC]})

    assert set(resolved) == {12}
    assert resolved[12]["jersey_number"] == 95


def test_segmented_primary_and_unsegmented_ctc_do_not_join():
    """Known limitation, asserted so it cannot regress silently."""
    evidence = jersey_assignment_evidence(
        {(12, 0): assignment(12, 10, segment_index=0)}, CONFIG_HASH
    )
    evidence += jersey_region_ctc_evidence(ctc_diagnostics(12, 95), CONFIG_HASH)

    resolved, diagnostics = resolve_jersey_assignments(
        evidence, {"sources": [PRODUCER_JERSEY_REGION_CTC, PRODUCER_JERSEY_PRIMARY]}
    )

    subjects = {d["subject_id"]: d["chosen_source"] for d in diagnostics["decisions"]}
    assert subjects == {
        "players:12": PRODUCER_JERSEY_REGION_CTC,
        "players:12:0": PRODUCER_JERSEY_PRIMARY,
    }
    assert set(resolved) == {12, (12, 0)}


@pytest.mark.parametrize(
    "policy",
    [{"sources": []}, {"sources": ["a", "a"]}, {"sources": ["a"], "on_abstain": "nonsense"}],
)
def test_invalid_policies_are_rejected(policy):
    with pytest.raises(ValueError):
        normalize_policy(policy)


def test_default_policy_is_single_source():
    assert DEFAULT_JERSEY_POLICY["sources"] == [PRODUCER_JERSEY_PRIMARY]
    assert DEFAULT_JERSEY_POLICY["on_abstain"] == "fallback"


# --- aliasing between assignments and OCR diagnostics -----------------------

def test_diagnostics_keep_raw_jersey_distribution_through_the_policy():
    """JerseyOCR aliases the same dict into assignments and diagnostics.

    The legacy path relied on an in-place backfill to populate
    raw_jersey_distribution in the exported jersey_ocr.json. The policy rebuilds
    fresh dicts, so the backfill must happen before the evidence is emitted --
    otherwise the artifact silently loses the field.
    """
    from ft.pipeline import backfill_raw_jersey_distribution

    voted = assignment(12, 10)
    voted.pop("raw_jersey_distribution")
    assignments = {12: voted}
    diagnostics = {"tracklets": {12: {"voted": voted}}}  # same object, as upstream

    backfill_raw_jersey_distribution(assignments)
    resolved, _ = resolve_jersey_assignments(
        jersey_assignment_evidence(assignments, CONFIG_HASH)
    )

    # The aliased diagnostics saw the backfill...
    assert diagnostics["tracklets"][12]["voted"]["raw_jersey_distribution"] == voted["candidates"]
    # ...and the rebuilt assignment carries it too.
    assert resolved[12]["raw_jersey_distribution"] == voted["candidates"]


def test_backfill_is_idempotent_and_preserves_an_existing_value():
    from ft.pipeline import backfill_raw_jersey_distribution

    existing = [{"jersey_number": 99}]
    assignments = {12: {**assignment(12, 10), "raw_jersey_distribution": existing}}
    backfill_raw_jersey_distribution(assignments)
    backfill_raw_jersey_distribution(assignments)
    assert assignments[12]["raw_jersey_distribution"] == existing
