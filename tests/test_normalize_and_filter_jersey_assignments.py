"""Contract test for ft.pipeline.normalize_and_filter_jersey_assignments.

Extracted 2026-07-21 from two near-identical blocks in _run_pipeline_impl
(one for the primary jersey OCR pass, one for the segment-candidate pass).
This locks in the shared behavior: raw_jersey_distribution backfill, roster
filtering, and goalkeeper filtering, for both call sites.
"""
from ft.pipeline import normalize_and_filter_jersey_assignments


BASE_CONFIG = {"jersey_ocr": {"roster_aware": True, "apply_to_goalkeepers": False}}


def test_backfills_raw_jersey_distribution_from_candidates():
    assignments = {
        1: {"jersey_number": 7, "candidates": {"7": 0.9, "17": 0.1}, "raw_jersey_distribution": None},
    }
    result, diagnostics = normalize_and_filter_jersey_assignments(
        assignments, {}, player_rows=[], roster=[], config=BASE_CONFIG
    )
    assert result[1]["raw_jersey_distribution"] == {"7": 0.9, "17": 0.1}
    assert "roster_filter" in diagnostics
    assert "goalkeeper_ocr_filter" in diagnostics


def test_does_not_overwrite_existing_raw_jersey_distribution():
    assignments = {
        1: {"jersey_number": 7, "candidates": {"7": 0.9}, "raw_jersey_distribution": {"7": 1.0}},
    }
    result, _ = normalize_and_filter_jersey_assignments(
        assignments, {}, player_rows=[], roster=[], config=BASE_CONFIG
    )
    assert result[1]["raw_jersey_distribution"] == {"7": 1.0}


def test_skips_roster_filter_when_roster_aware_disabled():
    config = {"jersey_ocr": {"roster_aware": False, "apply_to_goalkeepers": False}}
    assignments = {1: {"jersey_number": 7, "candidates": {"7": 0.9}}}
    result, diagnostics = normalize_and_filter_jersey_assignments(
        assignments, {}, player_rows=[], roster=[], config=config
    )
    assert "roster_filter" not in diagnostics
    assert result[1]["jersey_number"] == 7


def test_drops_goalkeeper_assignment_when_not_applied_to_goalkeepers():
    assignments = {1: {"jersey_number": 1, "candidates": {"1": 0.9}, "display_track_id": 1}}
    player_rows = [
        {"track_id": 1, "display_track_id": 1, "track_group": "players", "role_detection": "goalkeeper"},
    ]
    result, diagnostics = normalize_and_filter_jersey_assignments(
        assignments, {}, player_rows=player_rows, roster=[], config=BASE_CONFIG
    )
    assert result == {}
    assert diagnostics["goalkeeper_ocr_filter"]["dropped"]
