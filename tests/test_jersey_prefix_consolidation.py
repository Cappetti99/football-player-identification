import pytest

from ft.features.jersey_ocr import vote_numbers
from ft.features.jersey_prefix_consolidation import (
    JerseyPrefixConsolidator,
    supported_long_numbers,
)
from ft.validation import _validate_jersey_prefix_consolidation


def detections(short, long):
    return [
        {"number": short, "confidence": 0.9, "crop_path": f"s{i}.jpg", "frame": i}
        for i in range(3)
    ] + [
        {"number": long, "confidence": 1.0, "crop_path": f"l{i}.jpg", "frame": 10 + i}
        for i in range(2)
    ]


@pytest.mark.parametrize("short,long", [(1, 15), (5, 15), (2, 23), (3, 34)])
def test_supported_short_long_conflicts_reduce_only_short_weight(short, long):
    rows = detections(short, long)
    baseline = vote_numbers(rows)
    consolidator = JerseyPrefixConsolidator(
        mode="propose", selected_short_vote_weight=0.5
    )
    proposed = consolidator.consolidate(4, rows, vote_numbers)
    assert baseline["jersey_number"] == short
    assert proposed["jersey_number"] == long
    assert proposed["votes"] == 2
    assert {item["jersey_number"] for item in proposed["candidates"]} == {short, long}


def test_audit_never_changes_baseline_and_records_frozen_grid():
    rows = detections(1, 15)
    consolidator = JerseyPrefixConsolidator(mode="audit")
    result = consolidator.consolidate(9, rows, vote_numbers, segment_index=2)
    assert result == vote_numbers(rows)
    assert [row["short_vote_weight"] for row in consolidator.counterfactual_rows] == [1.0, .75, .5, .25]
    assert {row["segment_index"] for row in consolidator.counterfactual_rows} == {2}


def test_no_change_without_two_independent_long_crops():
    rows = detections(1, 15)[:-1]
    consolidator = JerseyPrefixConsolidator(mode="propose", selected_short_vote_weight=.25)
    assert consolidator.consolidate(1, rows, vote_numbers) == vote_numbers(rows)
    duplicate_passes = [
        {"number": 15, "confidence": 1.0, "crop_path": "same.jpg", "frame": 1, "pass": 0},
        {"number": 15, "confidence": 1.0, "crop_path": "same.jpg", "frame": 1, "pass": 1},
    ]
    assert supported_long_numbers(duplicate_passes, 2) == set()


def test_counterfactual_acceptance_respects_final_minimum_votes():
    rows = [{"number": 8, "confidence": 1.0, "crop_path": "one.jpg", "frame": 1}]
    consolidator = JerseyPrefixConsolidator(mode="audit")
    consolidator.consolidate(1, rows, vote_numbers, min_votes=2)
    assert all(not row["accepted"] for row in consolidator.counterfactual_rows)
    assert {row["rejection_reason"] for row in consolidator.counterfactual_rows} == {"insufficient_votes"}


def test_supported_long_tie_and_insufficient_margin_abstain():
    rows = detections(1, 15) + [
        {"number": 17, "confidence": 1.0, "crop_path": f"x{i}.jpg", "frame": 20 + i}
        for i in range(2)
    ]
    tie = JerseyPrefixConsolidator(mode="propose", selected_short_vote_weight=.25)
    assert tie.consolidate(2, rows, vote_numbers) is None

    margin = JerseyPrefixConsolidator(mode="propose", selected_short_vote_weight=.5)
    assert margin.consolidate(3, detections(1, 15), vote_numbers, min_margin=.2) is None


def test_unrelated_candidate_exposed_by_downweighting_causes_abstention():
    rows = [
        {"number": 1, "confidence": 1.0, "crop_path": f"s{i}.jpg", "frame": i}
        for i in range(3)
    ] + [
        {"number": 12, "confidence": .4, "crop_path": f"l{i}.jpg", "frame": 10 + i}
        for i in range(2)
    ] + [
        {"number": 5, "confidence": .5, "crop_path": f"u{i}.jpg", "frame": 20 + i}
        for i in range(2)
    ]
    consolidator = JerseyPrefixConsolidator(
        mode="propose", selected_short_vote_weight=.25
    )
    assert consolidator.consolidate(12, rows, vote_numbers) is None
    selected = [
        row for row in consolidator.counterfactual_rows
        if row["short_vote_weight"] == .25
    ][0]
    assert selected["counterfactual_winner"] == 5
    assert selected["rejection_reason"] == "winner_not_prefix_compatible"


def test_propose_requires_frozen_selected_weight():
    with pytest.raises(ValueError, match="required"):
        JerseyPrefixConsolidator(mode="propose")


def test_prefix_validation_rejects_apply_mode():
    errors = []
    _validate_jersey_prefix_consolidation({
        "enabled": True, "mode": "apply", "min_long_votes": 2,
        "short_vote_weights": [1.0, .5], "selected_short_vote_weight": .5,
    }, errors)
    assert any("unsupported" in error for error in errors)
