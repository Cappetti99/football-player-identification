from scripts.evaluate_jersey_region_ctc_preprocessing_ablation import (
    declared_sequences,
    grouped_metrics,
    summarize,
    validate_observed_sequences,
)


def test_summarize_reports_digit_groups_and_confident_errors():
    tracks = [
        {
            "assigned": True, "correct": True, "confidence": 0.99,
            "gt_in_top5": True, "digits": 1, "sequence": "A", "prediction": 7,
        },
        {
            "assigned": True, "correct": False, "confidence": 0.95,
            "gt_in_top5": False, "digits": 2, "sequence": "B", "prediction": 6,
        },
    ]
    metrics = summarize(
        [{"correct": True}, {"correct": False}], tracks, 4, 2
    )
    assert metrics["accuracy_all"] == 0.5
    assert metrics["high_confidence_wrong"] == 1
    assert metrics["by_digits"]["1"]["accuracy_all"] == 1.0
    assert metrics["by_digits"]["2"]["accuracy_all"] == 0.0


def test_grouped_metrics_keeps_unassigned_tracks_in_denominator():
    rows = [
        {"sequence": "A", "assigned": False, "correct": False, "gt_in_top5": False},
        {"sequence": "A", "assigned": True, "correct": True, "gt_in_top5": True},
    ]
    assert grouped_metrics(rows, "sequence")["A"] == {
        "tracklets": 2,
        "assigned": 1,
        "correct": 1,
        "coverage": 0.5,
        "accuracy_all": 0.5,
        "gt_in_top5_rate": 0.5,
    }


def test_frozen_sequence_guard_rejects_unexpected_ocr_sequence():
    manifest = {"frozen_validation_sequences": ["A", "B"]}
    allowed = declared_sequences(manifest, "frozen")
    assert allowed == {"A", "B"}
    try:
        validate_observed_sequences({("C", "1"): {}}, allowed)
    except ValueError as exc:
        assert "outside requested manifest part" in str(exc)
    else:
        raise AssertionError("unexpected sequence was accepted")
