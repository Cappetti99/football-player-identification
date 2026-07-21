from scripts.fuse_jersey_sar_region_ctc import fuse_rows, summarize


def row(track, truth, prediction, confidence=0.0):
    return {
        "sequence": "S",
        "gt_track_id": str(track),
        "eval_track_id": str(track),
        "gt_jersey_number": str(truth),
        "pred_jersey_number": "" if prediction is None else str(prediction),
        "confidence": str(confidence),
        "winner_margin": "0.2",
        "recognized_frames": "4",
    }


def test_conservative_fusion_overrides_only_assigned_baseline():
    baseline = {
        ("S", "1"): row(1, 10, 11),
        ("S", "2"): row(2, 20, 20),
        ("S", "3"): row(3, 30, None),
        ("S", "4"): row(4, 40, 41),
    }
    candidate = {
        ("S", "1"): row(1, 10, 10, 0.95),
        ("S", "2"): row(2, 20, 21, 0.95),
        ("S", "3"): row(3, 30, 30, 0.99),
        ("S", "4"): row(4, 40, 40, 0.89),
    }
    fused, decisions = fuse_rows(baseline, candidate, 0.90)
    assert [row["pred_jersey_number"] for row in fused] == [10, 21, "", 41]
    assert [row["reason"] for row in decisions] == [
        "high_confidence_override",
        "high_confidence_override",
        "baseline_abstention_preserved",
        "candidate_below_threshold",
    ]
    metrics = summarize(fused, decisions)
    assert metrics["overrides"] == 2
    assert metrics["transitions"]["recovered_correct"] == 1
    assert metrics["transitions"]["correct_to_wrong"] == 1
    assert metrics["zero_new_wrong_emissions"] is True


def test_agreement_keeps_baseline_provenance():
    baseline = {("S", "1"): row(1, 10, 10)}
    candidate = {("S", "1"): row(1, 10, 10, 0.1)}
    fused, decisions = fuse_rows(baseline, candidate, 0.90)
    assert fused[0]["fusion_source"] == "agreement"
    assert decisions[0]["transition"] == "unchanged_correct"
