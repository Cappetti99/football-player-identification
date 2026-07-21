import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_gsr_jersey_crop_pseudolabels import build_pseudolabels, source_winners
from build_jersey_crop_pseudolabel_audit import sequence_round_robin


def test_pseudolabels_require_independent_source_agreement():
    diagnostics = {
        "tracklets": {
            "1": {
                "display_track_id": 1,
                "selected_crops": [
                    {"crop_path": "positive.jpg", "frame": 1, "crop_quality": 0.8},
                    {"crop_path": "negative.jpg", "frame": 2, "crop_quality": 0.7},
                    {"crop_path": "disagree.jpg", "frame": 3, "crop_quality": 0.6},
                    {"crop_path": "empty.jpg", "frame": 4, "crop_quality": 0.5},
                ],
                "detections": [
                    detection("positive.jpg", "mmocr", 8, 0.8),
                    detection("positive.jpg", "easyocr", 8, 0.7),
                    detection("negative.jpg", "mmocr", 3, 0.9),
                    detection("negative.jpg", "easyocr", 3, 0.6),
                    detection("disagree.jpg", "mmocr", 8, 0.9),
                    detection("disagree.jpg", "easyocr", 3, 0.9),
                ],
            }
        }
    }
    predictions = {"1": {"sequence": "S", "gt_track_id": "T", "gt_jersey": 8}}
    rows = build_pseudolabels(diagnostics, predictions, 0.2, 2)
    labels = {Path(row["crop_path"]).name: row["pseudo_label"] for row in rows}
    assert labels == {
        "positive.jpg": "positive",
        "negative.jpg": "hard_negative",
        "disagree.jpg": "ignore",
        "empty.jpg": "ignore",
    }


def test_source_variants_do_not_create_independent_votes():
    rows = source_winners([
        detection("crop.jpg", "mmocr", 8, 0.5),
        detection("crop.jpg", "mmocr", 8, 0.9),
        detection("crop.jpg", "mmocr", 3, 0.8),
    ])
    assert rows == [{"source": "mmocr", "number": 8, "confidence": 0.9, "variants": 2}]


def test_single_source_can_be_compared_directly_with_gt():
    diagnostics = {
        "tracklets": {
            "1": {
                "display_track_id": 1,
                "selected_crops": [
                    {"crop_path": "positive.jpg", "frame": 1},
                    {"crop_path": "negative.jpg", "frame": 2},
                    {"crop_path": "empty.jpg", "frame": 3},
                ],
                "detections": [
                    detection("positive.jpg", "easyocr", 8, 0.8),
                    detection("negative.jpg", "easyocr", 3, 0.9),
                ],
            }
        }
    }
    predictions = {"1": {"sequence": "S", "gt_track_id": "T", "gt_jersey": 8}}
    rows = build_pseudolabels(diagnostics, predictions, 0.2, 1)
    by_name = {Path(row["crop_path"]).name: row for row in rows}
    assert by_name["positive.jpg"]["pseudo_label"] == "positive"
    assert by_name["positive.jpg"]["pseudo_reason"] == "single_source_matches_gt"
    assert by_name["negative.jpg"]["pseudo_label"] == "hard_negative"
    assert by_name["negative.jpg"]["pseudo_reason"] == "single_source_disagrees_with_gt"
    assert by_name["empty.jpg"]["pseudo_label"] == "ignore"


def test_audit_sampling_round_robins_sequences():
    rows = [
        {"sequence": "A", "value": index} for index in range(10)
    ] + [{"sequence": "B", "value": 1}]
    selected = sequence_round_robin(rows, limit=2, seed=7)
    assert {row["sequence"] for row in selected} == {"A", "B"}


def detection(path, source, number, confidence):
    return {
        "crop_path": path,
        "source": source,
        "number": number,
        "confidence": confidence,
    }
