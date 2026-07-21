import csv
import json
from pathlib import Path

from scripts.build_gsr_ocr_usable_dataset import build_sequence, normalize_split


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def test_builder_labels_exact_match_and_partial_negative(tmp_path):
    scores = tmp_path / "scores.csv"
    matches = tmp_path / "matches.csv"
    ocr = tmp_path / "ocr.json"
    write_csv(scores, [
        {"display_track_id": 7, "frame": 10, "crop_path": "a.jpg", "legibility_score": .9, "crop_quality": .5, "pred_role": "player"},
        {"display_track_id": 7, "frame": 20, "crop_path": "b.jpg", "legibility_score": .8, "crop_quality": .4, "pred_role": "player"},
    ])
    write_csv(matches, [
        {"pred_track_id": 7, "frame": 10, "gt_track_id": 1, "gt_jersey": 17, "iou": .9},
        {"pred_track_id": 7, "frame": 20, "gt_track_id": 1, "gt_jersey": 17, "iou": .9},
    ])
    ocr.write_text(json.dumps({"tracklets": {"7": {
        "display_track_id": 7,
        "selected_crops": [
            {"frame": 10, "crop_path": "a.jpg"},
            {"frame": 20, "crop_path": "b.jpg"},
        ],
        "aggregated_detections": [
            {"frame": 10, "crop_path": "a.jpg", "number": 17},
            {"frame": 20, "crop_path": "b.jpg", "number": 1},
        ],
    }}}))
    rows = build_sequence(
        {"scores": str(scores), "matches": str(matches), "ocr": str(ocr)},
        Path("."), "SNGS-X", "train",
    )
    assert [row["ocr_usable"] for row in rows] == [True, False]
    assert rows[1]["single_digit_prefix_error"] is True


def test_split_normalization():
    assert normalize_split("valid") == "validation"
    assert normalize_split("test") == "test"
