import csv
import json
from pathlib import Path

import pytest

from scripts.build_yolo_jersey_number_readability_dataset import (
    load_reviews,
    require_both_classes,
)
from scripts.evaluate_yolo_jersey_number_readability import evaluate


def write_reviews(path, rows):
    fields = [
        "audit_id", "review_label", "sequence", "gt_track_id", "frame", "crop_path"
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_reviews_map_manual_readability_and_ignore_uncertain(tmp_path):
    path = tmp_path / "review.csv"
    write_reviews(path, [
        {"audit_id": "a", "review_label": "readable", "sequence": "A", "gt_track_id": "1", "frame": 1, "crop_path": "a.jpg"},
        {"audit_id": "b", "review_label": "unreadable", "sequence": "A", "gt_track_id": "1", "frame": 2, "crop_path": "b.jpg"},
        {"audit_id": "c", "review_label": "uncertain", "sequence": "A", "gt_track_id": "1", "frame": 3, "crop_path": "c.jpg"},
    ])

    rows, ignored = load_reviews([path])

    assert [row["class_name"] for row in rows] == ["number_readable", "number_unreadable"]
    assert ignored == {"uncertain": 1}


def test_split_requires_both_classes():
    with pytest.raises(ValueError):
        require_both_classes([{"class_name": "number_readable"}], "validation")


def test_ranking_metrics_use_temporally_distinct_top_k():
    rows = [
        {"sequence": "A", "gt_track_id": "1", "frame": 1, "crop_path": "a", "is_readable": False, "readability_score": 0.9},
        {"sequence": "A", "gt_track_id": "1", "frame": 2, "crop_path": "b", "is_readable": True, "readability_score": 0.8},
        {"sequence": "A", "gt_track_id": "1", "frame": 10, "crop_path": "c", "is_readable": True, "readability_score": 0.7},
    ]

    metrics = evaluate(rows, top_k=2, min_frame_gap=5)

    assert metrics["eligible_tracklets"] == 1
    assert metrics["topk_hit_rate"] == 1.0
