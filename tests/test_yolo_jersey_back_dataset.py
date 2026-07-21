import csv
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_yolo_jersey_back_dataset import load_reviews, split_review_rows


def test_review_mapping_and_ignored_labels(tmp_path):
    review = tmp_path / "review.csv"
    write_review(review, [
        row("A", "1", 1, "a.jpg", "clean_back"),
        row("A", "1", 2, "b.jpg", "usable_not_back"),
        row("A", "1", 3, "c.jpg", "not_clean"),
        row("A", "1", 4, "d.jpg", "uncertain"),
    ])
    rows, ignored = load_reviews([review])
    assert [item["class_name"] for item in rows] == ["clean_back", "clean_back", "not_clean"]
    assert ignored["uncertain"] == 1


def test_dataset_split_is_sequence_disjoint_and_rejects_frozen():
    rows = []
    for sequence in ("A", "B", "C", "D"):
        rows.extend([
            {**row(sequence, "1", 1, f"{sequence}a.jpg", "clean_back"), "class_name": "clean_back"},
            {**row(sequence, "1", 2, f"{sequence}b.jpg", "not_clean"), "class_name": "not_clean"},
        ])
    manifest = {"split": "train", "train_sequences": list("ABCD"), "validation_sequences": ["V"]}
    train, validation, split = split_review_rows(rows, manifest, .75, 20260716)
    assert set(split["train_sequences"]).isdisjoint(split["validation_sequences"])
    assert {item["class_name"] for item in train} == {"clean_back", "not_clean"}
    assert {item["class_name"] for item in validation} == {"clean_back", "not_clean"}
    with pytest.raises(ValueError, match="frozen validation"):
        split_review_rows([{**rows[0], "sequence": "V"}], manifest, .8, 7)


def write_review(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def row(sequence, track, frame, crop, label):
    return {"sequence": sequence, "gt_track_id": track, "frame": frame,
            "crop_path": crop, "review_label": label}
