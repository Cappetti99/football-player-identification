import hashlib

import numpy as np
import pytest

from ft.features.jersey_subject_filter import JerseySubjectFilter, iterative_gaussian_filter
from ft.validation import _validate_jersey_subject_filter


class FakeExtractor:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def add_row_features(self, rows):
        for row in rows:
            row["visual_embedding"] = self.embeddings[row["crop_path"]]
        return rows

    def diagnostics(self):
        return {"backend": "fake", "status": "ok"}


def make_filter(tmp_path, embeddings, **overrides):
    checkpoint = tmp_path / "prtreid.pth"
    checkpoint.write_bytes(b"prtreid")
    config = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(b"prtreid").hexdigest(),
        "device": "cpu",
        "extractor": FakeExtractor(embeddings),
    }
    config.update(overrides)
    return JerseySubjectFilter(**config)


def make_rows(tmp_path, vectors, role="player"):
    rows = []
    embeddings = {}
    for index, vector in enumerate(vectors):
        path = tmp_path / f"crop_{index}.jpg"
        path.write_bytes(b"crop")
        embeddings[str(path)] = np.asarray(vector, dtype=float)
        rows.append({
            "frame": index,
            "crop_path": str(path),
            "crop_quality": 1.0,
            "role_detection": role,
        })
    return rows, embeddings


def test_iterative_filter_uses_mean_plus_std_and_finds_extreme_outlier():
    vectors = [[0.0, 0.0]] * 20 + [[100.0, 0.0]]
    result = iterative_gaussian_filter(vectors, rounds=3, std_threshold=3.5, min_samples=3)
    assert result["kept_indices"] == list(range(20))
    assert result["excluded_rounds"] == {20: 1}


def test_small_sample_and_zero_std_do_not_filter():
    assert iterative_gaussian_filter([[0], [10]], min_samples=3)["kept_indices"] == [0, 1]
    result = iterative_gaussian_filter([[1, 1]] * 4, min_samples=3)
    assert result["kept_indices"] == [0, 1, 2, 3]
    assert result["excluded_rounds"] == {}


def test_audit_preserves_rows_and_propose_excludes_per_display_track(tmp_path):
    rows, embeddings = make_rows(tmp_path, [[0.0, 0.0]] * 20 + [[100.0, 0.0]])
    audit = make_filter(tmp_path, embeddings, mode="audit")
    assert audit.filter(7, rows) == rows
    assert audit.track_rows[0]["display_track_id"] == 7
    assert audit.track_rows[0]["excluded_crops"] == 1

    propose = make_filter(tmp_path, embeddings, mode="propose")
    effective = propose.filter(8, rows)
    assert len(effective) == 20
    assert effective == rows[:-1]
    assert {row["display_track_id"] for row in propose.score_rows} == {8}


def test_role_exclusion_and_min_remaining_fallback(tmp_path):
    rows, embeddings = make_rows(tmp_path, [[0.0, 0.0]] * 20 + [[100.0, 0.0]])
    rows[0]["role_detection"] = "referee"
    subject_filter = make_filter(tmp_path, embeddings, mode="propose", min_remaining=30)
    assert subject_filter.filter(3, rows) == rows
    assert subject_filter.score_rows[0]["filter_reason"] == "role_not_allowed"
    assert subject_filter.track_rows[0]["fallback"] is True


def test_missing_or_corrupt_checkpoint_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        JerseySubjectFilter(tmp_path / "missing.pth", "0" * 64, extractor=FakeExtractor({}))
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        JerseySubjectFilter(checkpoint, "0" * 64, extractor=FakeExtractor({}))


def test_subject_filter_validation_rejects_apply_mode():
    errors = []
    _validate_jersey_subject_filter({
        "enabled": True, "mode": "apply", "checkpoint": "model.pth",
        "checkpoint_sha256": "0" * 64, "batch_size": 1, "min_samples": 3,
        "min_remaining": 2, "rounds": 3, "std_threshold": 3.5,
        "min_crop_quality": .08, "allowed_roles": ["player"],
    }, errors)
    assert any("unsupported" in error for error in errors)
