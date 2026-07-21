import pytest

from scripts.build_sjn210k_ctc_dataset import padded_quad_box, validate_source_manifest
from scripts.train_jersey_number_ctc import validate_dataset_manifest


def test_padded_quad_box_clips_to_image():
    box = padded_quad_box([0.0, 0.1, 0.5, 0.1, 0.5, 0.9, 0.0, 0.9], 0.25)
    assert box == [0.0, 0.0, 0.625, 1.0]


def test_accepts_safe_sjn_ctc_manifest():
    manifest = {
        "format": "jersey_numeric_ctc_sjn210k_v1",
        "official_split_preserved": True,
        "test_used_for_gradient_updates": False,
    }
    assert validate_dataset_manifest(manifest) == "jersey_numeric_ctc_sjn210k_v1"


def test_rejects_sjn_source_test_leakage():
    manifest = {
        "format": "sjn210k_ft_v1",
        "official_split_preserved": True,
        "test_used_for_gradient_updates": True,
        "recognition": {"train": "train.jsonl", "test": "test.jsonl"},
    }
    with pytest.raises(ValueError, match="gradient"):
        validate_source_manifest(manifest)
