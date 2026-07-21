from pathlib import Path

import pytest

from scripts.train_yolo_jersey_number_region import validate_dataset_manifest


def test_accepts_sjn_official_split(tmp_path):
    data = tmp_path / "number_region_yolo"
    data.mkdir()
    manifest = {
        "format": "sjn210k_ft_v1",
        "number_region_yolo": str(data / "data.yaml"),
        "official_split_preserved": True,
        "test_used_for_gradient_updates": False,
    }
    assert validate_dataset_manifest(manifest, data) == "sjn210k_ft_v1"


def test_rejects_sjn_test_gradient_leakage(tmp_path):
    data = tmp_path / "number_region_yolo"
    manifest = {
        "format": "sjn210k_ft_v1",
        "number_region_yolo": str(data / "data.yaml"),
        "official_split_preserved": True,
        "test_used_for_gradient_updates": True,
    }
    with pytest.raises(ValueError, match="gradients"):
        validate_dataset_manifest(manifest, data)


def test_preserves_legacy_gsr_validation():
    manifest = {
        "target": "jersey_number_region",
        "frozen_sequences_observed": [],
    }
    assert (
        validate_dataset_manifest(manifest, Path("unused"))
        == "gsr_jersey_number_region"
    )
