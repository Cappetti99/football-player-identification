from ft.validation import _validate_prtreid_identity_bridge, _validate_prtreid_linking, validate_run_config


def test_validation_reports_missing_inputs():
    config = {
        "video_path": "missing.mp4",
        "model_path": "missing.pt",
        "output_path": "output.mp4",
        "artifacts_dir": "artifacts/test",
        "max_frames": 0,
        "tracking": {"backend": "bad"},
        "calibration": {
            "enabled": True,
            "tvcalib": {
                "enabled": True,
                "path": "missing_tvcalib.json",
            },
        },
    }

    try:
        validate_run_config(config)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected config validation to fail")

    assert "video_path not found" in message
    assert "model_path not found" in message
    assert "calibration.tvcalib.path not found" in message
    assert "tracking.backend" in message
    assert "max_frames must be positive" in message


def test_validation_can_reject_unknown_config_keys(tmp_path):
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    video.write_bytes(b"x")
    model.write_bytes(b"x")
    config = {
        "run": {"unknown_config_keys": "error"},
        "video_path": str(video),
        "model_path": str(model),
        "output_path": str(tmp_path / "out.mp4"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "tracking": {"backend": "bytetrack"},
        "calibration": {"enabled": False},
        "jersey_ocr": {"typo_field": True},
    }

    try:
        validate_run_config(config)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected unknown config key validation to fail")

    assert "unknown config key: jersey_ocr.typo_field" in message


def test_validation_requires_prtreid_when_visual_mode_uses_prtreid(tmp_path):
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    video.write_bytes(b"x")
    model.write_bytes(b"x")
    config = {
        "video_path": str(video),
        "model_path": str(model),
        "output_path": str(tmp_path / "out.mp4"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "tracking": {"backend": "bytetrack"},
        "calibration": {"enabled": False},
        "visual": {"embedding_mode": "prtreid"},
        "prtreid": {"enabled": False},
    }

    try:
        validate_run_config(config)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected PRTReID validation to fail")

    assert "visual.embedding_mode=prtreid requires prtreid.enabled=true" in message


def test_validation_checks_prtreid_checkpoints_without_download(tmp_path):
    video = tmp_path / "video.mp4"
    model = tmp_path / "model.pt"
    video.write_bytes(b"x")
    model.write_bytes(b"x")
    config = {
        "video_path": str(video),
        "model_path": str(model),
        "output_path": str(tmp_path / "out.mp4"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "tracking": {"backend": "bytetrack"},
        "calibration": {"enabled": False},
        "visual": {"embedding_mode": "prtreid"},
        "prtreid": {
            "enabled": True,
            "weights_path": str(tmp_path / "missing.pth.tar"),
            "hrnet_pretrained_path": str(tmp_path / "missing_hrnet"),
            "download_weights": False,
            "batch_size": 1,
            "image_width": 128,
            "image_height": 256,
        },
    }

    try:
        validate_run_config(config)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected PRTReID checkpoint validation to fail")

    assert "prtreid.weights_path not found" in message
    assert "prtreid.hrnet_pretrained_path" in message


def test_prtreid_linking_validation_enforces_conservative_candidate():
    config = {
        "visual": {"embedding_mode": "prtreid"},
        "prtreid": {"enabled": True, "role_enabled": True},
        "team": {"assignment_mode": "prtreid_kmeans"},
        "identity_propagation": {"enabled": True},
        "prtreid_linking": {
            "enabled": True,
            "max_samples_per_tracklet": 2,
            "min_samples": 3,
            "min_crop_quality": 0.15,
            "min_prototype_consistency": 0.9,
            "min_team_confidence": 0.55,
            "max_diagnostic_records": 100,
            "same_scene": {"enabled": True, "max_gap": 10, "max_distance": 20, "min_similarity": 0.99, "min_margin": 0.01},
            "cross_scene": {"enabled": False},
        },
    }
    errors = []

    _validate_prtreid_linking(config, errors)

    assert "prtreid_linking candidate requires prtreid.role_enabled=false" in errors
    assert "prtreid_linking candidate requires team.assignment_mode=color" in errors
    assert "prtreid_linking candidate requires identity_propagation.enabled=false" in errors
    assert "prtreid_linking.min_samples cannot exceed max_samples_per_tracklet" in errors


def test_prtreid_identity_bridge_validation_requires_isolated_calibrated_apply():
    config = {
        "prtreid": {"enabled": True, "role_enabled": False},
        "identity_propagation": {"enabled": True},
        "jersey_identity_linking": {"enabled": True},
        "prtreid_linking": {"enabled": True},
        "prtreid_identity_bridge": {
            "enabled": True, "apply": True, "max_samples_per_tracklet": 8,
            "min_samples": 3, "max_segment_gap": 1, "max_gap": 60,
            "max_diagnostic_records": 10, "min_crop_quality": 0.15,
            "min_prototype_consistency": 0.9, "min_source_confidence": 0.75,
            "min_team_confidence": 0.55, "min_similarity": 1.0, "min_margin": 1.0,
        },
    }
    errors = []
    _validate_prtreid_identity_bridge(config, errors)
    assert any("identity_propagation" in error for error in errors)
    assert any("jersey_identity_linking" in error for error in errors)
    assert any("prtreid_linking" in error for error in errors)
    assert any("calibrated" in error for error in errors)
