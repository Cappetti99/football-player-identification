from ft.validation import validate_run_config


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
