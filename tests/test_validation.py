from ft.validation import (
    _validate_activity_segmentation,
    _validate_jersey_frame_selection,
    _validate_prtreid,
    _validate_prtreid_identity_bridge,
    _validate_prtreid_linking,
    _validate_scene_cuts,
    _validate_tracking,
    validate_run_config,
)


def test_jersey_back_validation_rejects_apply_and_bad_score():
    errors = []
    _validate_jersey_frame_selection(
        {
            "enabled": True,
            "model_type": "jersey_back_yolo11s_cls",
            "mode": "apply",
            "checkpoint": "model.pth",
            "checkpoint_sha256": "a" * 64,
            "batch_size": 1,
            "top_k": 1,
            "min_winner_votes": 1,
            "min_frame_gap": 0,
            "min_legibility_score": 0.5,
            "min_selection_score": 1.1,
            "min_margin": 0.0,
            "allowed_roles": ["player"],
        },
        errors,
    )
    assert "jersey back YOLO classifier supports only audit/propose" in errors
    assert "jersey_frame_selection.min_selection_score must be between 0 and 1" in errors


def test_jersey_back_yolo_validation_accepts_propose():
    errors = []
    _validate_jersey_frame_selection(
        {
            "enabled": True,
            "model_type": "jersey_back_yolo11s_cls",
            "mode": "propose",
            "checkpoint": "/tmp/model.pt",
            "checkpoint_sha256": "a" * 64,
            "batch_size": 32,
            "top_k": 5,
            "min_winner_votes": 2,
            "min_frame_gap": 5,
            "min_legibility_score": 0.5,
            "min_selection_score": 0.0,
            "min_margin": 0.0,
            "allowed_roles": ["player"],
        },
        errors,
    )
    assert errors == []


def test_activity_segmentation_validation_rejects_unsafe_values():
    errors = []
    _validate_activity_segmentation(
        {
            "enabled": True,
            "smoothing_window": 0,
            "count_change_threshold": 0,
            "persistence_frames": 0,
            "persistence_ratio": 0.4,
            "min_segment_frames": 0,
        },
        errors,
    )

    assert len(errors) == 5


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


def test_validation_rejects_removed_tracker_and_unsupported_scene_reset():
    errors = []
    _validate_tracking({"backend": "prtreid_strongsort"}, errors)
    _validate_scene_cuts(
        {"enabled": True, "tracking_reset_enabled": True, "threshold": 0.1},
        {"backend": "strongsort"},
        errors,
    )

    assert "tracking.backend must be bytetrack, got 'prtreid_strongsort'" in errors
    assert "scene_cuts.tracking_reset_enabled=true requires tracking.backend=bytetrack" in errors


def test_tracking_detection_enrichment_requires_prtreid():
    errors = []
    _validate_tracking(
        {"backend": "bytetrack", "prtreid_detection_enrichment": True},
        errors,
        prtreid={"enabled": False},
    )
    assert errors == [
        "tracking.prtreid_detection_enrichment=true requires prtreid.enabled=true"
    ]


def test_prtreid_detection_enrichment_suppresses_not_used_warning():
    warnings = []
    _validate_prtreid(
        {
            "enabled": True,
            "batch_size": 1,
            "image_width": 128,
            "image_height": 256,
            "download_weights": True,
        },
        {"embedding_mode": "hsv"},
        {"enabled": False},
        {"enabled": False},
        [],
        warnings,
        tracking={"prtreid_detection_enrichment": True},
    )
    assert warnings == []


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
