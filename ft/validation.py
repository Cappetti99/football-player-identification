from pathlib import Path
import sys

from ft.config import DEFAULT_CONFIG


def validate_run_config(config):
    errors = []
    warnings = []

    unknown_key_policy = str(config.get("run", {}).get("unknown_config_keys", "warn")).lower()
    _validate_unknown_keys(config, DEFAULT_CONFIG, errors, warnings, policy=unknown_key_policy)

    _require_existing_file(errors, config.get("video_path"), "video_path")
    _require_existing_file(errors, config.get("model_path"), "model_path")

    roster_path = config.get("roster_path")
    if roster_path:
        _require_existing_file(errors, roster_path, "roster_path")

    output_path = config.get("output_path")
    if not output_path:
        errors.append("output_path is required")
    else:
        _require_writable_parent(errors, output_path, "output_path")

    artifacts_dir = config.get("artifacts_dir")
    if not artifacts_dir:
        errors.append("artifacts_dir is required")
    else:
        _require_writable_parent(errors, Path(artifacts_dir) / "metadata", "artifacts_dir")

    max_frames = config.get("max_frames")
    if max_frames is not None and int(max_frames) <= 0:
        errors.append("max_frames must be positive when provided")

    _validate_calibration(config.get("calibration", {}), errors)
    _validate_detection(config.get("detection", {}), errors)
    _validate_tracking(config.get("tracking", {}), errors, prtreid=config.get("prtreid", {}))
    _validate_scene_cuts(config.get("scene_cuts", {}), config.get("tracking", {}), errors)
    _validate_activity_segmentation(config.get("activity_segmentation", {}), errors)
    _validate_linking_pitch_gate(config, errors)
    _validate_team(config.get("team", {}), errors)
    _validate_jersey_ocr(config.get("jersey_ocr", {}), errors, warnings)
    _validate_jersey_ocr_secondary_audit(
        config.get("jersey_ocr_secondary_audit", {}), errors, warnings
    )
    if (
        config.get("jersey_ocr_secondary_audit", {}).get("enabled", False)
        and not config.get("jersey_ocr", {}).get("enabled", False)
    ):
        errors.append("jersey_ocr_secondary_audit requires jersey_ocr.enabled=true")
    _validate_jersey_region_ctc_audit(
        config.get("jersey_region_ctc_audit", {}), errors
    )
    if (
        config.get("jersey_region_ctc_audit", {}).get("enabled", False)
        and not config.get("jersey_ocr", {}).get("enabled", False)
    ):
        errors.append("jersey_region_ctc_audit requires jersey_ocr.enabled=true")
    _validate_jersey_frame_selection(config.get("jersey_frame_selection", {}), errors)
    _validate_jersey_subject_filter(config.get("jersey_subject_filter", {}), errors)
    _validate_jersey_prefix_consolidation(config.get("jersey_prefix_consolidation", {}), errors)
    _validate_number_region(config.get("number_region", {}), errors, warnings)
    _validate_visual(config.get("visual", {}), config.get("prtreid", {}), errors)
    _validate_prtreid(
        config.get("prtreid", {}),
        config.get("visual", {}),
        config.get("prtreid_linking", {}),
        config.get("prtreid_identity_bridge", {}),
        errors,
        warnings,
        tracking=config.get("tracking", {}),
    )
    _validate_prtreid_linking(config, errors)
    _validate_prtreid_identity_bridge(config, errors)
    _validate_identity(config.get("identity", {}), errors)
    _validate_identity_propagation(config.get("identity_propagation", {}), errors, warnings)

    if warnings:
        print("FT config warnings:\n- " + "\n- ".join(warnings), file=sys.stderr, flush=True)

    if errors:
        raise ValueError("Invalid FT run config:\n- " + "\n- ".join(errors))


def _validate_linking_pitch_gate(config, errors):
    gate = config.get("linking", {}).get("pitch_gate") or {}
    reranking = config.get("linking", {}).get("pitch_reranking") or {}
    pitch_enabled = gate.get("enabled", False) or reranking.get("enabled", False)
    if pitch_enabled:
        if not config.get("linking", {}).get("enabled", True):
            errors.append("linking pitch features require linking.enabled=true")
        if not config.get("calibration", {}).get("tvcalib", {}).get("enabled", False):
            errors.append("linking pitch features require calibration.tvcalib.enabled=true")
    if gate.get("enabled", False):
        mode = str(gate.get("mode", "audit")).lower()
        if mode not in {"audit", "apply"}:
            errors.append("linking.pitch_gate.mode must be audit or apply")
        for key in ("max_speed_mps", "fps", "pitch_length", "pitch_width"):
            try:
                valid = float(gate.get(key, 0.0)) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"linking.pitch_gate.{key} must be positive")
    if reranking.get("enabled", False):
        mode = str(reranking.get("mode", "audit")).lower()
        if mode not in {"audit", "apply"}:
            errors.append("linking.pitch_reranking.mode must be audit or apply")
        for key in ("baseline_distance_scale", "pitch_distance_scale"):
            try:
                valid = float(reranking.get(key, 0.0)) > 0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"linking.pitch_reranking.{key} must be positive")
        try:
            weight_valid = float(reranking.get("weight", -1.0)) >= 0
        except (TypeError, ValueError):
            weight_valid = False
        if not weight_valid:
            errors.append("linking.pitch_reranking.weight must be non-negative")


def _validate_calibration(calibration, errors):
    if not calibration.get("enabled", True):
        return

    path = calibration.get("path")
    if path:
        _require_existing_file(errors, path, "calibration.path")

    tvcalib = calibration.get("tvcalib") or {}
    if tvcalib.get("enabled", False):
        _require_existing_file(errors, tvcalib.get("path"), "calibration.tvcalib.path")
        coordinate_system = str(tvcalib.get("coordinate_system", "tvcalib_centered")).lower()
        allowed = {
            "tvcalib_centered",
            "centered",
            "soccer_pitch_centered",
            "ft",
            "pitch",
            "top_left",
            "top_left_pitch",
            "none",
        }
        if coordinate_system not in allowed:
            errors.append(
                "calibration.tvcalib.coordinate_system must be one of "
                f"{sorted(allowed)}, got {coordinate_system!r}"
            )
        max_frame_gap = tvcalib.get("max_frame_gap")
        if max_frame_gap is not None and int(max_frame_gap) < 0:
            errors.append("calibration.tvcalib.max_frame_gap must be non-negative")


def _validate_detection(detection, errors):
    for key in (
        "confidence",
        "ball_confidence",
        "ball_max_area_ratio",
        "ball_size_penalty",
        "ball_temporal_distance_penalty",
        "ball_min_acquisition_confidence",
        "ball_temporal_min_confidence_after_miss",
        "ball_kalman_process_noise_scale",
        "ball_kalman_measurement_noise_scale",
        "ball_kalman_high_speed_threshold",
    ):
        if key in detection and float(detection.get(key) or 0.0) < 0.0:
            errors.append(f"detection.{key} must be non-negative")
    if "ball_temporal_max_distance" in detection and float(detection.get("ball_temporal_max_distance") or 0.0) <= 0.0:
        errors.append("detection.ball_temporal_max_distance must be positive")
    if "ball_temporal_max_distance_cap" in detection and float(detection.get("ball_temporal_max_distance_cap") or 0.0) < 0.0:
        errors.append("detection.ball_temporal_max_distance_cap must be non-negative")
    if "ball_low_confidence_max_distance" in detection and float(detection.get("ball_low_confidence_max_distance") or 0.0) <= 0.0:
        errors.append("detection.ball_low_confidence_max_distance must be positive")
    if "ball_temporal_miss_reset" in detection and int(detection.get("ball_temporal_miss_reset") or 0) < 0:
        errors.append("detection.ball_temporal_miss_reset must be non-negative")
    if "ball_kalman_max_lost_frames" in detection and int(detection.get("ball_kalman_max_lost_frames") or 0) < 0:
        errors.append("detection.ball_kalman_max_lost_frames must be non-negative")
    if "ball_kalman_high_speed_area_multiplier" in detection and float(detection.get("ball_kalman_high_speed_area_multiplier") or 0.0) < 1.0:
        errors.append("detection.ball_kalman_high_speed_area_multiplier must be >= 1")


def _validate_tracking(tracking, errors, prtreid=None):
    backend = str(tracking.get("backend", "bytetrack")).lower().replace("_", "")
    if backend not in {"bytetrack", "byte"}:
        errors.append(f"tracking.backend must be bytetrack, got {tracking.get('backend')!r}")
    if tracking.get("prtreid_detection_enrichment", False) and not (prtreid or {}).get("enabled", False):
        errors.append("tracking.prtreid_detection_enrichment=true requires prtreid.enabled=true")


def _validate_scene_cuts(scene_cuts, tracking, errors):
    if not scene_cuts.get("enabled", False):
        return
    backend = str(tracking.get("backend", "bytetrack")).lower().replace("_", "")
    if scene_cuts.get("tracking_reset_enabled", False) and backend not in {"bytetrack", "byte"}:
        errors.append("scene_cuts.tracking_reset_enabled=true requires tracking.backend=bytetrack")
    for key in ("threshold", "crop_top_fraction", "crop_bottom_fraction", "crop_left_fraction", "crop_right_fraction"):
        value = float(scene_cuts.get(key) or 0.0)
        if value < 0.0:
            errors.append(f"scene_cuts.{key} must be non-negative")
    threshold = float(scene_cuts.get("threshold") or 0.0)
    if threshold <= 0.0:
        errors.append("scene_cuts.threshold must be positive")
    hard_cut_threshold = scene_cuts.get("hard_cut_threshold")
    if hard_cut_threshold is not None and float(hard_cut_threshold) <= 0.0:
        errors.append("scene_cuts.hard_cut_threshold must be positive when provided")
    if int(scene_cuts.get("min_gap") or 0) <= 0:
        errors.append("scene_cuts.min_gap must be positive")
    if int(scene_cuts.get("resize_width") or 0) < 0:
        errors.append("scene_cuts.resize_width must be non-negative")
    for key in ("h_bins", "s_bins"):
        if int(scene_cuts.get(key) or 0) <= 1:
            errors.append(f"scene_cuts.{key} must be greater than 1")
    max_cuts = scene_cuts.get("max_cuts")
    if max_cuts is not None and int(max_cuts) <= 0:
        errors.append("scene_cuts.max_cuts must be positive when provided")


def _validate_activity_segmentation(activity, errors):
    if not activity.get("enabled", False):
        return
    for key in (
        "smoothing_window",
        "count_change_threshold",
        "persistence_frames",
        "min_segment_frames",
    ):
        if int(activity.get(key) or 0) <= 0:
            errors.append(f"activity_segmentation.{key} must be positive")
    ratio = float(activity.get("persistence_ratio") or 0.0)
    if not 0.5 <= ratio <= 1.0:
        errors.append("activity_segmentation.persistence_ratio must be between 0.5 and 1.0")


def _validate_team(team, errors):
    assignment_mode = str(team.get("assignment_mode", "color") or "color").lower()
    if assignment_mode not in {"color", "prtreid_kmeans"}:
        errors.append("team.assignment_mode must be color or prtreid_kmeans")


def _validate_jersey_ocr(jersey_ocr, errors, warnings):
    backend = str(jersey_ocr.get("backend", "auto") or "auto").lower()
    if backend not in {
        "auto",
        "default",
        "easyocr",
        "pytesseract",
        "paddleocr",
        "paddle_ocr",
        "paddle",
        "paddleocr_easyocr",
        "paddleocr+easyocr",
        "paddle_ocr_easyocr",
        "mmocr",
        "mmocr_easyocr",
        "mmocr+easyocr",
        "mmocr-fallback",
        "mmocr_auto",
        "mmocr_rec",
    } and "," not in backend and "+" not in backend:
        errors.append(f"jersey_ocr.backend is unsupported: {jersey_ocr.get('backend')!r}")

    for key in (
        "min_confidence",
        "min_raw_confidence",
        "min_winner_margin",
        "min_crop_candidate_ratio",
        "crop_quality_min_vote_weight",
        "template_min_score",
        "template_weight",
    ):
        if key in jersey_ocr:
            value = float(jersey_ocr.get(key) or 0.0)
            if value < 0.0 or value > 1.0:
                errors.append(f"jersey_ocr.{key} must be between 0 and 1, got {value}")

    for key in (
        "max_crops_per_tracklet",
        "min_votes",
        "max_candidates_per_crop",
        "mmocr_batch_size",
        "super_resolution_scale",
        "super_resolution_max_side",
    ):
        if key in jersey_ocr and int(jersey_ocr.get(key) or 0) <= 0:
            errors.append(f"jersey_ocr.{key} must be positive")

    if "broadcast_contrast_clip_limit" in jersey_ocr and float(jersey_ocr.get("broadcast_contrast_clip_limit") or 0.0) <= 0.0:
        errors.append("jersey_ocr.broadcast_contrast_clip_limit must be positive")
    if "crop_quality_vote_power" in jersey_ocr and float(jersey_ocr.get("crop_quality_vote_power") or 0.0) <= 0.0:
        errors.append("jersey_ocr.crop_quality_vote_power must be positive")
    if "broadcast_contrast_tile_grid_size" in jersey_ocr and int(jersey_ocr.get("broadcast_contrast_tile_grid_size") or 0) <= 1:
        errors.append("jersey_ocr.broadcast_contrast_tile_grid_size must be greater than 1")

    if "segment_frames" in jersey_ocr and int(jersey_ocr.get("segment_frames") or 0) < 0:
        errors.append(f"jersey_ocr.segment_frames must be non-negative")
    if "segment_candidate_frames" in jersey_ocr and int(jersey_ocr.get("segment_candidate_frames") or 0) < 0:
        errors.append(f"jersey_ocr.segment_candidate_frames must be non-negative")

    if jersey_ocr.get("roster_aware", True) and jersey_ocr.get("promote_roster_candidate", True):
        warnings.append(
            "jersey_ocr.roster_aware with promote_roster_candidate=true can promote noisy OCR alternatives; "
            "use cautiously on SGR-style experiments"
        )

    if jersey_ocr.get("mmocr_direct_recognition") and not jersey_ocr.get("number_roi_enabled", False):
        warnings.append(
            "jersey_ocr.mmocr_direct_recognition=true without number_roi_enabled=true may increase false positives"
        )
    if jersey_ocr.get("mmocr_rec_weights"):
        digest = str(jersey_ocr.get("mmocr_rec_weights_sha256") or "")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            errors.append(
                "jersey_ocr.mmocr_rec_weights_sha256 must be a SHA-256 hex digest "
                "with custom recognizer weights"
            )


def _validate_jersey_ocr_secondary_audit(audit, errors, warnings):
    if not audit.get("enabled", False):
        return
    mode = str(audit.get("mode", "audit") or "audit").lower()
    if mode not in {"audit", "propose"}:
        errors.append("jersey_ocr_secondary_audit supports only audit/propose")
    _validate_jersey_ocr(audit, errors, warnings)
    if not audit.get("mmocr_rec_weights"):
        errors.append("jersey_ocr_secondary_audit.mmocr_rec_weights is required")


def _validate_jersey_region_ctc_audit(audit, errors):
    if not audit.get("enabled", False):
        return
    if str(audit.get("mode", "audit")).lower() != "audit":
        errors.append("jersey_region_ctc_audit supports only audit")
    if not audit.get("audit_only", True):
        errors.append("jersey_region_ctc_audit.audit_only must remain true")
    if not isinstance(audit.get("fusion_preview_enabled", True), bool):
        errors.append("jersey_region_ctc_audit.fusion_preview_enabled must be boolean")
    if not isinstance(audit.get("roster_reranking_enabled", False), bool):
        errors.append("jersey_region_ctc_audit.roster_reranking_enabled must be boolean")
    for key in ("ctc_checkpoint", "detector_checkpoint"):
        if not audit.get(key):
            errors.append(f"jersey_region_ctc_audit.{key} is required")
        digest = str(audit.get(f"{key}_sha256") or "")
        if len(digest) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in digest
        ):
            errors.append(
                f"jersey_region_ctc_audit.{key}_sha256 must be a SHA-256 hex digest"
            )
    for key in ("detector_confidence", "min_override_confidence"):
        value = float(audit.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            errors.append(f"jersey_region_ctc_audit.{key} must be between 0 and 1")
    padding = float(audit.get("box_padding", 0.0))
    if not 0.0 <= padding <= 0.5:
        errors.append("jersey_region_ctc_audit.box_padding must be between 0 and 0.5")
    for key in ("batch_size", "detector_batch_size", "max_crops_per_tracklet"):
        if int(audit.get(key) or 0) <= 0:
            errors.append(f"jersey_region_ctc_audit.{key} must be positive")
    if int(audit.get("min_frame_gap") or 0) < 0:
        errors.append("jersey_region_ctc_audit.min_frame_gap must be non-negative")


def _validate_jersey_frame_selection(selector, errors):
    if not selector.get("enabled", False):
        return
    mode = str(selector.get("mode", "audit") or "audit").lower()
    model_type = str(selector.get("model_type", "legibility_resnet34") or "legibility_resnet34").lower()
    if mode not in {"audit", "propose", "apply"}:
        errors.append(f"jersey_frame_selection.mode is unsupported: {mode!r}")
    yolo_model_types = {
        "jersey_back_yolo11s_cls",
        "jersey_number_readability_yolo26s_cls",
    }
    if model_type not in {"legibility_resnet34", *yolo_model_types}:
        errors.append(f"jersey_frame_selection.model_type is unsupported: {model_type!r}")
    if model_type == "jersey_back_yolo11s_cls" and mode == "apply":
        errors.append("jersey back YOLO classifier supports only audit/propose")
    if model_type == "jersey_number_readability_yolo26s_cls" and mode == "apply":
        errors.append("jersey readability YOLO classifier supports only audit/propose")
    if not selector.get("checkpoint"):
        errors.append("jersey_frame_selection.checkpoint is required when enabled")
    digest = str(selector.get("checkpoint_sha256") or "")
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        errors.append("jersey_frame_selection.checkpoint_sha256 must be a SHA-256 hex digest")
    for key in ("batch_size", "top_k", "min_winner_votes"):
        if int(selector.get(key) or 0) <= 0:
            errors.append(f"jersey_frame_selection.{key} must be positive")
    if int(selector.get("min_frame_gap") or 0) < 0:
        errors.append("jersey_frame_selection.min_frame_gap must be non-negative")
    for key in ("min_legibility_score", "min_margin"):
        value = float(selector.get(key, 0.0))
        if value < 0.0 or value > 1.0:
            errors.append(f"jersey_frame_selection.{key} must be between 0 and 1")
    if selector.get("min_selection_score") is not None:
        value = float(selector["min_selection_score"])
        if value < 0.0 or value > 1.0:
            errors.append("jersey_frame_selection.min_selection_score must be between 0 and 1")
    weight_keys = (
        "clean_back_weight", "sharpness_weight", "size_weight", "crop_quality_weight"
    )
    weight_defaults = (0.70, 0.15, 0.05, 0.10)
    weights = [
        float(selector.get(key, default))
        for key, default in zip(weight_keys, weight_defaults)
    ]
    if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
        errors.append("jersey_frame_selection score weights must be non-negative and sum above zero")
    for key, default in (("sharpness_scale", 100.0), ("size_scale", 160.0)):
        if float(selector.get(key, default)) <= 0.0:
            errors.append(f"jersey_frame_selection.{key} must be positive")
    x0 = float(selector.get("torso_x_min", 0.0))
    x1 = float(selector.get("torso_x_max", 1.0))
    y0 = float(selector.get("torso_y_min", 0.0))
    y1 = float(selector.get("torso_y_max", 1.0))
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        errors.append("jersey_frame_selection torso bounds must be ordered values between 0 and 1")
    roles = selector.get("allowed_roles")
    if not isinstance(roles, list) or not roles:
        errors.append("jersey_frame_selection.allowed_roles must be a non-empty list")


def _validate_jersey_subject_filter(subject_filter, errors):
    if not subject_filter.get("enabled", False):
        return
    mode = str(subject_filter.get("mode", "audit") or "audit").lower()
    if mode not in {"audit", "propose"}:
        errors.append(f"jersey_subject_filter.mode is unsupported: {mode!r}")
    if not subject_filter.get("checkpoint"):
        errors.append("jersey_subject_filter.checkpoint is required when enabled")
    digest = str(subject_filter.get("checkpoint_sha256") or "")
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        errors.append("jersey_subject_filter.checkpoint_sha256 must be a SHA-256 hex digest")
    for key in ("batch_size", "min_samples", "min_remaining", "rounds"):
        if int(subject_filter.get(key) or 0) <= 0:
            errors.append(f"jersey_subject_filter.{key} must be positive")
    if float(subject_filter.get("std_threshold", -1.0)) < 0:
        errors.append("jersey_subject_filter.std_threshold must be non-negative")
    quality = float(subject_filter.get("min_crop_quality", -1.0))
    if quality < 0.0 or quality > 1.0:
        errors.append("jersey_subject_filter.min_crop_quality must be between 0 and 1")
    roles = subject_filter.get("allowed_roles")
    if not isinstance(roles, list) or not roles:
        errors.append("jersey_subject_filter.allowed_roles must be a non-empty list")


def _validate_jersey_prefix_consolidation(prefix, errors):
    if not prefix.get("enabled", False):
        return
    mode = str(prefix.get("mode", "audit") or "audit").lower()
    if mode not in {"audit", "propose"}:
        errors.append(f"jersey_prefix_consolidation.mode is unsupported: {mode!r}")
    if int(prefix.get("min_long_votes") or 0) <= 0:
        errors.append("jersey_prefix_consolidation.min_long_votes must be positive")
    weights = prefix.get("short_vote_weights")
    if not isinstance(weights, list) or not weights or any(float(value) < 0 or float(value) > 1 for value in weights):
        errors.append("jersey_prefix_consolidation.short_vote_weights must contain values between 0 and 1")
        weights = []
    selected = prefix.get("selected_short_vote_weight")
    if mode == "propose" and selected is None:
        errors.append("jersey_prefix_consolidation.selected_short_vote_weight is required in propose mode")
    elif selected is not None and float(selected) not in [float(value) for value in weights]:
        errors.append("jersey_prefix_consolidation.selected_short_vote_weight must be in short_vote_weights")


def _validate_number_region(number_region, errors, warnings):
    if not number_region.get("enabled", False):
        return
    if not number_region.get("audit_only", True):
        errors.append("number_region.audit_only must remain true; identity application is not implemented")
    if str(number_region.get("backend", "torso_heuristic")).lower() != "torso_heuristic":
        errors.append("number_region.backend currently supports only torso_heuristic")
    for key in ("frame_interval", "min_width", "min_height"):
        if int(number_region.get(key) or 0) <= 0:
            errors.append(f"number_region.{key} must be positive")
    if int(number_region.get("max_regions_per_tracklet") or 0) < 0:
        errors.append("number_region.max_regions_per_tracklet must be non-negative")
    if int(number_region.get("max_regions_per_tracklet") or 0) == 0:
        warnings.append(
            "number_region.max_regions_per_tracklet=0 is unbounded and can make OCR expensive on long videos"
        )
    for key in ("torso_x_min", "torso_x_max", "torso_y_min", "torso_y_max"):
        value = float(number_region.get(key, 0.0))
        if value < 0.0 or value > 1.0:
            errors.append(f"number_region.{key} must be between 0 and 1")
    if float(number_region.get("torso_x_min", 0.0)) >= float(number_region.get("torso_x_max", 1.0)):
        errors.append("number_region.torso_x_min must be lower than torso_x_max")
    if float(number_region.get("torso_y_min", 0.0)) >= float(number_region.get("torso_y_max", 1.0)):
        errors.append("number_region.torso_y_min must be lower than torso_y_max")
    if float(number_region.get("padding", 0.0)) < 0.0:
        errors.append("number_region.padding must be non-negative")
    if not number_region.get("audit_only", True):
        warnings.append("number_region is not isolated from identity application")


def _validate_visual(visual, prtreid, errors):
    mode = str(visual.get("embedding_mode", "hsv") or "hsv").strip().lower().replace("-", "_")
    if mode not in {
        "legacy",
        "hsv",
        "hsv_hist",
        "hsv_histogram",
        "hsv_lab_gradient",
        "lab_gradient",
        "rich",
        "enhanced",
        "prtreid",
        "prt_reid",
        "bpbreid",
        "prtreid_soccernet",
    }:
        errors.append(f"visual.embedding_mode is unsupported: {visual.get('embedding_mode')!r}")
    if mode in {"prtreid", "prt_reid", "bpbreid", "prtreid_soccernet"} and not prtreid.get("enabled", False):
        errors.append("visual.embedding_mode=prtreid requires prtreid.enabled=true")


def _validate_prtreid(
    prtreid,
    visual,
    prtreid_linking,
    prtreid_identity_bridge,
    errors,
    warnings,
    tracking=None,
):
    if not prtreid.get("enabled", False):
        return
    if int(prtreid.get("batch_size") or 0) <= 0:
        errors.append("prtreid.batch_size must be positive")
    if int(prtreid.get("image_width") or 0) <= 0:
        errors.append("prtreid.image_width must be positive")
    if int(prtreid.get("image_height") or 0) <= 0:
        errors.append("prtreid.image_height must be positive")
    if float(prtreid.get("role_min_confidence", 0.0) or 0.0) < 0.0:
        errors.append("prtreid.role_min_confidence must be non-negative")
    if not prtreid.get("download_weights", False):
        _require_existing_file(errors, prtreid.get("weights_path"), "prtreid.weights_path")
        hrnet_dir = Path(prtreid.get("hrnet_pretrained_path") or "")
        _require_existing_file(errors, hrnet_dir / "hrnetv2_w32_imagenet_pretrained.pth", "prtreid.hrnet_pretrained_path/hrnetv2_w32_imagenet_pretrained.pth")
    if (
        str(visual.get("embedding_mode", "")).lower().replace("-", "_") != "prtreid"
        and not prtreid_linking.get("enabled", False)
        and not prtreid_identity_bridge.get("enabled", False)
        and not (tracking or {}).get("prtreid_detection_enrichment", False)
    ):
        warnings.append("prtreid.enabled=true but visual.embedding_mode is not prtreid; PRTReID will not run")


def _validate_prtreid_identity_bridge(config, errors):
    bridge = config.get("prtreid_identity_bridge", {})
    if not bridge.get("enabled", False):
        return
    if not config.get("prtreid", {}).get("enabled", False):
        errors.append("prtreid_identity_bridge.enabled=true requires prtreid.enabled=true")
    if config.get("prtreid", {}).get("role_enabled", False):
        errors.append("prtreid_identity_bridge requires prtreid.role_enabled=false")
    if not config.get("team", {}).get("enabled", True):
        errors.append("prtreid_identity_bridge requires team.enabled=true")
    if str(config.get("team", {}).get("assignment_mode", "color")).lower() != "color":
        errors.append("prtreid_identity_bridge requires team.assignment_mode=color")
    if config.get("identity_propagation", {}).get("enabled", False):
        errors.append("prtreid_identity_bridge requires identity_propagation.enabled=false")
    if config.get("jersey_identity_linking", {}).get("enabled", False):
        errors.append("prtreid_identity_bridge requires jersey_identity_linking.enabled=false")
    if config.get("prtreid_linking", {}).get("enabled", False):
        errors.append("prtreid_identity_bridge requires prtreid_linking.enabled=false for an isolated ablation")
    if bridge.get("apply", False) and (
        float(bridge.get("min_similarity", 0.0)) >= 1.0
        and float(bridge.get("min_margin", 0.0)) >= 1.0
    ):
        errors.append("prtreid_identity_bridge.apply=true requires calibrated similarity or margin below 1")
    for key in ("max_samples_per_tracklet", "min_samples", "max_segment_gap", "max_gap", "max_diagnostic_records"):
        if int(bridge.get(key) or 0) <= 0:
            errors.append(f"prtreid_identity_bridge.{key} must be positive")
    if int(bridge.get("min_samples") or 0) > int(bridge.get("max_samples_per_tracklet") or 0):
        errors.append("prtreid_identity_bridge.min_samples cannot exceed max_samples_per_tracklet")
    for key in (
        "min_crop_quality", "min_prototype_consistency", "min_source_confidence",
        "min_team_confidence", "min_similarity", "min_margin",
    ):
        value = float(bridge.get(key) or 0.0)
        if value < 0.0 or value > 1.0:
            errors.append(f"prtreid_identity_bridge.{key} must be between 0 and 1, got {value}")


def _validate_prtreid_linking(config, errors):
    linking = config.get("prtreid_linking", {})
    if not linking.get("enabled", False):
        return
    prtreid = config.get("prtreid", {})
    if not prtreid.get("enabled", False):
        errors.append("prtreid_linking.enabled=true requires prtreid.enabled=true")
    if prtreid.get("role_enabled", False):
        errors.append("prtreid_linking candidate requires prtreid.role_enabled=false")
    if not config.get("linking", {}).get("enabled", True):
        errors.append("prtreid_linking candidate requires linking.enabled=true so it remains additive")
    if not config.get("team", {}).get("enabled", True):
        errors.append("prtreid_linking candidate requires team.enabled=true")
    if str(config.get("team", {}).get("assignment_mode", "color")).lower() != "color":
        errors.append("prtreid_linking candidate requires team.assignment_mode=color")
    if config.get("identity_propagation", {}).get("enabled", False):
        errors.append("prtreid_linking candidate requires identity_propagation.enabled=false")
    for key in ("max_samples_per_tracklet", "min_samples", "max_diagnostic_records"):
        if int(linking.get(key) or 0) <= 0:
            errors.append(f"prtreid_linking.{key} must be positive")
    for key in ("min_crop_quality", "min_prototype_consistency", "min_team_confidence"):
        value = float(linking.get(key) or 0.0)
        if value < 0.0 or value > 1.0:
            errors.append(f"prtreid_linking.{key} must be between 0 and 1, got {value}")
    if int(linking.get("min_samples") or 0) > int(linking.get("max_samples_per_tracklet") or 0):
        errors.append("prtreid_linking.min_samples cannot exceed max_samples_per_tracklet")
    for policy_name in ("same_scene", "cross_scene"):
        policy = linking.get(policy_name, {})
        if not policy.get("enabled", False):
            continue
        if int(policy.get("max_gap") or 0) <= 0:
            errors.append(f"prtreid_linking.{policy_name}.max_gap must be positive")
        for key in ("min_similarity", "min_margin"):
            value = float(policy.get(key) or 0.0)
            if value < 0.0 or value > 1.0:
                errors.append(f"prtreid_linking.{policy_name}.{key} must be between 0 and 1, got {value}")
    same_scene = linking.get("same_scene", {})
    if same_scene.get("enabled", False) and float(same_scene.get("max_distance") or 0.0) <= 0.0:
        errors.append("prtreid_linking.same_scene.max_distance must be positive")
    cross_scene = linking.get("cross_scene", {})
    if cross_scene.get("enabled", False) and int(cross_scene.get("max_segment_gap") or 0) <= 0:
        errors.append("prtreid_linking.cross_scene.max_segment_gap must be positive")


def _validate_identity_propagation(identity_propagation, errors, warnings):
    if not identity_propagation.get("enabled", False):
        return
    for key in (
        "min_composite_score",
        "min_score_margin",
        "min_source_confidence",
        "min_team_confidence",
        "min_appearance_similarity",
        "strong_appearance_similarity",
        "min_partial_fraction",
        "temporal_overlap_score",
    ):
        value = float(identity_propagation.get(key) or 0.0)
        if value < 0.0 or value > 1.0:
            errors.append(f"identity_propagation.{key} must be between 0 and 1, got {value}")
    for key in ("max_hops", "max_temporal_gap", "min_partial_frames"):
        if int(identity_propagation.get(key) or 0) <= 0:
            errors.append(f"identity_propagation.{key} must be positive")
    if int(identity_propagation.get("cut_bridge_max_gap") or 0) <= 0:
        errors.append("identity_propagation.cut_bridge_max_gap must be positive")
    if int(identity_propagation.get("cut_bridge_min_jersey_votes") or 0) < 0:
        errors.append("identity_propagation.cut_bridge_min_jersey_votes must be non-negative")
    if float(identity_propagation.get("cut_bridge_min_jersey_confidence") or 0.0) < 0.0:
        errors.append("identity_propagation.cut_bridge_min_jersey_confidence must be non-negative")
    if float(identity_propagation.get("max_spatial_distance") or 0.0) <= 0.0:
        errors.append("identity_propagation.max_spatial_distance must be positive")
    if int(identity_propagation.get("conflict_buffer") or 0) < 0:
        errors.append("identity_propagation.conflict_buffer must be non-negative")
    if int(identity_propagation.get("max_hops", 1) or 1) > 1 and identity_propagation.get("allow_propagated_sources", False):
        warnings.append(
            "identity_propagation with max_hops>1 and allow_propagated_sources=true can amplify mistakes; "
            "verify identity_propagation diagnostics and constraints"
        )

def _validate_identity(identity, errors):
    if str(identity.get("assignment_scope", "global")).lower() not in {"global", "scene_segment"}:
        errors.append("identity.assignment_scope must be global or scene_segment")
    for key in ("number_region_bonus_weight", "number_region_mismatch_penalty"):
        if key in identity and float(identity.get(key) or 0.0) < 0.0:
            errors.append(f"identity.{key} must be non-negative")
    for key in ("number_region_min_votes", "number_region_min_consecutive_support"):
        if key in identity and int(identity.get(key) or 0) < 0:
            errors.append(f"identity.{key} must be non-negative")
    if "number_region_min_mean_confidence" in identity:
        value = float(identity.get("number_region_min_mean_confidence") or 0.0)
        if value < 0.0 or value > 1.0:
            errors.append(f"identity.number_region_min_mean_confidence must be between 0 and 1, got {value}")
    if "goalkeeper_only_alternate_min_confidence" in identity:
        value = float(identity.get("goalkeeper_only_alternate_min_confidence") or 0.0)
        if value < 0.0 or value > 1.0:
            errors.append(f"identity.goalkeeper_only_alternate_min_confidence must be between 0 and 1, got {value}")
    if "goalkeeper_only_alternate_min_votes" in identity and int(identity.get("goalkeeper_only_alternate_min_votes") or 0) < 0:
        errors.append("identity.goalkeeper_only_alternate_min_votes must be non-negative")
    if "goalkeeper_only_alternate_max_rank" in identity and int(identity.get("goalkeeper_only_alternate_max_rank") or 0) <= 0:
        errors.append("identity.goalkeeper_only_alternate_max_rank must be positive")
    candidate_fallback = identity.get("candidate_fallback") or {}
    if "max_jersey_display_spread" in candidate_fallback:
        value = candidate_fallback.get("max_jersey_display_spread")
        if value not in (None, "", 0, "0") and int(value) <= 0:
            errors.append("identity.candidate_fallback.max_jersey_display_spread must be positive when enabled")
    if str(candidate_fallback.get("display_spread_scope", "team")).lower() not in {"team", "global"}:
        errors.append("identity.candidate_fallback.display_spread_scope must be 'team' or 'global'")


def _validate_unknown_keys(config, schema, errors, warnings, prefix="", policy="warn"):
    if not isinstance(config, dict) or not isinstance(schema, dict):
        return
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            message = f"unknown config key: {path}"
            if policy == "error":
                errors.append(message)
            elif policy != "ignore":
                warnings.append(message)
            continue
        if isinstance(value, dict) and isinstance(schema.get(key), dict):
            _validate_unknown_keys(value, schema[key], errors, warnings, path, policy=policy)


def _require_existing_file(errors, path, label):
    if not path:
        errors.append(f"{label} is required")
        return
    path = Path(path)
    if not path.exists():
        errors.append(f"{label} not found: {path}")
    elif not path.is_file():
        errors.append(f"{label} is not a file: {path}")


def _require_writable_parent(errors, path, label):
    parent = Path(path).parent
    if parent.exists() and not parent.is_dir():
        errors.append(f"{label} parent is not a directory: {parent}")
