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
    _validate_tracking(config.get("tracking", {}), errors)
    _validate_scene_cuts(config.get("scene_cuts", {}), errors)
    _validate_jersey_ocr(config.get("jersey_ocr", {}), errors, warnings)
    _validate_visual(config.get("visual", {}), errors)
    _validate_identity(config.get("identity", {}), errors)
    _validate_identity_propagation(config.get("identity_propagation", {}), errors, warnings)

    if warnings:
        print("FT config warnings:\n- " + "\n- ".join(warnings), file=sys.stderr, flush=True)

    if errors:
        raise ValueError("Invalid FT run config:\n- " + "\n- ".join(errors))


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


def _validate_tracking(tracking, errors):
    backend = str(tracking.get("backend", "bytetrack")).lower().replace("_", "")
    if backend not in {"bytetrack", "byte", "strongsort", "strong"}:
        errors.append(f"tracking.backend must be bytetrack or strongsort, got {tracking.get('backend')!r}")


def _validate_scene_cuts(scene_cuts, errors):
    if not scene_cuts.get("enabled", False):
        return
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


def _validate_visual(visual, errors):
    mode = str(visual.get("embedding_mode", "hsv") or "hsv").strip().lower().replace("-", "_")
    if mode not in {"legacy", "hsv", "hsv_hist", "hsv_histogram", "hsv_lab_gradient", "lab_gradient", "rich", "enhanced"}:
        errors.append(f"visual.embedding_mode is unsupported: {visual.get('embedding_mode')!r}")


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
