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
    _validate_tracking(config.get("tracking", {}), errors)
    _validate_jersey_ocr(config.get("jersey_ocr", {}), errors, warnings)

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


def _validate_tracking(tracking, errors):
    backend = str(tracking.get("backend", "bytetrack")).lower().replace("_", "")
    if backend not in {"bytetrack", "byte", "strongsort", "strong"}:
        errors.append(f"tracking.backend must be bytetrack or strongsort, got {tracking.get('backend')!r}")


def _validate_jersey_ocr(jersey_ocr, errors, warnings):
    backend = str(jersey_ocr.get("backend", "auto") or "auto").lower()
    if backend not in {
        "auto",
        "default",
        "easyocr",
        "pytesseract",
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
        "template_min_score",
        "template_weight",
    ):
        if key in jersey_ocr:
            value = float(jersey_ocr.get(key) or 0.0)
            if value < 0.0 or value > 1.0:
                errors.append(f"jersey_ocr.{key} must be between 0 and 1, got {value}")

    for key in ("max_crops_per_tracklet", "min_votes", "max_candidates_per_crop", "mmocr_batch_size"):
        if key in jersey_ocr and int(jersey_ocr.get(key) or 0) <= 0:
            errors.append(f"jersey_ocr.{key} must be positive")

    if jersey_ocr.get("roster_aware", True) and jersey_ocr.get("promote_roster_candidate", True):
        warnings.append(
            "jersey_ocr.roster_aware with promote_roster_candidate=true can promote noisy OCR alternatives; "
            "use cautiously on SGR-style experiments"
        )

    if jersey_ocr.get("mmocr_direct_recognition") and not jersey_ocr.get("number_roi_enabled", False):
        warnings.append(
            "jersey_ocr.mmocr_direct_recognition=true without number_roi_enabled=true may increase false positives"
        )


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
