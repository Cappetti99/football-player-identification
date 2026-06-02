from pathlib import Path


DEFAULT_CONFIG = {
    "model_path": "models/best_yolo26x_gsr_light.pt",
    "video_path": "input_videos/08fd33_4.mp4",
    "output_path": "output_videos/ft_output.mp4",
    "artifacts_dir": "artifacts_ft/run",
    "roster_path": None,
    "max_frames": None,
    "run": {
        "validate_inputs": True,
        "unknown_config_keys": "warn",
    },
    "detection": {
        "confidence": 0.05,
        "ball_confidence": 0.002,
        "ball_max_area_ratio": 0.0015,
        "ball_size_penalty": 0.5,
    },
    "tracking": {
        "backend": "bytetrack",
        "track_activation_threshold": 0.10,
        "lost_track_buffer": 150,
        "minimum_matching_threshold": 0.95,
        "frame_rate": 25,
        "minimum_consecutive_frames": 2,
        "progress_every": 250,
        "strongsort": {
            "max_age": 150,
            "min_hits": 2,
            "max_center_distance": 180.0,
            "max_cost": 0.78,
            "iou_weight": 0.45,
            "appearance_weight": 0.35,
            "center_weight": 0.20,
            "appearance_min_similarity": 0.15,
            "appearance_ema": 0.85,
            "class_gate": True,
            "progress_every": 250,
        },
    },
    "linking": {
        "enabled": True,
        "max_gap": 90,
        "max_distance": 160.0,
        "min_frames": 4,
        "team_gate_enabled": True,
        "team_gate_min_confidence": 0.65,
        "appearance_gate_enabled": True,
        "appearance_min_similarity": 0.72,
        "max_rejection_records": 5000,
    },
    "jersey_identity_linking": {
        "enabled": True,
        "max_gap": 90,
        "max_distance": 140.0,
        "min_frames": 3,
        "min_confidence": 0.20,
        "min_head_confidence": 0.55,
        "min_winner_margin": 0.10,
        "min_votes": 5,
        "team_gate_min_confidence": 0.60,
        "max_rejection_records": 5000,
    },
    "calibration": {
        "enabled": True,
        "path": None,
        "auto": True,
        "auto_frame": 50,
        "auto_recalibrate_every": 0,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "tvcalib": {
            "enabled": False,
            "path": None,
            "per_frame": True,
            "coordinate_system": "tvcalib_centered",
            "invert": False,
            "temporal_index": 0,
            "frame_offset": 0,
            "nearest_frame": True,
            "max_frame_gap": None,
            "image_id": None,
            "frame": None,
        },
    },
    "team": {
        "enabled": True,
        "max_seed_frames": 12,
        "min_seed_colors": 8,
        "min_cluster_separation": 30.0,
        "min_classification_margin": 12.0,
        "min_tracklet_colors": 3,
        "roster_color_min_fraction": 0.16,
        "roster_color_min_margin": 0.04,
        "prefer_roster_palette": True,
        "trusted_palette_min_fraction": 0.18,
        "trusted_palette_min_margin": 0.04,
        "trusted_palette_min_samples": 3,
    },
    "jersey_ocr": {
        "enabled": False,
        "backend": "auto",
        "min_confidence": 0.4,
        "max_crops_per_tracklet": 12,
        "temporal_passes": 1,
        "augment": True,
        "min_crop_quality": 0.08,
        "min_votes": 2,
        "min_raw_confidence": 0.05,
        "min_winner_margin": 0.15,
        "easyocr_gpu": False,
        "debug_crops": False,
        "template_matching": False,
        "template_font_image": None,
        "template_min_score": 0.62,
        "template_weight": 0.03,
        "template_max_candidates": 4,
        "aggregate_by_crop": True,
        "max_candidates_per_crop": 3,
        "min_crop_candidate_ratio": 0.35,
        "mmocr_device": None,
        "mmocr_det": "dbnet_resnet18_fpnc_1200e_icdar2015",
        "mmocr_rec": "SAR",
        "mmocr_batch_size": 8,
        "mmocr_direct_recognition": None,
        "progress_every": 5,
        "cache_enabled": True,
        "cache_dir": ".ft_cache/ocr",
        "number_roi_enabled": False,
        "number_roi_upscale": 3,
        "number_roi_clahe": True,
        "demote_direct_only_single_digits": True,
        "prefer_two_digit_candidates": True,
        "apply_to_goalkeepers": False,
        "roster_aware": True,
        "roster_filter_mode": "degrade",
        "roster_unknown_team_policy": "keep",
        "roster_degrade_confidence_scale": 0.60,
        "promote_roster_candidate": True,
        "min_promoted_candidate_confidence": 0.12,
        "min_promoted_candidate_votes": 1,
    },
    "referee": {
        "enabled": True,
        "min_color_fraction": 0.28,
        "min_tracklet_frames": 10,
        "reclassify_player_candidates": True,
        "player_candidate_min_color_fraction": 0.28,
        "player_candidate_max_team_confidence": 0.50,
        "require_palette_color": True,
        "trusted_color_min_fraction": 0.28,
        "trusted_color_override_team_confidence": True,
    },
    "goalkeeper": {
        "enabled": True,
        "min_color_fraction": 0.20,
        "min_tracklet_frames": 2,
        "assign_team_from_color": False,
        "team_correction_min_score": 0.55,
    },
    "identity": {
        "unknown_threshold": 0.55,
        "candidate_top_k": 8,
        "enforce_unique_team_jersey": True,
        "reliable_jersey_min_votes": 5,
        "reliable_jersey_min_confidence": 0.20,
        "reliable_jersey_min_head_confidence": 0.55,
        "reliable_jersey_min_winner_margin": 0.10,
        "position_prior_max_cost": 0.08,
        "position_prior_tiebreak_only": True,
        "require_assignment_evidence": True,
        "reliable_jersey_min_candidate_score": 0.45,
        "strong_evidence_min_team_confidence": 0.75,
        "strong_evidence_min_visual_similarity": 0.82,
        "strong_evidence_min_tracklet_frames": 45,
        "strong_evidence_max_position_distance": 18.0,
        "frame_team_consistency": True,
        "frame_team_min_confidence": 0.70,
        "frame_team_split_enabled": True,
        "frame_team_split_min_frames": 8,
        "frame_team_split_max_gap": 4,
        "global_team_jersey_owner": True,
        "goalkeeper_number_one_prior": True,
        "number_one_goalkeeper_bonus": 0.08,
        "number_one_non_goalkeeper_penalty": 0.08,
        "candidate_fallback": {
            "enabled": True,
            "min_confidence": 0.35,
            "max_cost": 0.85,
            "min_margin": 0.0,
        },
    },
    "overlay": {
        "show_display_id": True,
        "show_jersey": True,
        "show_jersey_winner": False,
        "show_jersey_min_confidence": 0.30,
        "show_jersey_min_votes": 3,
        "show_jersey_min_stable_votes": 20,
        "show_jersey_min_winner_margin": 0.10,
        "show_jersey_min_head_votes": 3,
        "show_jersey_min_head_confidence": 0.55,
        "require_ocr_jersey_evidence": True,
        "show_player_id": False,
        "show_player_id_min_confidence": 0.80,
        "show_identity_confidence": False,
    },
    "wandb": {
        "enabled": False,
        "project": "football-tracking",
        "entity": None,
        "name": None,
        "init_timeout": 180,
        "tags": [],
        "notes": None,
        "log_artifacts": True,
        "log_video": False,
        "alert_on_finish": True,
        "alert_on_failure": True,
    },
    "progress": {
        "artifact_rows": 5000,
    },
    "export": {
        "save_crops": True,
        "deduplicate_crops": True,
        "save_pre_identity_json": False,
        "save_pre_identity_csv": True,
        "save_final_json": True,
        "save_final_csv": True,
    },
}


def deep_merge(base, override):
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path:
        import yaml

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        base_config = user_config.pop("base_config", None)
        if base_config:
            base_path = Path(base_config)
            if not base_path.is_absolute():
                base_path = path.parent / base_path
            config = load_config(base_path)
        config = deep_merge(config, user_config)
    return config


def apply_overrides(config, overrides):
    mapping = {
        "video_path": ("video_path",),
        "model_path": ("model_path",),
        "output_path": ("output_path",),
        "artifacts_dir": ("artifacts_dir",),
        "roster_path": ("roster_path",),
        "max_frames": ("max_frames",),
        "wandb_project": ("wandb", "project"),
        "wandb_name": ("wandb", "name"),
    }
    for attr, path in mapping.items():
        value = getattr(overrides, attr, None)
        if value is None:
            continue
        target = config
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    if getattr(overrides, "wandb", False):
        config.setdefault("wandb", {})["enabled"] = True
    return config
