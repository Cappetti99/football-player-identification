from pathlib import Path


DEFAULT_CONFIG = {
    "model_path": "models/best_yolo26x_gsr_light.pt",
    "video_path": "input_videos/08fd33_4.mp4",
    "output_path": "output_videos/ft_output.mp4",
    "artifacts_dir": "artifacts_ft/run",
    "roster_path": None,
    "max_frames": None,
    "detection": {
        "confidence": 0.05,
        "ball_confidence": 0.002,
        "ball_max_area_ratio": 0.0015,
        "ball_size_penalty": 0.5,
    },
    "tracking": {
        "track_activation_threshold": 0.10,
        "lost_track_buffer": 150,
        "minimum_matching_threshold": 0.95,
        "frame_rate": 25,
        "minimum_consecutive_frames": 2,
    },
    "linking": {
        "enabled": True,
        "max_gap": 90,
        "max_distance": 160.0,
        "min_frames": 4,
    },
    "calibration": {
        "enabled": True,
        "path": None,
        "auto": True,
        "auto_frame": 50,
        "auto_recalibrate_every": 0,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    },
    "team": {
        "enabled": True,
        "max_seed_frames": 12,
        "min_seed_colors": 8,
        "min_cluster_separation": 30.0,
        "min_classification_margin": 12.0,
        "min_tracklet_colors": 3,
    },
    "jersey_ocr": {
        "enabled": False,
        "backend": "auto",
        "min_confidence": 0.4,
        "max_crops_per_tracklet": 12,
        "debug_crops": False,
    },
    "identity": {
        "unknown_threshold": 0.55,
        "candidate_top_k": 8,
    },
    "wandb": {
        "enabled": False,
        "project": "football-tracking",
        "entity": None,
        "name": None,
        "tags": [],
        "notes": None,
        "log_artifacts": True,
        "log_video": False,
        "alert_on_finish": True,
        "alert_on_failure": True,
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

        with Path(path).open("r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
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
