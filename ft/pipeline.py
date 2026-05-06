from pathlib import Path

from ft.calibration.pitch_transform import PitchTransform
from ft.config import load_config
from ft.export.artifacts import ArtifactExporter, write_json, write_table
from ft.features.groups import SemanticGroupAssigner
from ft.features.jersey_ocr import JerseyOCR
from ft.features.referee import RefereeAppearanceAssigner
from ft.features.roster_aware_ocr import RosterAwareOCRFilter
from ft.features.team import TeamAssigner
from ft.features.visual import VisualFeatureExtractor
from ft.identity.hungarian import HungarianPlayerIdentifier, apply_assignments
from ft.identity.roster import load_roster, validate_unique_team_jersey
from ft.linking.tracklet_linker import TrackletLinker
from ft.tracking.yolo_bytetrack import YoloByteTracker
from ft.utils.video import read_video, save_video
from ft.utils.wandb_logger import WandbLogger
from ft.visualization.overlay import draw_overlay


def run_pipeline(config):
    video_id = Path(config["video_path"]).stem
    wandb_logger = WandbLogger.from_config(config, video_id)
    try:
        result = _run_pipeline_impl(config)
    except Exception as exc:
        wandb_logger.log_failure(exc)
        wandb_logger.finish()
        raise
    wandb_logger.log_success(result)
    wandb_logger.finish()
    return result


def _run_pipeline_impl(config):
    video_path = config["video_path"]
    model_path = config["model_path"]
    output_path = config["output_path"]
    artifacts_dir = Path(config["artifacts_dir"])
    video_id = Path(video_path).stem
    roster = load_roster(config.get("roster_path"))
    if config["identity"].get("enforce_unique_team_jersey", True):
        validate_unique_team_jersey(roster)

    print(f"FT video: {video_path}")
    print(f"FT model: {model_path}")
    frames = read_video(video_path, max_frames=config.get("max_frames"))
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    tracker = YoloByteTracker(
        model_path=model_path,
        detection_confidence=config["detection"]["confidence"],
        ball_confidence=config["detection"]["ball_confidence"],
        ball_max_area_ratio=config["detection"]["ball_max_area_ratio"],
        ball_size_penalty=config["detection"]["ball_size_penalty"],
        **config["tracking"],
    )
    tracks = tracker.run(frames)

    if config["linking"].get("enabled", True):
        TrackletLinker(
            max_gap=config["linking"]["max_gap"],
            max_distance=config["linking"]["max_distance"],
            min_frames=config["linking"]["min_frames"],
        ).apply(tracks)
    else:
        TrackletLinker.ensure_display_ids(tracks)

    calibrator = PitchTransform.from_config(config["calibration"], frames)
    calibrator.apply_tracks(tracks)
    print(f"FT calibration: {calibrator.source}")

    if config["team"].get("enabled", True):
        team_cfg = {k: v for k, v in config["team"].items() if k != "enabled"}
        team_assignments = TeamAssigner(**team_cfg).fit_apply(frames, tracks)
    else:
        team_assignments = {}

    if config.get("referee", {}).get("enabled", True):
        referee_cfg = {k: v for k, v in config["referee"].items() if k != "enabled"}
        referee_diagnostics = RefereeAppearanceAssigner(**referee_cfg).apply(frames, tracks)
        print(
            "FT referee colour:"
            f" referee_tracklets={len(referee_diagnostics.get('referees', {}))}"
            f" player_tracklets={len(referee_diagnostics.get('players', {}))}"
        )
    else:
        referee_diagnostics = {"enabled": False, "status": "disabled"}

    semantic_groups = SemanticGroupAssigner().apply(tracks)

    exporter = ArtifactExporter(artifacts_dir, video_id)
    rows = exporter.export_tracklets(frames, tracks)

    VisualFeatureExtractor().add_row_features(rows)
    _copy_row_features_to_tracks(rows, tracks)

    if config["jersey_ocr"].get("enabled", False):
        debug_dir = (
            artifacts_dir / "jersey_ocr_debug" / video_id
            if config["jersey_ocr"].get("debug_crops", False)
            else None
        )
        ocr = JerseyOCR(
            backend=config["jersey_ocr"]["backend"],
            min_confidence=config["jersey_ocr"]["min_confidence"],
            max_crops_per_tracklet=config["jersey_ocr"]["max_crops_per_tracklet"],
            temporal_passes=config["jersey_ocr"].get("temporal_passes", 1),
            augment=config["jersey_ocr"].get("augment", True),
            min_crop_quality=config["jersey_ocr"].get("min_crop_quality", 0.08),
            min_votes=config["jersey_ocr"].get("min_votes", 2),
            min_raw_confidence=config["jersey_ocr"].get("min_raw_confidence", 0.05),
            min_winner_margin=config["jersey_ocr"].get("min_winner_margin", 0.15),
            easyocr_gpu=config["jersey_ocr"].get("easyocr_gpu", False),
            debug_dir=debug_dir,
        )
        jersey_assignments, jersey_diagnostics = ocr.recognize(rows)
        if config["jersey_ocr"].get("roster_aware", True):
            roster_filter = RosterAwareOCRFilter(
                roster,
                mode=config["jersey_ocr"].get("roster_filter_mode", "degrade"),
                unknown_team_policy=config["jersey_ocr"].get("roster_unknown_team_policy", "keep"),
                confidence_scale=config["jersey_ocr"].get("roster_degrade_confidence_scale", 0.60),
                promote_roster_candidate=config["jersey_ocr"].get("promote_roster_candidate", True),
                min_promoted_candidate_confidence=config["jersey_ocr"].get("min_promoted_candidate_confidence", 0.12),
                min_promoted_candidate_votes=config["jersey_ocr"].get("min_promoted_candidate_votes", 1),
            )
            jersey_assignments, roster_filter_diagnostics = roster_filter.apply(jersey_assignments, rows)
            jersey_diagnostics["roster_filter"] = roster_filter_diagnostics
        _apply_jersey(jersey_assignments, rows, tracks)
        print(
            "FT jersey OCR:"
            f" status={jersey_diagnostics.get('status')}"
            f" backend={jersey_diagnostics.get('backend')}"
            f" assigned_tracklets={len(jersey_assignments)}"
        )
    else:
        jersey_assignments = {}
        jersey_diagnostics = {"enabled": False, "status": "disabled"}
        print("FT jersey OCR: disabled")

    identifier = HungarianPlayerIdentifier(
        roster_path=config.get("roster_path"),
        unknown_threshold=config["identity"]["unknown_threshold"],
        enforce_unique_team_jersey=config["identity"].get("enforce_unique_team_jersey", True),
        reliable_jersey_min_votes=config["identity"].get("reliable_jersey_min_votes", 2),
        reliable_jersey_min_confidence=config["identity"].get("reliable_jersey_min_confidence", 0.5),
        goalkeeper_number_one_prior=config["identity"].get("goalkeeper_number_one_prior", True),
        number_one_goalkeeper_bonus=config["identity"].get("number_one_goalkeeper_bonus", 0.08),
        number_one_non_goalkeeper_penalty=config["identity"].get("number_one_non_goalkeeper_penalty", 0.08),
    )
    summaries = identifier.summarize(rows)
    assignments, candidate_scores = identifier.assign(summaries)
    apply_assignments(tracks, assignments)

    final_rows = exporter.export_tracklets(frames, tracks)
    write_json(
        {
            "calibration": {
                "enabled": calibrator.enabled,
                "source": calibrator.source,
                "points": calibrator.calibration_points,
            },
            "team_assignments": {str(k): v for k, v in team_assignments.items()},
            "referee_colour": referee_diagnostics,
            "semantic_groups": semantic_groups,
            "jersey_ocr": jersey_diagnostics,
            "tracklet_summaries": summaries,
            "assignments": {str(k): v for k, v in assignments.items()},
        },
        artifacts_dir / "metadata" / f"{video_id}_identity_assignments.json",
    )
    write_json(referee_diagnostics, artifacts_dir / "metadata" / f"{video_id}_referee_colour.json")
    write_table(summaries, artifacts_dir / "metadata" / f"{video_id}_tracklet_summaries.csv")
    write_table(candidate_scores, artifacts_dir / "metadata" / f"{video_id}_candidate_scores.csv")
    write_json(jersey_diagnostics, artifacts_dir / "metadata" / f"{video_id}_jersey_ocr.json")

    output_frames = draw_overlay(frames, tracks)
    save_video(output_frames, output_path, fps=config["tracking"].get("frame_rate", 25))

    print(f"FT output video: {output_path}")
    print(f"FT artifacts: {artifacts_dir}")
    print(f"FT tracklet rows: {len(final_rows)}")
    assigned = [row for row in final_rows if row.get("player_id") not in (None, "unknown")]
    print(f"FT assigned row count: {len(assigned)}")
    return {
        "output_path": output_path,
        "artifacts_dir": str(artifacts_dir),
        "video_id": video_id,
        "rows": final_rows,
        "summaries": summaries,
        "assignments": assignments,
        "candidate_scores": candidate_scores,
    }


def run_from_config(path):
    return run_pipeline(load_config(path))


def _copy_row_features_to_tracks(rows, tracks):
    by_key = {
        (int(row["frame"]), int(row["track_id"])): row
        for row in rows
    }
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        for raw_id, track in frame_tracks.items():
            row = by_key.get((frame_num, int(raw_id)))
            if row:
                track["visual_embedding"] = row.get("visual_embedding")


def _apply_jersey(jersey_assignments, rows, tracks):
    for row in rows:
        display_id = int(row.get("display_track_id", row["track_id"]))
        assignment = jersey_assignments.get(display_id)
        if not assignment:
            continue
        row["jersey_number"] = assignment["jersey_number"]
        row["jersey_confidence"] = assignment["confidence"]
        row["jersey_votes"] = assignment["votes"]
        row["jersey_roster_filter"] = assignment.get("roster_filter")
        row["jersey_candidates"] = assignment.get("candidates")
        row["jersey_distribution"] = assignment.get("jersey_distribution")
        row["jersey_roster_mass"] = assignment.get("jersey_roster_mass", 0.0)
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            assignment = jersey_assignments.get(display_id)
            if not assignment:
                continue
            track["jersey_number"] = assignment["jersey_number"]
            track["jersey_confidence"] = assignment["confidence"]
            track["jersey_votes"] = assignment["votes"]
            track["jersey_evidence"] = assignment
            track["jersey_roster_filter"] = assignment.get("roster_filter")
            track["jersey_candidates"] = assignment.get("candidates")
            track["jersey_distribution"] = assignment.get("jersey_distribution")
            track["jersey_roster_mass"] = assignment.get("jersey_roster_mass", 0.0)
