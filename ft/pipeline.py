from pathlib import Path

from ft.calibration.pitch_transform import PitchTransform
from ft.config import load_config
from ft.export.artifacts import ArtifactExporter, write_json, write_table
from ft.features.groups import SemanticGroupAssigner
from ft.features.goalkeeper import GoalkeeperAppearanceAssigner, goalkeeper_color_ranges_by_team_from_roster
from ft.features.jersey_ocr import JerseyOCR
from ft.features.referee import RefereeAppearanceAssigner, referee_color_ranges_from_roster
from ft.features.roster_aware_ocr import RosterAwareOCRFilter
from ft.features.team import TeamAssigner, team_color_ranges_by_team_from_roster
from ft.features.visual import VisualFeatureExtractor
from ft.identity.candidates import build_identity_candidates, apply_identity_candidates, identity_candidate_rows
from ft.identity.constraints import enforce_identity_constraints
from ft.identity.hungarian import HungarianPlayerIdentifier, apply_assignments
from ft.identity.roster import load_roster, validate_unique_team_jersey
from ft.linking.jersey_identity_linker import JerseyIdentityLinker
from ft.linking.tracklet_linker import TrackletLinker
from ft.tracking.yolo_bytetrack import YoloByteTracker
from ft.tracking.yolo_strongsort import YoloStrongSortTracker
from ft.utils.run_diagnostics import RunDiagnostics
from ft.utils.video import read_video, save_video
from ft.utils.wandb_logger import WandbLogger
from ft.validation import validate_run_config
from ft.visualization.overlay import draw_overlay


def run_pipeline(config):
    """Run the full pipeline and let the W&B helper record success/failure.

    The implementation below deliberately keeps W&B outside the core logic so
    the same pipeline can run locally, on SSH, or in tests without changing the
    identity code.
    """
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
    if config.get("run", {}).get("validate_inputs", True):
        validate_run_config(config)
    run_diagnostics = RunDiagnostics(artifacts_dir, video_id)
    run_diagnostics.write_manifest(config)

    roster = load_roster(config.get("roster_path"))
    if config["identity"].get("enforce_unique_team_jersey", True):
        validate_unique_team_jersey(roster)

    print(f"FT video: {video_path}", flush=True)
    print(f"FT model: {model_path}", flush=True)
    with run_diagnostics.stage("read_video"):
        frames = read_video(video_path, max_frames=config.get("max_frames"))
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    print(f"FT video frames: {len(frames)}", flush=True)

    with run_diagnostics.stage("tracking"):
        tracker = build_tracker(config, model_path)
        tracks = tracker.run(frames)

    with run_diagnostics.stage("calibration"):
        calibrator = PitchTransform.from_config(config["calibration"], frames)
        calibrator.apply_tracks(tracks)
    print(f"FT calibration: {calibrator.source}", flush=True)

    # First colour pass: gives the linker early team/role context. These labels
    # are allowed to be imperfect because a second pass runs after display IDs
    # have been stabilized.
    team_assignments = {}
    if config["team"].get("enabled", True):
        with run_diagnostics.stage("team_first_pass"):
            team_cfg = team_config(config, roster)
            team_assignments = TeamAssigner(**team_cfg).fit_apply(frames, tracks)

    referee_diagnostics = {"enabled": False, "status": "disabled"}
    if config.get("referee", {}).get("enabled", True):
        with run_diagnostics.stage("referee_first_pass"):
            referee_cfg = referee_config(config, roster)
            referee_diagnostics = RefereeAppearanceAssigner(**referee_cfg).apply(frames, tracks)
        print(
            "FT referee colour:"
            f" referee_tracklets={len(referee_diagnostics.get('referees', {}))}"
            f" player_tracklets={len(referee_diagnostics.get('players', {}))}",
            flush=True,
        )

    if config["linking"].get("enabled", True):
        with run_diagnostics.stage("linking"):
            linking_cfg = {k: v for k, v in config["linking"].items() if k != "enabled"}
            linker = TrackletLinker(**linking_cfg)
            linker.apply(tracks, frames=frames)
            linking_diagnostics = linker.diagnostics
    else:
        TrackletLinker.ensure_display_ids(tracks)
        linking_diagnostics = {"enabled": False, "status": "disabled"}

    # Second colour pass: linking can merge raw IDs into a display_track_id, so
    # team/referee votes are recomputed on the final tracklet grouping.
    if config["team"].get("enabled", True):
        with run_diagnostics.stage("team_second_pass"):
            team_cfg = team_config(config, roster)
            team_assignments = TeamAssigner(**team_cfg).fit_apply(frames, tracks)

    if config.get("referee", {}).get("enabled", True):
        with run_diagnostics.stage("referee_second_pass"):
            referee_cfg = referee_config(config, roster)
            referee_diagnostics = RefereeAppearanceAssigner(**referee_cfg).apply(frames, tracks)

    goalkeeper_diagnostics = {"enabled": False, "status": "disabled"}
    if config.get("goalkeeper", {}).get("enabled", True):
        with run_diagnostics.stage("goalkeeper"):
            goalkeeper_cfg = goalkeeper_config(config, roster)
            goalkeeper_diagnostics = GoalkeeperAppearanceAssigner(**goalkeeper_cfg).apply(frames, tracks)
        print(
            "FT goalkeeper colour:"
            f" enabled={goalkeeper_diagnostics.get('enabled')}"
            f" tracklets={len(goalkeeper_diagnostics.get('tracklets', {}))}",
            flush=True,
        )

    with run_diagnostics.stage("semantic_groups"):
        semantic_groups = SemanticGroupAssigner().apply(tracks)

    # Export before identity so OCR and visual features operate on the exact
    # crops and per-frame metadata that will later be auditable in artifacts.
    exporter = ArtifactExporter(
        artifacts_dir,
        video_id,
        progress_every=config.get("progress", {}).get("artifact_rows", 5000),
        save_crops=config.get("export", {}).get("save_crops", True),
        deduplicate_crops=config.get("export", {}).get("deduplicate_crops", True),
    )
    with run_diagnostics.stage("export_pre_identity"):
        rows = exporter.export_tracklets(
            frames,
            tracks,
            stage="pre_identity",
            save_json=config.get("export", {}).get("save_pre_identity_json", False),
            save_csv=config.get("export", {}).get("save_pre_identity_csv", True),
        )
    player_rows = [row for row in rows if row.get("track_group", "players") == "players"]

    with run_diagnostics.stage("visual_features"):
        visual_extractor = VisualFeatureExtractor(cache_enabled=True)
        visual_extractor.add_row_features(player_rows)
    _copy_row_features_to_tracks(player_rows, tracks)

    if config["jersey_ocr"].get("enabled", False):
        with run_diagnostics.stage("jersey_ocr"):
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
                template_matching=config["jersey_ocr"].get("template_matching", False),
                template_font_image=config["jersey_ocr"].get("template_font_image"),
                template_min_score=config["jersey_ocr"].get("template_min_score", 0.62),
                template_weight=config["jersey_ocr"].get("template_weight", 0.03),
                template_max_candidates=config["jersey_ocr"].get("template_max_candidates", 4),
                aggregate_by_crop=config["jersey_ocr"].get("aggregate_by_crop", True),
                max_candidates_per_crop=config["jersey_ocr"].get("max_candidates_per_crop", 3),
                min_crop_candidate_ratio=config["jersey_ocr"].get("min_crop_candidate_ratio", 0.35),
                mmocr_device=config["jersey_ocr"].get("mmocr_device"),
                mmocr_det=config["jersey_ocr"].get("mmocr_det", "dbnet_resnet18_fpnc_1200e_icdar2015"),
                mmocr_rec=config["jersey_ocr"].get("mmocr_rec", "SAR"),
                mmocr_batch_size=config["jersey_ocr"].get("mmocr_batch_size", 8),
                mmocr_direct_recognition=config["jersey_ocr"].get("mmocr_direct_recognition"),
                progress_every=config["jersey_ocr"].get("progress_every", 5),
                cache_enabled=config["jersey_ocr"].get("cache_enabled", True),
                cache_dir=config["jersey_ocr"].get("cache_dir", ".ft_cache/ocr"),
                number_roi_enabled=config["jersey_ocr"].get("number_roi_enabled", False),
                number_roi_upscale=config["jersey_ocr"].get("number_roi_upscale", 3),
                number_roi_clahe=config["jersey_ocr"].get("number_roi_clahe", True),
                demote_direct_only_single_digits=config["jersey_ocr"].get(
                    "demote_direct_only_single_digits", True
                ),
                prefer_two_digit_candidates=config["jersey_ocr"].get("prefer_two_digit_candidates", True),
            )
            jersey_assignments, jersey_diagnostics = ocr.recognize(player_rows)
            if config["jersey_ocr"].get("roster_aware", True):
                # OCR is intentionally treated as a noisy proposal generator. The
                # roster filter removes impossible numbers before identity sees them.
                roster_filter = RosterAwareOCRFilter(
                    roster,
                    mode=config["jersey_ocr"].get("roster_filter_mode", "degrade"),
                    unknown_team_policy=config["jersey_ocr"].get("roster_unknown_team_policy", "keep"),
                    confidence_scale=config["jersey_ocr"].get("roster_degrade_confidence_scale", 0.60),
                    promote_roster_candidate=config["jersey_ocr"].get("promote_roster_candidate", True),
                    min_promoted_candidate_confidence=config["jersey_ocr"].get("min_promoted_candidate_confidence", 0.12),
                    min_promoted_candidate_votes=config["jersey_ocr"].get("min_promoted_candidate_votes", 1),
                )
                jersey_assignments, roster_filter_diagnostics = roster_filter.apply(jersey_assignments, player_rows)
                jersey_diagnostics["roster_filter"] = roster_filter_diagnostics
            jersey_assignments, goalkeeper_ocr_diagnostics = filter_goalkeeper_jersey_assignments(
                jersey_assignments,
                player_rows,
                apply_to_goalkeepers=config["jersey_ocr"].get("apply_to_goalkeepers", False),
            )
            jersey_diagnostics["goalkeeper_ocr_filter"] = goalkeeper_ocr_diagnostics
            _apply_jersey(jersey_assignments, player_rows, tracks)
            print(
                "FT jersey OCR:"
                f" status={jersey_diagnostics.get('status')}"
                f" backend={jersey_diagnostics.get('backend')}"
                f" template={jersey_diagnostics.get('template_matching', {}).get('status')}"
                f" assigned_tracklets={len(jersey_assignments)}",
                flush=True,
            )
        with run_diagnostics.stage("jersey_identity_linking"):
            jersey_linking_diagnostics = apply_jersey_identity_linking(config, tracks, player_rows)
    else:
        jersey_assignments = {}
        jersey_diagnostics = {"enabled": False, "status": "disabled"}
        jersey_linking_diagnostics = {"enabled": False, "status": "jersey_ocr_disabled"}
        print("FT jersey OCR: disabled", flush=True)

    with run_diagnostics.stage("identity_assignment"):
        identifier = HungarianPlayerIdentifier(
            roster_path=config.get("roster_path"),
            unknown_threshold=config["identity"]["unknown_threshold"],
            enforce_unique_team_jersey=config["identity"].get("enforce_unique_team_jersey", True),
            reliable_jersey_min_votes=config["identity"].get("reliable_jersey_min_votes", 2),
            reliable_jersey_min_confidence=config["identity"].get("reliable_jersey_min_confidence", 0.5),
            reliable_jersey_min_head_confidence=config["identity"].get("reliable_jersey_min_head_confidence", 0.60),
            reliable_jersey_min_winner_margin=config["identity"].get("reliable_jersey_min_winner_margin", 0.10),
            goalkeeper_number_one_prior=config["identity"].get("goalkeeper_number_one_prior", True),
            number_one_goalkeeper_bonus=config["identity"].get("number_one_goalkeeper_bonus", 0.08),
            number_one_non_goalkeeper_penalty=config["identity"].get("number_one_non_goalkeeper_penalty", 0.08),
            position_prior_max_cost=config["identity"].get("position_prior_max_cost", 0.08),
            position_prior_tiebreak_only=config["identity"].get("position_prior_tiebreak_only", True),
            require_assignment_evidence=config["identity"].get("require_assignment_evidence", True),
            reliable_jersey_min_candidate_score=config["identity"].get("reliable_jersey_min_candidate_score", 0.45),
            strong_evidence_min_team_confidence=config["identity"].get("strong_evidence_min_team_confidence", 0.75),
            strong_evidence_min_visual_similarity=config["identity"].get("strong_evidence_min_visual_similarity", 0.82),
            strong_evidence_min_tracklet_frames=config["identity"].get("strong_evidence_min_tracklet_frames", 45),
            strong_evidence_max_position_distance=config["identity"].get("strong_evidence_max_position_distance", 18.0),
            goalkeeper_singleton_gate=config["identity"].get("goalkeeper_singleton_gate", True),
            goalkeeper_singleton_min_team_confidence=config["identity"].get("goalkeeper_singleton_min_team_confidence", 0.75),
            goalkeeper_singleton_min_tracklet_frames=config["identity"].get("goalkeeper_singleton_min_tracklet_frames", 30),
        )
        summaries = identifier.summarize(player_rows)
        assignments, candidate_scores = identifier.assign(summaries)
        apply_assignments(tracks, assignments)
    # Hard constraints run after Hungarian because they can invalidate otherwise
    # plausible assignments when per-frame evidence exposes a contradiction.
    with run_diagnostics.stage("constraints"):
        constraints_diagnostics = enforce_identity_constraints(
            tracks,
            roster,
            frame_team_consistency=config["identity"].get("frame_team_consistency", True),
            frame_team_min_confidence=config["identity"].get("frame_team_min_confidence", 0.70),
            frame_team_split_enabled=config["identity"].get("frame_team_split_enabled", True),
            frame_team_split_min_frames=config["identity"].get("frame_team_split_min_frames", 8),
            frame_team_split_max_gap=config["identity"].get("frame_team_split_max_gap", 2),
            global_team_jersey_owner=config["identity"].get("global_team_jersey_owner", True),
        )

    candidate_cfg = config["identity"].get("candidate_fallback", {})
    with run_diagnostics.stage("identity_candidates"):
        identity_candidates = build_identity_candidates(
            candidate_scores,
            tracks=tracks,
            enabled=candidate_cfg.get("enabled", True),
            min_confidence=candidate_cfg.get("min_confidence", 0.35),
            max_cost=candidate_cfg.get("max_cost", 0.85),
            min_margin=candidate_cfg.get("min_margin", 0.0),
        )
        apply_identity_candidates(tracks, identity_candidates)

    with run_diagnostics.stage("export_final"):
        final_rows = exporter.export_tracklets(
            frames,
            tracks,
            stage="final",
            save_json=config.get("export", {}).get("save_final_json", True),
            save_csv=config.get("export", {}).get("save_final_csv", True),
        )
    write_json(
        {
            "calibration": calibrator.diagnostics(),
            "linking": linking_diagnostics,
            "team_assignments": {str(k): v for k, v in team_assignments.items()},
            "referee_colour": referee_diagnostics,
            "goalkeeper_colour": goalkeeper_diagnostics,
            "semantic_groups": semantic_groups,
            "jersey_ocr": jersey_diagnostics,
            "jersey_identity_linking": jersey_linking_diagnostics,
            "constraints": constraints_diagnostics,
            "tracklet_summaries": summaries,
            "assignments": {str(k): v for k, v in assignments.items()},
            "identity_candidates": {str(k): v for k, v in identity_candidates.items()},
        },
        artifacts_dir / "metadata" / f"{video_id}_identity_assignments.json",
    )
    write_json(linking_diagnostics, artifacts_dir / "metadata" / f"{video_id}_linking.json")
    write_json(calibrator.diagnostics(), artifacts_dir / "metadata" / f"{video_id}_calibration.json")
    write_json(constraints_diagnostics, artifacts_dir / "metadata" / f"{video_id}_constraints.json")
    write_json(referee_diagnostics, artifacts_dir / "metadata" / f"{video_id}_referee_colour.json")
    write_json(goalkeeper_diagnostics, artifacts_dir / "metadata" / f"{video_id}_goalkeeper_colour.json")
    write_json(jersey_linking_diagnostics, artifacts_dir / "metadata" / f"{video_id}_jersey_identity_linking.json")
    write_table(summaries, artifacts_dir / "metadata" / f"{video_id}_tracklet_summaries.csv")
    write_table(candidate_scores, artifacts_dir / "metadata" / f"{video_id}_candidate_scores.csv")
    write_table(
        identity_candidate_rows(identity_candidates),
        artifacts_dir / "metadata" / f"{video_id}_identity_candidates.csv",
    )
    write_json(
        {str(k): v for k, v in identity_candidates.items()},
        artifacts_dir / "metadata" / f"{video_id}_identity_candidates.json",
    )
    write_json(jersey_diagnostics, artifacts_dir / "metadata" / f"{video_id}_jersey_ocr.json")
    write_json(exporter.diagnostics(), artifacts_dir / "metadata" / f"{video_id}_export.json")
    write_json(visual_extractor.diagnostics(), artifacts_dir / "metadata" / f"{video_id}_visual_features.json")

    with run_diagnostics.stage("overlay"):
        output_frames = draw_overlay(frames, tracks, config=config.get("overlay", {}))
    with run_diagnostics.stage("save_video"):
        save_video(output_frames, output_path, fps=config["tracking"].get("frame_rate", 25))

    print(f"FT output video: {output_path}", flush=True)
    print(f"FT artifacts: {artifacts_dir}", flush=True)
    print(f"FT tracklet rows: {len(final_rows)}", flush=True)
    final_player_rows = [row for row in final_rows if row.get("track_group", "players") == "players"]
    assigned = [row for row in final_player_rows if row.get("player_id") not in (None, "unknown")]
    print(f"FT assigned row count: {len(assigned)}", flush=True)
    run_diagnostics.write_summary(
        {
            "rows": len(final_rows),
            "assigned_rows": len(assigned),
            "export": exporter.diagnostics(),
            "visual_features": visual_extractor.diagnostics(),
            "identity_candidates": len(identity_candidates),
        }
    )
    return {
        "output_path": output_path,
        "artifacts_dir": str(artifacts_dir),
        "video_id": video_id,
        "rows": final_rows,
        "summaries": summaries,
        "assignments": assignments,
        "candidate_scores": candidate_scores,
        "identity_candidates": identity_candidates,
    }


def run_from_config(path):
    return run_pipeline(load_config(path))


def team_config(config, roster):
    team_cfg = {k: v for k, v in config.get("team", {}).items() if k != "enabled"}
    roster_color_ranges = team_color_ranges_by_team_from_roster(roster)
    if roster_color_ranges:
        configured_ranges = team_cfg.get("color_ranges_by_team") or {}
        team_cfg["color_ranges_by_team"] = {**configured_ranges, **roster_color_ranges}
    return team_cfg


def referee_config(config, roster):
    referee_cfg = {k: v for k, v in config.get("referee", {}).items() if k != "enabled"}
    roster_color_ranges = referee_color_ranges_from_roster(roster)
    if roster_color_ranges:
        configured_ranges = referee_cfg.get("color_ranges") or {}
        referee_cfg["color_ranges"] = {**configured_ranges, **roster_color_ranges}
    return referee_cfg


def goalkeeper_config(config, roster):
    goalkeeper_cfg = {k: v for k, v in config.get("goalkeeper", {}).items() if k != "enabled"}
    roster_color_ranges = goalkeeper_color_ranges_by_team_from_roster(roster)
    if roster_color_ranges:
        configured_ranges = goalkeeper_cfg.get("color_ranges_by_team") or {}
        goalkeeper_cfg["color_ranges_by_team"] = {**configured_ranges, **roster_color_ranges}
    return goalkeeper_cfg


def apply_jersey_identity_linking(config, tracks, rows):
    cfg = config.get("jersey_identity_linking", {})
    if not cfg.get("enabled", True):
        return {"enabled": False, "status": "disabled"}
    linker_cfg = {key: value for key, value in cfg.items() if key != "enabled"}
    diagnostics = JerseyIdentityLinker(**linker_cfg).apply(tracks, rows=rows)
    print(
        "FT jersey identity linking:"
        f" links={len(diagnostics.get('accepted_links', []))}"
        f" changed_rows={diagnostics.get('changed_rows', 0)}",
        flush=True,
    )
    return diagnostics


def build_tracker(config, model_path):
    tracking_cfg = dict(config.get("tracking", {}))
    backend = str(tracking_cfg.pop("backend", "bytetrack")).lower().replace("_", "")
    strongsort_cfg = tracking_cfg.pop("strongsort", {}) or {}
    common = {
        "model_path": model_path,
        "detection_confidence": config["detection"]["confidence"],
        "ball_confidence": config["detection"]["ball_confidence"],
        "ball_max_area_ratio": config["detection"]["ball_max_area_ratio"],
        "ball_size_penalty": config["detection"]["ball_size_penalty"],
    }
    if backend in {"bytetrack", "byte"}:
        print("FT tracking backend: bytetrack", flush=True)
        return YoloByteTracker(**common, **tracking_cfg)
    if backend in {"strongsort", "strong"}:
        print("FT tracking backend: strongsort", flush=True)
        strongsort_args = dict(tracking_cfg)
        strongsort_args.update(strongsort_cfg)
        return YoloStrongSortTracker(**common, **strongsort_args)
    raise ValueError(f"Unknown tracking backend: {backend}")


def _copy_row_features_to_tracks(rows, tracks):
    by_key = {
        (int(row["frame"]), int(row["track_id"])): row
        for row in rows
        if row.get("track_group", "players") == "players"
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
        row["jersey_head_confidence"] = assignment.get("head_confidence")
        row["jersey_winner_margin"] = assignment.get("winner_margin")
        row["jersey_winner_score_ratio"] = assignment.get("winner_score_ratio")
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
            track["jersey_head_confidence"] = assignment.get("head_confidence")
            track["jersey_winner_margin"] = assignment.get("winner_margin")
            track["jersey_winner_score_ratio"] = assignment.get("winner_score_ratio")
            track["jersey_votes"] = assignment["votes"]
            track["jersey_evidence"] = assignment
            track["jersey_roster_filter"] = assignment.get("roster_filter")
            track["jersey_candidates"] = assignment.get("candidates")
            track["jersey_distribution"] = assignment.get("jersey_distribution")
            track["jersey_roster_mass"] = assignment.get("jersey_roster_mass", 0.0)


def filter_goalkeeper_jersey_assignments(jersey_assignments, rows, apply_to_goalkeepers=False):
    if apply_to_goalkeepers:
        return jersey_assignments, {
            "enabled": False,
            "apply_to_goalkeepers": True,
            "dropped": {},
        }
    goalkeeper_tracklets = goalkeeper_display_track_ids(rows)
    if not goalkeeper_tracklets:
        return jersey_assignments, {
            "enabled": True,
            "apply_to_goalkeepers": False,
            "dropped": {},
        }
    filtered = {}
    dropped = {}
    for track_id, assignment in sorted((jersey_assignments or {}).items()):
        if int(track_id) in goalkeeper_tracklets:
            dropped[str(track_id)] = {
                "reason": "goalkeeper_ocr_disabled",
                "jersey_number": assignment.get("jersey_number"),
                "confidence": assignment.get("confidence"),
                "votes": assignment.get("votes"),
            }
            continue
        filtered[track_id] = assignment
    return filtered, {
        "enabled": True,
        "apply_to_goalkeepers": False,
        "goalkeeper_tracklets": sorted(goalkeeper_tracklets),
        "dropped": dropped,
    }


def goalkeeper_display_track_ids(rows):
    role_votes = {}
    total_votes = {}
    for row in rows:
        display_id = int(row.get("display_track_id", row["track_id"]))
        total_votes[display_id] = total_votes.get(display_id, 0) + 1
        role = str(row.get("role_detection") or "").lower()
        if role in {"goalkeeper", "keeper", "gk"}:
            role_votes[display_id] = role_votes.get(display_id, 0) + 1
    return {
        track_id
        for track_id, count in role_votes.items()
        if count >= max(1, int(total_votes.get(track_id, 0) * 0.50))
    }
