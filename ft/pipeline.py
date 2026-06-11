from collections import Counter
from pathlib import Path

import numpy as np

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
from ft.identity.propagation import IdentityPropagationEngine, propagation_rows
from ft.identity.roster import load_roster, validate_unique_team_jersey
from ft.linking.jersey_identity_linker import JerseyIdentityLinker
from ft.linking.tracklet_linker import TrackletLinker
from ft.tracking.yolo_bytetrack import YoloByteTracker
from ft.tracking.yolo_strongsort import YoloStrongSortTracker
from ft.utils.run_diagnostics import RunDiagnostics
from ft.utils.geometry import bbox_center
from ft.utils.scene_cuts import (
    annotate_tracks_with_scene_segments,
    detect_scene_cuts,
    scene_cut_rows,
)
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

    scene_cut_diagnostics = {"enabled": False, "status": "disabled", "cuts": [], "cut_frames": [], "segments": []}
    scene_cut_frames = []
    if config.get("scene_cuts", {}).get("enabled", False):
        with run_diagnostics.stage("scene_cuts"):
            scene_cut_diagnostics = detect_scene_cuts(frames, **scene_cut_config(config))
            scene_cut_frames = [int(frame) for frame in scene_cut_diagnostics.get("cut_frames", [])]
        print(
            "FT scene cuts:"
            f" cuts={len(scene_cut_frames)}"
            f" reset_tracking={config.get('scene_cuts', {}).get('tracking_reset_enabled', False)}",
            flush=True,
        )

    with run_diagnostics.stage("tracking"):
        tracker = build_tracker(config, model_path)
        tracks = run_tracker_with_scene_cuts(
            tracker,
            frames,
            scene_cut_frames=scene_cut_frames,
            reset_enabled=config.get("scene_cuts", {}).get("tracking_reset_enabled", False),
        )
    annotate_tracks_with_scene_segments(
        tracks,
        scene_cut_frames,
        scene_cut_lookup={int(row["frame"]): row for row in scene_cut_diagnostics.get("cuts", [])},
    )

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
        visual_cfg = config.get("visual", {})
        visual_extractor = VisualFeatureExtractor(
            cache_enabled=visual_cfg.get("cache_enabled", True),
            embedding_mode=visual_cfg.get("embedding_mode", "hsv"),
        )
        visual_extractor.add_row_features(player_rows)
    _copy_row_features_to_tracks(player_rows, tracks)

    segment_jersey_candidates = []
    segment_jersey_diagnostics = {"enabled": False, "status": "disabled"}
    if config["jersey_ocr"].get("enabled", False):
        with run_diagnostics.stage("jersey_ocr"):
            ocr = build_jersey_ocr(
                config,
                artifacts_dir,
                video_id,
                debug_subdir=video_id,
                segment_frames=config["jersey_ocr"].get("segment_frames", 0),
            )
            jersey_assignments, jersey_diagnostics = ocr.recognize(player_rows)
            for assignment in (jersey_assignments or {}).values():
                if assignment.get("raw_jersey_distribution") is None:
                    assignment["raw_jersey_distribution"] = assignment.get("candidates")
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
                    preserve_dropped_evidence=config["jersey_ocr"].get("roster_preserve_dropped_evidence", False),
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
        segment_candidate_frames = int(config["jersey_ocr"].get("segment_candidate_frames") or 0)
        if segment_candidate_frames > 0:
            with run_diagnostics.stage("segment_jersey_candidates"):
                segment_ocr = build_jersey_ocr(
                    config,
                    artifacts_dir,
                    video_id,
                    debug_subdir=f"{video_id}_segment_candidates",
                    segment_frames=segment_candidate_frames,
                )
                segment_assignments, segment_jersey_diagnostics = segment_ocr.recognize(player_rows)
                for assignment in (segment_assignments or {}).values():
                    if assignment.get("raw_jersey_distribution") is None:
                        assignment["raw_jersey_distribution"] = assignment.get("candidates")
                if config["jersey_ocr"].get("roster_aware", True):
                    roster_filter = RosterAwareOCRFilter(
                        roster,
                        mode=config["jersey_ocr"].get("roster_filter_mode", "degrade"),
                        unknown_team_policy=config["jersey_ocr"].get("roster_unknown_team_policy", "keep"),
                        confidence_scale=config["jersey_ocr"].get("roster_degrade_confidence_scale", 0.60),
                        promote_roster_candidate=config["jersey_ocr"].get("promote_roster_candidate", True),
                        min_promoted_candidate_confidence=config["jersey_ocr"].get("min_promoted_candidate_confidence", 0.12),
                        min_promoted_candidate_votes=config["jersey_ocr"].get("min_promoted_candidate_votes", 1),
                        preserve_dropped_evidence=config["jersey_ocr"].get("roster_preserve_dropped_evidence", False),
                    )
                    segment_assignments, roster_filter_diagnostics = roster_filter.apply(segment_assignments, player_rows)
                    segment_jersey_diagnostics["roster_filter"] = roster_filter_diagnostics
                segment_assignments, goalkeeper_ocr_diagnostics = filter_goalkeeper_jersey_assignments(
                    segment_assignments,
                    player_rows,
                    apply_to_goalkeepers=config["jersey_ocr"].get("apply_to_goalkeepers", False),
                )
                segment_jersey_diagnostics["goalkeeper_ocr_filter"] = goalkeeper_ocr_diagnostics
                segment_jersey_candidates = segment_jersey_candidate_rows(
                    segment_assignments,
                    player_rows,
                    roster,
                )
                segment_jersey_diagnostics["candidate_rows"] = len(segment_jersey_candidates)
                print(
                    "FT segment jersey candidates:"
                    f" frames={segment_candidate_frames}"
                    f" candidates={len(segment_jersey_candidates)}",
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

    identity_propagation_diagnostics = {"enabled": False, "status": "disabled"}
    propagation_cfg = config.get("identity_propagation", {})
    if propagation_cfg.get("enabled", False) and roster:
        with run_diagnostics.stage("identity_propagation"):
            propagator = IdentityPropagationEngine(
                roster=roster,
                min_composite_score=propagation_cfg.get("min_composite_score", 0.40),
                min_score_margin=propagation_cfg.get("min_score_margin", 0.08),
                max_hops=propagation_cfg.get("max_hops", 1),
                allow_propagated_sources=propagation_cfg.get("allow_propagated_sources", False),
                min_source_confidence=propagation_cfg.get("min_source_confidence", 0.55),
                conflict_buffer=propagation_cfg.get("conflict_buffer", 0),
                allow_partial_conflict_frames=propagation_cfg.get("allow_partial_conflict_frames", False),
                min_partial_frames=propagation_cfg.get("min_partial_frames", 20),
                min_partial_fraction=propagation_cfg.get("min_partial_fraction", 0.25),
                propagate_goalkeepers=propagation_cfg.get("propagate_goalkeepers", True),
                max_temporal_gap=propagation_cfg.get("max_temporal_gap", 300),
                max_spatial_distance=propagation_cfg.get("max_spatial_distance", 25.0),
                min_team_confidence=propagation_cfg.get("min_team_confidence", 0.50),
                min_appearance_similarity=propagation_cfg.get("min_appearance_similarity", 0.50),
                require_team_match=propagation_cfg.get("require_team_match", True),
                require_jersey_match=propagation_cfg.get("require_jersey_match", False),
                block_goalkeeper_mismatch=propagation_cfg.get("block_goalkeeper_mismatch", True),
                require_jersey_or_strong_appearance=propagation_cfg.get("require_jersey_or_strong_appearance", True),
                strong_appearance_similarity=propagation_cfg.get("strong_appearance_similarity", 0.72),
                allow_temporal_overlap=propagation_cfg.get("allow_temporal_overlap", False),
                temporal_overlap_score=propagation_cfg.get("temporal_overlap_score", 0.10),
                scene_cut_frames=scene_cut_frames,
                cut_bridge_enabled=propagation_cfg.get("cut_bridge_enabled", False),
                cut_bridge_max_gap=propagation_cfg.get("cut_bridge_max_gap", 5),
                cut_bridge_min_jersey_confidence=propagation_cfg.get("cut_bridge_min_jersey_confidence", 0.20),
                cut_bridge_min_jersey_votes=propagation_cfg.get("cut_bridge_min_jersey_votes", 3),
            )
            identity_propagation_diagnostics = propagator.apply(
                tracks=tracks,
                summaries=summaries,
                assignments=assignments,
            )
        print(
            "FT identity propagation:"
            f" propagated={identity_propagation_diagnostics.get('total_propagated', 0)}"
            f" hops={identity_propagation_diagnostics.get('hops', {})}",
            flush=True,
        )
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
            goalkeeper_only_alternate_enabled=config["identity"].get("goalkeeper_only_alternate_enabled", False),
            goalkeeper_only_alternate_min_confidence=config["identity"].get("goalkeeper_only_alternate_min_confidence", 0.10),
            goalkeeper_only_alternate_min_votes=config["identity"].get("goalkeeper_only_alternate_min_votes", 1),
            goalkeeper_only_alternate_max_rank=config["identity"].get("goalkeeper_only_alternate_max_rank", 5),
            goalkeeper_only_alternate_block_known_owner=config["identity"].get("goalkeeper_only_alternate_block_known_owner", True),
            goalkeeper_only_alternate_stop_on_known_owner_conflict=config["identity"].get(
                "goalkeeper_only_alternate_stop_on_known_owner_conflict", True
            ),
        )

    candidate_cfg = config["identity"].get("candidate_fallback", {})
    segment_candidate_diagnostics = {"enabled": False, "status": "disabled"}
    with run_diagnostics.stage("identity_candidates"):
        identity_candidates = build_identity_candidates(
            candidate_scores,
            tracks=tracks,
            enabled=candidate_cfg.get("enabled", True),
            min_confidence=candidate_cfg.get("min_confidence", 0.35),
            max_cost=candidate_cfg.get("max_cost", 0.85),
            min_margin=candidate_cfg.get("min_margin", 0.0),
            max_jersey_display_spread=candidate_cfg.get("max_jersey_display_spread"),
            display_spread_scope=candidate_cfg.get("display_spread_scope", "team"),
            display_spread_only_unknown=candidate_cfg.get("display_spread_only_unknown", True),
        )
        apply_identity_candidates(tracks, identity_candidates)
        segment_candidate_diagnostics = apply_segment_jersey_identity_candidates(
            tracks,
            segment_jersey_candidates,
            enabled=candidate_cfg.get("segment_enabled", False),
            min_confidence=candidate_cfg.get("segment_min_confidence", 0.20),
            min_votes=candidate_cfg.get("segment_min_votes", 8),
            max_jersey_display_spread=candidate_cfg.get("segment_max_jersey_display_spread", 3),
            require_unique_roster_player=candidate_cfg.get("segment_require_unique_roster_player", True),
            only_unknown_identity=candidate_cfg.get("segment_only_unknown_identity", True),
        )

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
            "scene_cuts": scene_cut_diagnostics,
            "linking": linking_diagnostics,
            "team_assignments": {str(k): v for k, v in team_assignments.items()},
            "referee_colour": referee_diagnostics,
            "goalkeeper_colour": goalkeeper_diagnostics,
            "semantic_groups": semantic_groups,
            "jersey_ocr": jersey_diagnostics,
            "jersey_identity_linking": jersey_linking_diagnostics,
            "constraints": constraints_diagnostics,
            "identity_propagation": identity_propagation_diagnostics,
            "tracklet_summaries": summaries,
            "assignments": {str(k): v for k, v in assignments.items()},
            "identity_candidates": {str(k): v for k, v in identity_candidates.items()},
            "segment_jersey_candidates": segment_jersey_candidates,
            "segment_candidate_fallback": segment_candidate_diagnostics,
        },
        artifacts_dir / "metadata" / f"{video_id}_identity_assignments.json",
    )
    write_json(linking_diagnostics, artifacts_dir / "metadata" / f"{video_id}_linking.json")
    write_json(scene_cut_diagnostics, artifacts_dir / "metadata" / f"{video_id}_scene_cuts.json")
    write_table(scene_cut_rows(scene_cut_diagnostics), artifacts_dir / "metadata" / f"{video_id}_scene_cuts.csv")
    write_json(ball_tracking_diagnostics(tracks), artifacts_dir / "metadata" / f"{video_id}_ball_tracking.json")
    write_json(calibrator.diagnostics(), artifacts_dir / "metadata" / f"{video_id}_calibration.json")
    write_json(identity_propagation_diagnostics, artifacts_dir / "metadata" / f"{video_id}_identity_propagation.json")
    write_table(
        propagation_rows(identity_propagation_diagnostics),
        artifacts_dir / "metadata" / f"{video_id}_identity_propagation.csv",
    )
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
    write_table(
        segment_jersey_candidates,
        artifacts_dir / "metadata" / f"{video_id}_segment_jersey_candidates.csv",
    )
    write_json(
        {
            "diagnostics": segment_jersey_diagnostics,
            "candidates": segment_jersey_candidates,
        },
        artifacts_dir / "metadata" / f"{video_id}_segment_jersey_candidates.json",
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
            "scene_cuts": scene_cut_diagnostics,
            "visual_features": visual_extractor.diagnostics(),
            "identity_propagation": identity_propagation_diagnostics,
            "identity_candidates": len(identity_candidates),
            "segment_jersey_candidates": len(segment_jersey_candidates),
            "segment_candidate_fallback": segment_candidate_diagnostics,
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
        "scene_cuts": scene_cut_diagnostics,
        "identity_propagation": identity_propagation_diagnostics,
        "identity_candidates": identity_candidates,
        "segment_jersey_candidates": segment_jersey_candidates,
        "segment_candidate_fallback": segment_candidate_diagnostics,
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


def scene_cut_config(config):
    return {
        key: value
        for key, value in config.get("scene_cuts", {}).items()
        if key != "tracking_reset_enabled"
    }


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


def build_jersey_ocr(config, artifacts_dir, video_id, debug_subdir=None, segment_frames=None):
    jersey_cfg = config["jersey_ocr"]
    debug_dir = (
        Path(artifacts_dir) / "jersey_ocr_debug" / (debug_subdir or video_id)
        if jersey_cfg.get("debug_crops", False)
        else None
    )
    return JerseyOCR(
        backend=jersey_cfg["backend"],
        min_confidence=jersey_cfg["min_confidence"],
        max_crops_per_tracklet=jersey_cfg["max_crops_per_tracklet"],
        temporal_passes=jersey_cfg.get("temporal_passes", 1),
        augment=jersey_cfg.get("augment", True),
        min_crop_quality=jersey_cfg.get("min_crop_quality", 0.08),
        min_votes=jersey_cfg.get("min_votes", 2),
        min_raw_confidence=jersey_cfg.get("min_raw_confidence", 0.05),
        min_winner_margin=jersey_cfg.get("min_winner_margin", 0.15),
        easyocr_gpu=jersey_cfg.get("easyocr_gpu", False),
        debug_dir=debug_dir,
        template_matching=jersey_cfg.get("template_matching", False),
        template_font_image=jersey_cfg.get("template_font_image"),
        template_min_score=jersey_cfg.get("template_min_score", 0.62),
        template_weight=jersey_cfg.get("template_weight", 0.03),
        template_max_candidates=jersey_cfg.get("template_max_candidates", 4),
        aggregate_by_crop=jersey_cfg.get("aggregate_by_crop", True),
        max_candidates_per_crop=jersey_cfg.get("max_candidates_per_crop", 3),
        min_crop_candidate_ratio=jersey_cfg.get("min_crop_candidate_ratio", 0.35),
        crop_quality_vote_weighting=jersey_cfg.get("crop_quality_vote_weighting", False),
        crop_quality_min_vote_weight=jersey_cfg.get("crop_quality_min_vote_weight", 0.35),
        crop_quality_vote_power=jersey_cfg.get("crop_quality_vote_power", 1.0),
        mmocr_device=jersey_cfg.get("mmocr_device"),
        mmocr_det=jersey_cfg.get("mmocr_det", "dbnet_resnet18_fpnc_1200e_icdar2015"),
        mmocr_rec=jersey_cfg.get("mmocr_rec", "SAR"),
        mmocr_batch_size=jersey_cfg.get("mmocr_batch_size", 8),
        mmocr_direct_recognition=jersey_cfg.get("mmocr_direct_recognition"),
        progress_every=jersey_cfg.get("progress_every", 5),
        cache_enabled=jersey_cfg.get("cache_enabled", True),
        cache_dir=jersey_cfg.get("cache_dir", ".ft_cache/ocr"),
        number_roi_enabled=jersey_cfg.get("number_roi_enabled", False),
        number_roi_upscale=jersey_cfg.get("number_roi_upscale", 3),
        number_roi_clahe=jersey_cfg.get("number_roi_clahe", True),
        broadcast_contrast_enabled=jersey_cfg.get("broadcast_contrast_enabled", False),
        broadcast_contrast_clip_limit=jersey_cfg.get("broadcast_contrast_clip_limit", 4.0),
        broadcast_contrast_tile_grid_size=jersey_cfg.get("broadcast_contrast_tile_grid_size", 4),
        super_resolution_enabled=jersey_cfg.get("super_resolution_enabled", False),
        super_resolution_scale=jersey_cfg.get("super_resolution_scale", 4),
        super_resolution_max_side=jersey_cfg.get("super_resolution_max_side", 100),
        segment_frames=jersey_cfg.get("segment_frames", 0) if segment_frames is None else segment_frames,
        demote_direct_only_single_digits=jersey_cfg.get("demote_direct_only_single_digits", True),
        prefer_two_digit_candidates=jersey_cfg.get("prefer_two_digit_candidates", True),
        digit_confusion_overrides=jersey_cfg.get("digit_confusion_overrides"),
    )


def segment_jersey_candidate_rows(jersey_assignments, rows, roster):
    roster_lookup = roster_players_by_team_jersey(roster)
    candidate_rows = []
    for track_id, assignment in sorted((jersey_assignments or {}).items(), key=lambda item: assignment_key_string(item[0])):
        display_id = assignment_display_id(track_id, assignment)
        segment_index = assignment.get("segment_index")
        scoped_rows = rows_for_segment_assignment(rows, display_id, assignment)
        if not scoped_rows:
            continue
        if assignment.get("jersey_number") in (None, "", "None"):
            continue

        team_id = majority_int(scoped_rows, "team_id")
        role = majority_text(scoped_rows, "role_detection")
        jersey = int(assignment["jersey_number"])
        if team_id is not None:
            roster_matches = roster_lookup.get((int(team_id), jersey), [])
        else:
            roster_matches = [
                player
                for (candidate_team, candidate_jersey), players in roster_lookup.items()
                if int(candidate_jersey) == jersey
                for player in players
            ]
        frames = [int(row.get("frame", 0) or 0) for row in scoped_rows]
        candidate_rows.append(
            {
                "display_track_id": int(display_id),
                "identity_tracklet_id": identity_tracklet_id(display_id, segment_index),
                "segment_index": segment_index,
                "segment_start_frame": assignment.get("segment_start_frame"),
                "segment_end_frame": assignment.get("segment_end_frame"),
                "observed_start_frame": min(frames),
                "observed_end_frame": max(frames),
                "rows": len(scoped_rows),
                "team_id": team_id,
                "role_detection": role,
                "jersey_number": jersey,
                "confidence": float(assignment.get("confidence", 0.0) or 0.0),
                "head_confidence": assignment.get("head_confidence"),
                "winner_margin": assignment.get("winner_margin"),
                "winner_score_ratio": assignment.get("winner_score_ratio"),
                "votes": int(assignment.get("votes", 0) or 0),
                "jersey_roster_mass": float(assignment.get("jersey_roster_mass", 0.0) or 0.0),
                "candidate_player_ids": [player["player_id"] for player in roster_matches],
                "candidate_player_names": [player.get("name", player["player_id"]) for player in roster_matches],
                "raw_jersey_distribution": assignment.get("raw_jersey_distribution"),
                "jersey_distribution": assignment.get("jersey_distribution"),
                "jersey_candidates": assignment.get("candidates"),
                "jersey_roster_filter": assignment.get("roster_filter"),
            }
        )
    return candidate_rows


def apply_segment_jersey_identity_candidates(
    tracks,
    segment_candidates,
    enabled=False,
    min_confidence=0.20,
    min_votes=8,
    max_jersey_display_spread=3,
    require_unique_roster_player=True,
    only_unknown_identity=True,
):
    if not enabled:
        return {"enabled": False, "status": "disabled", "applied_tracklets": 0}
    if not segment_candidates:
        return {"enabled": True, "status": "missing_segment_candidates", "applied_tracklets": 0}

    spread = segment_candidate_display_spread(segment_candidates)
    accepted = {}
    rejected = Counter()
    rejected_examples = []
    for row in segment_candidates:
        reason = segment_candidate_rejection_reason(
            row,
            spread,
            min_confidence=min_confidence,
            min_votes=min_votes,
            max_jersey_display_spread=max_jersey_display_spread,
            require_unique_roster_player=require_unique_roster_player,
        )
        if reason:
            rejected[reason] += 1
            if len(rejected_examples) < 25:
                rejected_examples.append(
                    {
                        "display_track_id": row.get("display_track_id"),
                        "segment_index": row.get("segment_index"),
                        "jersey_number": row.get("jersey_number"),
                        "reason": reason,
                    }
                )
            continue
        key = segment_candidate_key(row)
        current = accepted.get(key)
        if current is None or segment_candidate_sort_key(row, spread) > segment_candidate_sort_key(current, spread):
            accepted[key] = row

    accepted_by_display = {}
    for (display_id, _segment_index), row in accepted.items():
        accepted_by_display.setdefault(display_id, []).append(row)
    for display_id in accepted_by_display:
        accepted_by_display[display_id].sort(
            key=lambda row: (
                int(row.get("segment_start_frame") or 0),
                int(row.get("segment_end_frame") or 0),
            )
        )

    applied_segments = set()
    skipped_assigned = set()
    applied_rows = 0
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            candidate = segment_candidate_for_frame(accepted_by_display.get(display_id, []), frame_num)
            if not candidate:
                continue
            if only_unknown_identity and track.get("player_id") not in (None, "", "unknown"):
                skipped_assigned.add(segment_candidate_key(candidate))
                continue
            player_ids = csv_json_list(candidate.get("candidate_player_ids"))
            player_names = csv_json_list(candidate.get("candidate_player_names"))
            if not player_ids:
                continue
            track["segment_candidate_player_id"] = player_ids[0]
            track["segment_candidate_player_name"] = player_names[0] if player_names else player_ids[0]
            track["segment_candidate_team_id"] = candidate.get("team_id")
            track["segment_candidate_jersey_number"] = candidate.get("jersey_number")
            track["segment_candidate_confidence"] = float(candidate.get("confidence", 0.0) or 0.0)
            track["segment_candidate_votes"] = int(float(candidate.get("votes", 0) or 0))
            track["segment_candidate_reason"] = "segment_jersey_candidate"
            track["segment_candidate_evidence"] = {
                "segment_index": candidate.get("segment_index"),
                "segment_start_frame": candidate.get("segment_start_frame"),
                "segment_end_frame": candidate.get("segment_end_frame"),
                "jersey_display_spread": spread.get(str(candidate.get("jersey_number")), 0),
                "raw_jersey_distribution": candidate.get("raw_jersey_distribution"),
                "jersey_distribution": candidate.get("jersey_distribution"),
            }
            applied_segments.add(segment_candidate_key(candidate))
            applied_rows += 1

    return {
        "enabled": True,
        "status": "ok",
        "input_candidates": len(segment_candidates),
        "accepted_segments": len(accepted),
        "applied_segments": len(applied_segments),
        "applied_rows": applied_rows,
        "skipped_assigned_segments": len(skipped_assigned),
        "rejected": dict(rejected),
        "rejected_examples": rejected_examples,
        "accepted": {
            segment_candidate_key_string(key): {
                "jersey_number": row.get("jersey_number"),
                "display_track_id": row.get("display_track_id"),
                "segment_index": row.get("segment_index"),
                "segment_start_frame": row.get("segment_start_frame"),
                "segment_end_frame": row.get("segment_end_frame"),
                "team_id": row.get("team_id"),
                "confidence": row.get("confidence"),
                "votes": row.get("votes"),
                "candidate_player_ids": csv_json_list(row.get("candidate_player_ids")),
                "jersey_display_spread": spread.get(str(row.get("jersey_number")), 0),
            }
            for key, row in sorted(accepted.items())
        },
    }


def segment_candidate_key(row):
    return (
        int(row.get("display_track_id")),
        int(float(row.get("segment_index") or 0)),
    )


def segment_candidate_key_string(key):
    display_id, segment_index = key
    return f"{int(display_id)}:{int(segment_index)}"


def segment_candidate_for_frame(candidates, frame_num):
    for candidate in candidates:
        start = candidate.get("segment_start_frame")
        end = candidate.get("segment_end_frame")
        if start not in (None, "", "None") and int(frame_num) < int(float(start)):
            continue
        if end not in (None, "", "None") and int(frame_num) > int(float(end)):
            continue
        return candidate
    return None


def segment_candidate_rejection_reason(
    row,
    display_spread,
    min_confidence,
    min_votes,
    max_jersey_display_spread,
    require_unique_roster_player,
):
    confidence = float(row.get("confidence", 0.0) or 0.0)
    if confidence < float(min_confidence):
        return "low_confidence"
    votes = int(float(row.get("votes", 0) or 0))
    if votes < int(min_votes):
        return "low_votes"
    jersey = str(row.get("jersey_number"))
    if int(display_spread.get(jersey, 0)) > int(max_jersey_display_spread):
        return "high_jersey_display_spread"
    player_ids = csv_json_list(row.get("candidate_player_ids"))
    if require_unique_roster_player and len(player_ids) != 1:
        return "not_unique_roster_player"
    return None


def segment_candidate_sort_key(row, display_spread):
    jersey = str(row.get("jersey_number"))
    return (
        -int(display_spread.get(jersey, 0)),
        float(row.get("confidence", 0.0) or 0.0),
        int(float(row.get("votes", 0) or 0)),
    )


def segment_candidate_display_spread(segment_candidates):
    displays_by_jersey = {}
    for row in segment_candidates:
        jersey = str(row.get("jersey_number"))
        displays_by_jersey.setdefault(jersey, set()).add(int(row.get("display_track_id")))
    return {jersey: len(displays) for jersey, displays in displays_by_jersey.items()}


def csv_json_list(value):
    if isinstance(value, list):
        return value
    if value in (None, "", "None", "unknown"):
        return []
    if isinstance(value, str):
        import json

        try:
            decoded = json.loads(value)
        except Exception:
            return [value]
        return decoded if isinstance(decoded, list) else [decoded]
    return [value]


def roster_players_by_team_jersey(roster):
    lookup = {}
    for player in roster or []:
        team_id = player.get("team_id")
        jersey = player.get("jersey_number")
        if team_id is None or jersey is None:
            continue
        lookup.setdefault((int(team_id), int(jersey)), []).append(player)
    return lookup


def rows_for_segment_assignment(rows, display_id, assignment):
    start = assignment.get("segment_start_frame")
    end = assignment.get("segment_end_frame")
    scoped = []
    for row in rows:
        row_display_id = int(row.get("display_track_id", row["track_id"]))
        if row_display_id != int(display_id):
            continue
        frame = int(row.get("frame", 0) or 0)
        if start is not None and frame < int(start):
            continue
        if end is not None and frame > int(end):
            continue
        scoped.append(row)
    return scoped


def majority_int(rows, field):
    counts = Counter()
    for row in rows:
        value = row.get(field)
        if value in (None, "", "None", "unknown"):
            continue
        try:
            counts[int(value)] += 1
        except Exception:
            continue
    return counts.most_common(1)[0][0] if counts else None


def majority_text(rows, field):
    counts = Counter(
        str(row.get(field) or "").strip()
        for row in rows
        if str(row.get(field) or "").strip()
    )
    return counts.most_common(1)[0][0] if counts else None


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
        "ball_temporal_consistency": config["detection"].get("ball_temporal_consistency", True),
        "ball_temporal_max_distance": config["detection"].get("ball_temporal_max_distance", 120.0),
        "ball_temporal_max_distance_cap": config["detection"].get("ball_temporal_max_distance_cap", 120.0),
        "ball_temporal_distance_penalty": config["detection"].get("ball_temporal_distance_penalty", 0.35),
        "ball_temporal_reject_outliers": config["detection"].get("ball_temporal_reject_outliers", True),
        "ball_min_acquisition_confidence": config["detection"].get("ball_min_acquisition_confidence", 0.05),
        "ball_low_confidence_max_distance": config["detection"].get("ball_low_confidence_max_distance", 30.0),
        "ball_temporal_min_confidence_after_miss": config["detection"].get(
            "ball_temporal_min_confidence_after_miss", 0.05
        ),
        "ball_temporal_miss_reset": config["detection"].get("ball_temporal_miss_reset", 12),
        "ball_kalman_enabled": config["detection"].get("ball_kalman_enabled", False),
        "ball_kalman_max_lost_frames": config["detection"].get("ball_kalman_max_lost_frames", 8),
        "ball_kalman_process_noise_scale": config["detection"].get("ball_kalman_process_noise_scale", 50.0),
        "ball_kalman_measurement_noise_scale": config["detection"].get("ball_kalman_measurement_noise_scale", 5.0),
        "ball_kalman_high_speed_threshold": config["detection"].get("ball_kalman_high_speed_threshold", 30.0),
        "ball_kalman_high_speed_area_multiplier": config["detection"].get(
            "ball_kalman_high_speed_area_multiplier", 3.0
        ),
    }
    if backend in {"bytetrack", "byte"}:
        print("FT tracking backend: bytetrack", flush=True)
        return YoloByteTracker(**common, **tracking_cfg)
    if backend in {"strongsort", "strong"}:
        print("FT tracking backend: strongsort", flush=True)
        strongsort_args = dict(tracking_cfg)
        strongsort_args.pop("inference_mode", None)
        strongsort_args.pop("half_precision", None)
        for key in (
            "ball_temporal_consistency",
            "ball_temporal_max_distance",
            "ball_temporal_max_distance_cap",
            "ball_temporal_distance_penalty",
            "ball_temporal_reject_outliers",
            "ball_min_acquisition_confidence",
            "ball_low_confidence_max_distance",
            "ball_temporal_min_confidence_after_miss",
            "ball_temporal_miss_reset",
            "ball_kalman_enabled",
            "ball_kalman_max_lost_frames",
            "ball_kalman_process_noise_scale",
            "ball_kalman_measurement_noise_scale",
            "ball_kalman_high_speed_threshold",
            "ball_kalman_high_speed_area_multiplier",
        ):
            common.pop(key, None)
        strongsort_args.update(strongsort_cfg)
        return YoloStrongSortTracker(**common, **strongsort_args)
    raise ValueError(f"Unknown tracking backend: {backend}")


def run_tracker_with_scene_cuts(tracker, frames, scene_cut_frames=None, reset_enabled=False):
    if reset_enabled and isinstance(tracker, YoloByteTracker):
        return tracker.run(frames, scene_cut_frames=scene_cut_frames)
    if reset_enabled:
        print("FT scene cuts: tracking reset is currently supported only for ByteTrack", flush=True)
    return tracker.run(frames)


def ball_tracking_diagnostics(tracks):
    frames = tracks.get("ball", [])
    detected = []
    interpolated = 0
    centers = []
    detected_jumps = []
    jump_records = []
    gated_jump_records = []
    reacquisition_jump_records = []
    previous_detected = None
    previous_detected_frame = None
    for frame_num, frame_tracks in enumerate(frames):
        track = frame_tracks.get(1) if isinstance(frame_tracks, dict) else None
        if not track or len(track.get("bbox", [])) != 4:
            continue
        center = bbox_center(track["bbox"])
        centers.append(center)
        if track.get("interpolated"):
            interpolated += 1
            continue
        detected.append(
            {
                "frame": int(frame_num),
                "confidence": track.get("confidence"),
                "score": track.get("score"),
                "base_score": track.get("base_score"),
                "temporal_distance": track.get("temporal_distance"),
                "temporal_gap": track.get("temporal_gap"),
                "area_ratio": track.get("area_ratio"),
            }
        )
        if previous_detected is not None:
            jump = float(np.linalg.norm(np.asarray(center) - np.asarray(previous_detected)))
            detected_jumps.append(jump)
            record = {
                "from_frame": int(previous_detected_frame),
                "to_frame": int(frame_num),
                "frame_gap": int(frame_num) - int(previous_detected_frame),
                "jump_px": jump,
                "from_center": list(previous_detected),
                "to_center": list(center),
                "confidence": track.get("confidence"),
                "score": track.get("score"),
                "temporal_distance": track.get("temporal_distance"),
            }
            jump_records.append(record)
            if track.get("temporal_distance") is None:
                reacquisition_jump_records.append(record)
            else:
                gated_jump_records.append(record)
        previous_detected = center
        previous_detected_frame = frame_num

    confidences = [float(item["confidence"]) for item in detected if item.get("confidence") is not None]
    temporal_distances = [
        float(item["temporal_distance"]) for item in detected if item.get("temporal_distance") is not None
    ]
    return {
        "total_frames": len(frames),
        "ball_frames": len(centers),
        "detected_frames": len(detected),
        "interpolated_frames": int(interpolated),
        "mean_detection_confidence": float(np.mean(confidences)) if confidences else None,
        "median_temporal_distance": float(np.median(temporal_distances)) if temporal_distances else None,
        "p95_temporal_distance": float(np.percentile(temporal_distances, 95)) if temporal_distances else None,
        "max_detected_jump_px": float(max(detected_jumps)) if detected_jumps else None,
        "p95_detected_jump_px": float(np.percentile(detected_jumps, 95)) if detected_jumps else None,
        "max_gated_detected_jump_px": max_jump(gated_jump_records),
        "p95_gated_detected_jump_px": percentile_jump(gated_jump_records, 95),
        "max_reacquisition_jump_px": max_jump(reacquisition_jump_records),
        "largest_detected_jumps": sorted(jump_records, key=lambda item: item["jump_px"], reverse=True)[:20],
        "largest_gated_detected_jumps": sorted(
            gated_jump_records, key=lambda item: item["jump_px"], reverse=True
        )[:20],
        "largest_reacquisition_jumps": sorted(
            reacquisition_jump_records, key=lambda item: item["jump_px"], reverse=True
        )[:20],
        "detected_samples": detected[:25],
    }


def max_jump(records):
    if not records:
        return None
    return float(max(float(item["jump_px"]) for item in records))


def percentile_jump(records, percentile):
    if not records:
        return None
    return float(np.percentile([float(item["jump_px"]) for item in records], percentile))


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
    assignment_lookup = build_jersey_assignment_lookup(jersey_assignments)
    for row in rows:
        assignment = row_jersey_assignment(row, assignment_lookup)
        if not assignment:
            continue
        row["identity_tracklet_id"] = identity_tracklet_id_for_row(row, assignment)
        row["jersey_segment_index"] = assignment.get("segment_index")
        row["jersey_number"] = assignment["jersey_number"]
        row["jersey_confidence"] = assignment["confidence"]
        row["jersey_head_confidence"] = assignment.get("head_confidence")
        row["jersey_winner_margin"] = assignment.get("winner_margin")
        row["jersey_winner_score_ratio"] = assignment.get("winner_score_ratio")
        row["jersey_votes"] = assignment["votes"]
        row["jersey_roster_filter"] = assignment.get("roster_filter")
        row["jersey_candidates"] = assignment.get("candidates")
        row["raw_jersey_distribution"] = assignment.get("raw_jersey_distribution")
        row["jersey_distribution"] = assignment.get("jersey_distribution")
        row["jersey_roster_mass"] = assignment.get("jersey_roster_mass", 0.0)
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        for raw_id, track in frame_tracks.items():
            assignment = track_jersey_assignment(raw_id, track, frame_num, assignment_lookup)
            if not assignment:
                continue
            track["identity_tracklet_id"] = identity_tracklet_id_for_track(raw_id, track, assignment)
            track["jersey_segment_index"] = assignment.get("segment_index")
            track["jersey_number"] = assignment["jersey_number"]
            track["jersey_confidence"] = assignment["confidence"]
            track["jersey_head_confidence"] = assignment.get("head_confidence")
            track["jersey_winner_margin"] = assignment.get("winner_margin")
            track["jersey_winner_score_ratio"] = assignment.get("winner_score_ratio")
            track["jersey_votes"] = assignment["votes"]
            track["jersey_evidence"] = assignment
            track["jersey_roster_filter"] = assignment.get("roster_filter")
            track["jersey_candidates"] = assignment.get("candidates")
            track["raw_jersey_distribution"] = assignment.get("raw_jersey_distribution")
            track["jersey_distribution"] = assignment.get("jersey_distribution")
            track["jersey_roster_mass"] = assignment.get("jersey_roster_mass", 0.0)


def build_jersey_assignment_lookup(jersey_assignments):
    direct = {}
    segments = {}
    for key, assignment in (jersey_assignments or {}).items():
        display_id = assignment_display_id(key, assignment)
        if assignment.get("segment_index") is None:
            direct[display_id] = assignment
            continue
        segments.setdefault(display_id, []).append(assignment)
    for display_id in segments:
        segments[display_id].sort(
            key=lambda assignment: (
                int(assignment.get("segment_start_frame") or 0),
                int(assignment.get("segment_end_frame") or 0),
            )
        )
    return {"direct": direct, "segments": segments}


def row_jersey_assignment(row, assignment_lookup):
    display_id = int(row.get("display_track_id", row["track_id"]))
    frame = int(row.get("frame", 0) or 0)
    for assignment in assignment_lookup["segments"].get(display_id, []):
        start = assignment.get("segment_start_frame")
        end = assignment.get("segment_end_frame")
        if start is not None and frame < int(start):
            continue
        if end is not None and frame > int(end):
            continue
        return assignment
    return assignment_lookup["direct"].get(display_id)


def track_jersey_assignment(raw_id, track, frame_num, assignment_lookup):
    display_id = int(track.get("display_track_id", raw_id))
    return row_jersey_assignment(
        {"display_track_id": display_id, "track_id": raw_id, "frame": frame_num},
        assignment_lookup,
    )


def assignment_display_id(track_id, assignment):
    if assignment.get("display_track_id") is not None:
        return int(assignment["display_track_id"])
    if isinstance(track_id, tuple):
        return int(track_id[0])
    return int(track_id)


def assignment_key_string(track_id):
    if isinstance(track_id, tuple):
        return f"{int(track_id[0])}:{int(track_id[1])}"
    return str(int(track_id))


def identity_tracklet_id(display_id, segment_index):
    if segment_index is None:
        return int(display_id)
    return int(display_id) * 100000 + int(segment_index)


def identity_tracklet_id_for_row(row, assignment):
    display_id = int(row.get("display_track_id", row["track_id"]))
    return identity_tracklet_id(display_id, assignment.get("segment_index"))


def identity_tracklet_id_for_track(raw_id, track, assignment):
    display_id = int(track.get("display_track_id", raw_id))
    return identity_tracklet_id(display_id, assignment.get("segment_index"))


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
    for track_id, assignment in sorted((jersey_assignments or {}).items(), key=lambda item: assignment_key_string(item[0])):
        display_id = assignment_display_id(track_id, assignment)
        if display_id in goalkeeper_tracklets:
            dropped[assignment_key_string(track_id)] = {
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
