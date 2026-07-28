from ft.features.jersey_region_ctc_audit import (
    collect_selected_crops,
    conservative_preview,
    display_track_id_from_key,
    frame_scene_segment_lookup,
    roster_numbers_by_team,
    roster_rerank_preview,
    roster_rerank_summary,
    standalone_by_plain_display_id,
    team_by_display_track_id,
    tracklet_key,
)
from ft.validation import _validate_jersey_region_ctc_audit


def test_collect_selected_crops_is_deduplicated(tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"x")
    diagnostics = {
        "tracklets": {
            "one": {
                "display_track_id": 7,
                "selected_crops": [
                    {"crop_path": str(crop), "frame": 10},
                    {"crop_path": str(crop), "frame": 10},
                ],
            }
        }
    }
    rows = collect_selected_crops(diagnostics)
    assert len(rows) == 1
    assert rows[0]["display_track_id"] == 7


def test_collect_selected_crops_enforces_topk_and_temporal_gap(tmp_path):
    crops = []
    scores = []
    for frame, score in ((1, 0.8), (2, 0.9), (10, 0.7), (20, 0.6)):
        path = tmp_path / f"{frame}.jpg"
        path.write_bytes(b"x")
        crops.append({"crop_path": str(path), "frame": frame})
        scores.append({
            "crop_path": str(path), "selection_score": score,
            "selection_reason": "selected", "selection_rank": frame,
        })
    diagnostics = {"tracklets": {"one": {"display_track_id": 7, "selected_crops": crops}}}
    rows = collect_selected_crops(
        diagnostics,
        frame_selection_rows=scores,
        max_crops_per_tracklet=2,
        min_frame_gap=5,
    )
    assert [row["frame"] for row in rows] == [2, 10]
    assert rows[0]["selection_reason"] == "selected"
    assert rows[0]["selection_rank"] == 2


def test_conservative_preview_never_fills_primary_abstention():
    assert conservative_preview(None, 10, 0.99, 0.90) == (
        None,
        "primary_abstention_preserved",
    )
    assert conservative_preview(11, 10, 0.89, 0.90) == (
        11,
        "candidate_below_threshold",
    )
    assert conservative_preview(11, 10, 0.90, 0.90) == (
        10,
        "high_confidence_override",
    )


def test_roster_numbers_by_team_groups_by_team_id():
    roster = [
        {"team_id": 1, "jersey_number": 7},
        {"team_id": 1, "jersey_number": 9},
        {"team_id": 2, "jersey_number": 7},
        {"team_id": None, "jersey_number": 5},
    ]
    by_team = roster_numbers_by_team(roster)
    assert by_team == {1: {7, 9}, 2: {7}}


def test_team_by_display_track_id_majority_vote():
    rows = [
        {"display_track_id": 3, "team_id": 1},
        {"display_track_id": 3, "team_id": 1},
        {"display_track_id": 3, "team_id": 2},
        {"display_track_id": 4, "team_id": 2},
    ]
    assert team_by_display_track_id(rows) == {"3": 1, "4": 2}


def test_roster_rerank_preview_disabled_is_a_no_op():
    preview = roster_rerank_preview(
        "1", 55, {"55": 0.9, "95": 0.05}, {}, {}, 5, enabled=False,
    )
    assert preview["preview_number"] == 55
    assert preview["changed"] is False
    assert preview["reason"] == "roster_reranking_disabled"


def test_roster_rerank_preview_corrects_within_topk_when_roster_available():
    # Documented failure mode: CTC reads 55 with high confidence but the true
    # number 95 is a lower-ranked candidate; the team roster only contains 95.
    scores = {"55": 0.6, "95": 0.3, "12": 0.1}
    team_by_display = {"1": 2}
    roster_by_team = {2: {95, 30, 7}}
    preview = roster_rerank_preview(
        "1", 55, scores, team_by_display, roster_by_team, top_k=5, enabled=True,
    )
    assert preview["preview_number"] == 95
    assert preview["changed"] is True
    assert preview["reason"] == "roster_constrained_topk"


def test_roster_rerank_preview_never_invents_a_candidate_outside_topk():
    # If no top-k candidate matches the roster, the preview must fall back to
    # the original prediction rather than guessing a roster member absent
    # from the CTC's own ranking.
    scores = {"55": 0.9, "12": 0.05}
    team_by_display = {"1": 2}
    roster_by_team = {2: {95, 30, 7}}
    preview = roster_rerank_preview(
        "1", 55, scores, team_by_display, roster_by_team, top_k=5, enabled=True,
    )
    assert preview["preview_number"] == 55
    assert preview["changed"] is False
    assert preview["reason"] == "no_topk_candidate_in_roster"


def test_roster_rerank_preview_missing_team_or_roster_is_a_safe_no_op():
    scores = {"55": 0.9}
    preview = roster_rerank_preview(
        "1", 55, scores, {}, {2: {95}}, top_k=5, enabled=True,
    )
    assert preview["preview_number"] == 55
    assert preview["reason"] == "no_roster_for_team"


def test_roster_rerank_summary_counts_only_changed_rows():
    preview = {
        "1": {"display_track_id": 1, "changed": True},
        "2": {"display_track_id": 2, "changed": False},
    }
    summary = roster_rerank_summary(preview)
    assert summary == {"tracklets": 2, "changed": 1, "changed_display_ids": [1]}


def test_region_ctc_validation_enforces_audit_and_hashes():
    errors = []
    _validate_jersey_region_ctc_audit(
        {
            "enabled": True,
            "mode": "apply",
            "audit_only": False,
            "ctc_checkpoint": "ctc.pth",
            "ctc_checkpoint_sha256": "bad",
            "detector_checkpoint": "detector.pt",
            "detector_checkpoint_sha256": "bad",
            "detector_confidence": 1.1,
            "min_override_confidence": 0.9,
            "box_padding": 0.1,
            "fusion_preview_enabled": "false",
            "batch_size": 1,
            "detector_batch_size": 1,
        },
        errors,
    )
    assert "jersey_region_ctc_audit supports only audit" in errors
    assert "jersey_region_ctc_audit.audit_only must remain true" in errors
    assert any("ctc_checkpoint_sha256" in error for error in errors)
    assert any("detector_checkpoint_sha256" in error for error in errors)
    assert any("fusion_preview_enabled" in error for error in errors)


def test_tracklet_key_falls_back_to_plain_display_id_without_scene_segment():
    assert tracklet_key(4, None) == "4"
    assert tracklet_key(4, "") == "4"


def test_tracklet_key_disambiguates_reused_display_track_id_by_scene_segment():
    first = tracklet_key(4, "0")
    second = tracklet_key(4, "2")
    assert first != second
    assert display_track_id_from_key(first) == 4
    assert display_track_id_from_key(second) == 4


def test_frame_scene_segment_lookup_indexes_by_display_id_and_frame():
    player_rows = [
        {"display_track_id": 4, "frame": 10, "scene_segment_id": "0"},
        {"display_track_id": 4, "frame": 400, "scene_segment_id": "2"},
    ]
    lookup = frame_scene_segment_lookup(player_rows)
    assert lookup[("4", 10)] == "0"
    assert lookup[("4", 400)] == "2"


def test_collect_selected_crops_separates_reused_display_track_id_by_scene(tmp_path):
    # Same raw display_track_id, but frame 10 belongs to scene 0 (player A)
    # and frame 400 belongs to scene 2 (a different physical player reusing
    # the same tracker ID after a reset). Without scene disambiguation these
    # would be merged into one (wrong) tracklet for CTC aggregation.
    early = tmp_path / "early.jpg"
    late = tmp_path / "late.jpg"
    early.write_bytes(b"x")
    late.write_bytes(b"x")
    diagnostics = {
        "tracklets": {
            "one": {
                "display_track_id": 4,
                "selected_crops": [
                    {"crop_path": str(early), "frame": 10},
                    {"crop_path": str(late), "frame": 400},
                ],
            }
        }
    }
    scene_segment_by_frame = {("4", 10): "0", ("4", 400): "2"}
    rows = collect_selected_crops(diagnostics, scene_segment_by_frame=scene_segment_by_frame)
    keys = {tracklet_key(row["display_track_id"], row["scene_segment_id"]) for row in rows}
    assert keys == {"4#0", "4#2"}


def test_team_by_display_track_id_disambiguates_by_scene_segment():
    # Track "4" is team 1 in scene 0 but team 2 in scene 2 -- a different
    # physical player reusing the raw ID after a reset.
    rows = [
        {"display_track_id": 4, "team_id": 1, "scene_segment_id": "0"},
        {"display_track_id": 4, "team_id": 1, "scene_segment_id": "0"},
        {"display_track_id": 4, "team_id": 2, "scene_segment_id": "2"},
    ]
    result = team_by_display_track_id(rows)
    assert result == {"4#0": 1, "4#2": 2}


def test_roster_rerank_preview_uses_scene_aware_key_for_team_lookup():
    key = tracklet_key(4, "2")
    scores = {"55": 0.6, "95": 0.3}
    team_by_display = {tracklet_key(4, "0"): 1, tracklet_key(4, "2"): 2}
    roster_by_team = {1: {55}, 2: {95}}
    preview = roster_rerank_preview(key, 55, scores, team_by_display, roster_by_team, top_k=5, enabled=True)
    assert preview["display_track_id"] == 4
    assert preview["team_id"] == 2
    assert preview["preview_number"] == 95


def test_standalone_by_plain_display_id_picks_most_evidenced_segment():
    standalone = {
        tracklet_key(4, "0"): {"display_track_id": 4, "recognized_frames": 2, "jersey_number": 19},
        tracklet_key(4, "2"): {"display_track_id": 4, "recognized_frames": 5, "jersey_number": 8},
        "9": {"display_track_id": 9, "recognized_frames": 1, "jersey_number": 7},
    }
    collapsed = standalone_by_plain_display_id(standalone)
    assert set(collapsed) == {"4", "9"}
    assert collapsed["4"]["jersey_number"] == 8
