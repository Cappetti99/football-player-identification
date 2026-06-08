from ft.identity.identity_graph import IdentityGraphBuilder, TrackletNode, player_jersey_from_id
from ft.identity.propagation import IdentityPropagationEngine


def test_identity_graph_builds_same_player_edge():
    source = TrackletNode(
        display_track_id=10,
        player_id="team1_09",
        identity_confidence=0.85,
        team_id=1,
        mean_team_confidence=0.90,
        jersey_number=9,
        start_frame=0,
        end_frame=120,
        num_frames=121,
        last_position=[32.0, 21.0],
    )
    target = TrackletNode(
        display_track_id=55,
        team_id=1,
        mean_team_confidence=0.85,
        jersey_number=9,
        jersey_votes=8,
        jersey_confidence=0.70,
        jersey_head_confidence=0.80,
        start_frame=180,
        end_frame=350,
        num_frames=171,
        first_position=[29.0, 19.0],
    )

    edges = IdentityGraphBuilder(max_temporal_gap=200, max_spatial_distance=30.0).build([source, target])

    assert len(edges) == 1
    assert edges[0].source_id == 10
    assert edges[0].target_id == 55
    assert edges[0].team_match is True
    assert edges[0].jersey_match is True
    assert edges[0].composite_score > 0.40


def test_identity_graph_blocks_temporal_overlap():
    source = TrackletNode(
        display_track_id=1,
        player_id="team1_09",
        identity_confidence=0.90,
        team_id=1,
        mean_team_confidence=0.90,
        start_frame=0,
        end_frame=200,
    )
    target = TrackletNode(
        display_track_id=2,
        team_id=1,
        mean_team_confidence=0.80,
        start_frame=100,
        end_frame=300,
    )

    edges = IdentityGraphBuilder(require_jersey_or_strong_appearance=False).build([source, target])

    assert edges == []


def test_identity_graph_can_allow_temporal_overlap_for_partial_apply():
    source = TrackletNode(
        display_track_id=1,
        player_id="team1_09",
        identity_confidence=0.90,
        team_id=1,
        mean_team_confidence=0.90,
        jersey_number=9,
        start_frame=0,
        end_frame=200,
    )
    target = TrackletNode(
        display_track_id=2,
        team_id=1,
        mean_team_confidence=0.80,
        jersey_number=9,
        jersey_votes=4,
        jersey_confidence=0.50,
        jersey_head_confidence=0.60,
        start_frame=100,
        end_frame=300,
    )

    edges = IdentityGraphBuilder(
        allow_temporal_overlap=True,
        require_jersey_or_strong_appearance=True,
        min_composite_score=0.30,
    ).build([source, target])

    assert len(edges) == 1
    assert edges[0].has_temporal_overlap is True
    assert edges[0].jersey_match is True


def test_identity_graph_cut_bridge_ignores_spatial_jump_when_team_jersey_match():
    source = TrackletNode(
        display_track_id=10,
        player_id="team1_09",
        identity_confidence=0.90,
        team_id=1,
        mean_team_confidence=0.90,
        jersey_number=9,
        start_frame=0,
        end_frame=99,
        last_position=[5.0, 5.0],
    )
    target = TrackletNode(
        display_track_id=47,
        team_id=1,
        mean_team_confidence=0.85,
        jersey_number=9,
        jersey_votes=6,
        jersey_confidence=0.45,
        jersey_head_confidence=0.70,
        start_frame=100,
        end_frame=180,
        first_position=[90.0, 60.0],
    )

    edges = IdentityGraphBuilder(
        scene_cut_frames=[100],
        cut_bridge_enabled=True,
        cut_bridge_max_gap=5,
        cut_bridge_min_jersey_confidence=0.20,
        cut_bridge_min_jersey_votes=3,
        max_spatial_distance=10.0,
        min_composite_score=0.30,
    ).build([source, target])

    assert len(edges) == 1
    assert edges[0].cut_bridge is True
    assert edges[0].cut_frame == 100
    assert edges[0].spatial_distance is None
    assert edges[0].jersey_match is True


def test_identity_graph_uses_summary_jersey_before_player_id_suffix():
    source = TrackletNode(
        display_track_id=58,
        player_id="team1_05",
        identity_confidence=0.90,
        team_id=1,
        mean_team_confidence=0.90,
        jersey_number=28,
        start_frame=0,
        end_frame=120,
        last_position=[10.0, 10.0],
    )
    target = TrackletNode(
        display_track_id=99,
        team_id=1,
        mean_team_confidence=0.85,
        jersey_number=28,
        jersey_votes=5,
        jersey_confidence=0.45,
        jersey_head_confidence=0.70,
        start_frame=180,
        end_frame=250,
        first_position=[12.0, 11.0],
    )

    edges = IdentityGraphBuilder(max_temporal_gap=200, max_spatial_distance=30.0).build([source, target])

    assert len(edges) == 1
    assert edges[0].jersey_match is True


def test_identity_graph_blocks_goalkeeper_role_mismatch():
    source = TrackletNode(
        display_track_id=10,
        player_id="team1_09",
        identity_confidence=0.90,
        team_id=1,
        mean_team_confidence=0.90,
        jersey_number=9,
        role_detection="player",
        semantic_group_id=1,
        start_frame=0,
        end_frame=120,
        last_position=[10.0, 10.0],
    )
    target = TrackletNode(
        display_track_id=55,
        team_id=1,
        mean_team_confidence=0.90,
        jersey_number=9,
        jersey_votes=5,
        jersey_confidence=0.45,
        jersey_head_confidence=0.70,
        role_detection="goalkeeper",
        semantic_group_id=3,
        start_frame=180,
        end_frame=250,
        first_position=[12.0, 11.0],
    )
    builder = IdentityGraphBuilder(max_temporal_gap=200, max_spatial_distance=30.0)

    edges = builder.build([source, target])

    assert edges == []
    assert builder.diagnostics()["rejected"]["goalkeeper_role_mismatch"] == 1


def test_identity_propagation_applies_to_unknown_tracklet():
    roster = [
        {"player_id": "team1_09", "team_id": 1, "jersey_number": 9, "name": "Lautaro"},
    ]
    tracks = {
        "players": [
            {10: {"display_track_id": 10, "player_id": "team1_09", "identity_confidence": 0.85, "team": 1}},
            {},
            {},
            {55: {"display_track_id": 55, "player_id": "unknown", "identity_confidence": 0.0, "team": 1}},
        ]
    }
    summaries = [
        summary(10, 0, 0, team_id=1, jersey_number=9, jersey_votes=8, jersey_confidence=0.70),
        summary(55, 3, 3, team_id=1, jersey_number=9, jersey_votes=4, jersey_confidence=0.45),
    ]
    assignments = {
        10: {"player_id": "team1_09", "player_name": "Lautaro", "confidence": 0.85},
        55: {"player_id": "unknown", "player_name": "unknown", "confidence": 0.0},
    }

    diagnostics = IdentityPropagationEngine(
        roster=roster,
        min_composite_score=0.30,
        min_score_margin=0.0,
        min_source_confidence=0.50,
        max_temporal_gap=100,
        require_jersey_or_strong_appearance=True,
        min_partial_frames=1,
    ).apply(tracks, summaries, assignments)

    assert diagnostics["total_propagated"] == 1
    assert tracks["players"][3][55]["player_id"] == "team1_09"
    assert tracks["players"][3][55]["jersey_number"] == 9
    assert tracks["players"][3][55]["identity_evidence"]["status"] == "propagated"


def test_identity_propagation_blocks_frame_conflict():
    roster = [
        {"player_id": "team1_09", "team_id": 1, "jersey_number": 9, "name": "Lautaro"},
    ]
    tracks = {
        "players": [
            {10: {"display_track_id": 10, "player_id": "team1_09", "identity_confidence": 0.85, "team": 1}},
            {},
            {},
            {
                55: {"display_track_id": 55, "player_id": "unknown", "identity_confidence": 0.0, "team": 1},
                99: {"display_track_id": 99, "player_id": "team1_09", "identity_confidence": 0.80, "team": 1, "jersey_number": 9},
            },
        ]
    }
    summaries = [
        summary(10, 0, 0, team_id=1, jersey_number=9, jersey_votes=8, jersey_confidence=0.70),
        summary(55, 3, 3, team_id=1, jersey_number=9, jersey_votes=4, jersey_confidence=0.45),
    ]
    assignments = {
        10: {"player_id": "team1_09", "player_name": "Lautaro", "confidence": 0.85},
        55: {"player_id": "unknown", "player_name": "unknown", "confidence": 0.0},
    }

    diagnostics = IdentityPropagationEngine(
        roster=roster,
        min_composite_score=0.30,
        min_score_margin=0.0,
        min_source_confidence=0.50,
        max_temporal_gap=100,
        require_jersey_or_strong_appearance=True,
        min_partial_frames=1,
    ).apply(tracks, summaries, assignments)

    assert diagnostics["total_propagated"] == 0
    assert diagnostics["rejected_propagations"][0]["reason"] == "frame_conflict"
    assert tracks["players"][3][55]["player_id"] == "unknown"


def test_identity_propagation_can_apply_only_non_conflicting_frames():
    roster = [
        {"player_id": "team1_09", "team_id": 1, "jersey_number": 9, "name": "Lautaro"},
    ]
    tracks = {
        "players": [
            {10: {"display_track_id": 10, "player_id": "team1_09", "identity_confidence": 0.85, "team": 1}},
            {},
            {55: {"display_track_id": 55, "player_id": "unknown", "identity_confidence": 0.0, "team": 1}},
            {
                55: {"display_track_id": 55, "player_id": "unknown", "identity_confidence": 0.0, "team": 1},
                99: {"display_track_id": 99, "player_id": "team1_09", "identity_confidence": 0.80, "team": 1, "jersey_number": 9},
            },
            {55: {"display_track_id": 55, "player_id": "unknown", "identity_confidence": 0.0, "team": 1}},
        ]
    }
    summaries = [
        summary(10, 0, 0, team_id=1, jersey_number=9, jersey_votes=8, jersey_confidence=0.70),
        summary(55, 2, 4, team_id=1, jersey_number=9, jersey_votes=4, jersey_confidence=0.45),
    ]
    assignments = {
        10: {"player_id": "team1_09", "player_name": "Lautaro", "confidence": 0.85},
        55: {"player_id": "unknown", "player_name": "unknown", "confidence": 0.0},
    }

    diagnostics = IdentityPropagationEngine(
        roster=roster,
        min_composite_score=0.30,
        min_score_margin=0.0,
        min_source_confidence=0.50,
        max_temporal_gap=100,
        require_jersey_or_strong_appearance=True,
        allow_partial_conflict_frames=True,
        min_partial_frames=2,
        min_partial_fraction=0.50,
    ).apply(tracks, summaries, assignments)

    assert diagnostics["total_propagated"] == 1
    assert diagnostics["propagations"][0]["applied_frames"] == 2
    assert tracks["players"][2][55]["player_id"] == "team1_09"
    assert tracks["players"][3][55]["player_id"] == "unknown"
    assert tracks["players"][4][55]["player_id"] == "team1_09"


def test_identity_propagation_can_disable_goalkeeper_sources():
    roster = [
        {"player_id": "team1_sub_12", "team_id": 1, "jersey_number": 12, "role": "goalkeeper", "name": "Keeper"},
    ]
    tracks = {
        "players": [
            {12: {"display_track_id": 12, "player_id": "team1_sub_12", "identity_confidence": 0.95, "team": 1}},
            {},
            {60: {"display_track_id": 60, "player_id": "unknown", "identity_confidence": 0.0, "team": 1}},
        ]
    }
    summaries = [
        summary(12, 0, 0, team_id=1, jersey_number=12, jersey_votes=8, jersey_confidence=0.80),
        summary(60, 2, 2, team_id=1, jersey_number=12, jersey_votes=4, jersey_confidence=0.45),
    ]
    assignments = {
        12: {"player_id": "team1_sub_12", "player_name": "Keeper", "confidence": 0.95},
        60: {"player_id": "unknown", "player_name": "unknown", "confidence": 0.0},
    }

    diagnostics = IdentityPropagationEngine(
        roster=roster,
        min_composite_score=0.30,
        min_score_margin=0.0,
        min_source_confidence=0.50,
        max_temporal_gap=100,
        require_jersey_match=True,
        require_jersey_or_strong_appearance=True,
        min_partial_frames=1,
        propagate_goalkeepers=False,
    ).apply(tracks, summaries, assignments)

    assert diagnostics["total_propagated"] == 0
    assert diagnostics["rejected_propagations"][0]["reason"] == "goalkeeper_source_propagation_disabled"
    assert tracks["players"][2][60]["player_id"] == "unknown"


def test_player_jersey_from_id_parsing():
    assert player_jersey_from_id("team1_09") == 9
    assert player_jersey_from_id("team2_23") == 23
    assert player_jersey_from_id("unknown") is None
    assert player_jersey_from_id("referee_yellow") is None


def summary(track_id, start, end, team_id=None, jersey_number=None, jersey_votes=0, jersey_confidence=0.0):
    return {
        "track_id": track_id,
        "team_id": team_id,
        "mean_team_confidence": 0.85 if team_id is not None else 0.0,
        "jersey_number": jersey_number,
        "jersey_votes": jersey_votes,
        "jersey_confidence": jersey_confidence,
        "jersey_head_confidence": 0.70 if jersey_number is not None else 0.0,
        "jersey_winner_margin": 0.25 if jersey_number is not None else 0.0,
        "start_frame": start,
        "end_frame": end,
        "num_frames": end - start + 1,
        "mean_pitch_position": None,
        "visual_embedding": None,
        "mean_crop_quality": 0.4,
        "role_detection": "player",
    }


if __name__ == "__main__":
    test_identity_graph_builds_same_player_edge()
    test_identity_graph_blocks_temporal_overlap()
    test_identity_graph_can_allow_temporal_overlap_for_partial_apply()
    test_identity_graph_cut_bridge_ignores_spatial_jump_when_team_jersey_match()
    test_identity_graph_uses_summary_jersey_before_player_id_suffix()
    test_identity_graph_blocks_goalkeeper_role_mismatch()
    test_identity_propagation_applies_to_unknown_tracklet()
    test_identity_propagation_blocks_frame_conflict()
    test_identity_propagation_can_apply_only_non_conflicting_frames()
    test_identity_propagation_can_disable_goalkeeper_sources()
    test_player_jersey_from_id_parsing()
    print("identity propagation tests passed")
