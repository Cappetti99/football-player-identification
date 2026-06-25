from copy import deepcopy

from ft.identity.constraints import enforce_identity_constraints
from ft.identity.prtreid_bridge import PRTReIDIdentityBridge


def test_bridge_audit_mode_proposes_without_changing_target():
    tracks, features = fixture_tracks()
    report = bridge(apply=False).apply(tracks, features)

    assert report["anchors"] == 1
    assert len(report["proposed_links"]) == 1
    assert report["applied_rows"] == 0
    assert tracks["players"][3][2]["player_id"] == "unknown"


def test_bridge_apply_assigns_only_unknown_target():
    tracks, features = fixture_tracks()
    report = bridge(apply=True).apply(tracks, features)

    assert report["applied_rows"] == 2
    assert tracks["players"][3][2]["player_id"] == "team1_08"
    assert tracks["players"][3][2]["identity_status"] == "prtreid_bridge"


def test_final_constraints_clear_bridge_identity_conflict():
    tracks, features = fixture_tracks()
    conflict = {
        "display_track_id": 30, "player_id": "team1_08", "player_name": "P8",
        "identity_status": "assigned", "identity_confidence": 0.99,
        "team": 1, "team_confidence": 0.9, "jersey_number": None,
    }
    tracks["players"][3][3] = deepcopy(conflict)
    tracks["players"][4][3] = deepcopy(conflict)
    bridge(apply=True).apply(tracks, features)

    enforce_identity_constraints(
        tracks,
        [{"player_id": "team1_08", "team_id": 1, "jersey_number": 8}],
        frame_team_consistency=False,
        frame_team_split_enabled=False,
        global_team_jersey_owner=False,
    )

    assert tracks["players"][3][2]["player_id"] == "unknown"


def test_bridge_rejects_non_jersey_anchor_assigned_target_overlap_team_and_weak_prototype():
    tracks, features = fixture_tracks()
    tracks["players"][0][1]["identity_evidence"]["assignment_gate"]["reason"] = "strong_team_visual_trajectory"
    tracks["players"][1][1]["identity_evidence"]["assignment_gate"]["reason"] = "strong_team_visual_trajectory"
    assert bridge().apply(tracks, features)["anchors"] == 0

    tracks, features = fixture_tracks()
    for frame in (3, 4):
        tracks["players"][frame][2]["player_id"] = "team1_09"
    assert bridge().apply(tracks, features)["unknown_targets"] == 0

    tracks, features = fixture_tracks()
    tracks["players"][1][2] = deepcopy(tracks["players"][3][2])
    assert bridge().apply(tracks, features)["rejection_counts"]["overlap"] == 1

    tracks, features = fixture_tracks()
    for frame in (3, 4):
        tracks["players"][frame][2]["team"] = 2
    assert bridge().apply(tracks, features)["rejection_counts"]["team"] == 1

    tracks, features = fixture_tracks()
    features[1]["sample_count"] = 1
    assert bridge().apply(tracks, features)["rejection_counts"]["prototype"] == 1


def bridge(apply=False):
    return PRTReIDIdentityBridge(
        apply=apply, min_samples=2, min_prototype_consistency=0.9,
        min_source_confidence=0.75, min_similarity=0.9, min_margin=0.1,
        max_segment_gap=1, max_gap=60,
    )


def fixture_tracks():
    anchor = {
        "display_track_id": 10, "player_id": "team1_08", "player_name": "P8",
        "identity_status": "assigned", "identity_confidence": 0.9,
        "identity_evidence": {"assignment_gate": {"reason": "reliable_jersey"}},
        "team": 1, "team_confidence": 0.9, "scene_segment_id": 0,
    }
    target = {
        "display_track_id": 20, "player_id": "unknown", "player_name": "unknown",
        "identity_status": "unknown", "identity_confidence": 0.0,
        "identity_evidence": {}, "team": 1, "team_confidence": 0.9, "scene_segment_id": 1,
    }
    tracks = {"players": [{1: deepcopy(anchor)}, {1: deepcopy(anchor)}, {}, {2: deepcopy(target)}, {2: deepcopy(target)}]}
    features = [
        {"display_track_id": 10, "visual_embedding": [1.0, 0.0], "sample_count": 3, "prototype_consistency": 0.99},
        {"display_track_id": 20, "visual_embedding": [0.99, 0.01], "sample_count": 3, "prototype_consistency": 0.99},
    ]
    return tracks, features
