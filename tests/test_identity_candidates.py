from ft.identity.candidates import (
    apply_identity_candidates,
    build_identity_candidates,
    identity_candidate_rows,
)


def test_identity_candidates_do_not_overwrite_assigned_tracks():
    tracks = {
        "players": [
            {
                1: {"display_track_id": 10, "player_id": "inter_01"},
                2: {"display_track_id": 20, "player_id": "unknown"},
            }
        ]
    }
    scores = [
        candidate_score(10, "inter_02", cost=0.20, confidence=0.80),
        candidate_score(20, "juve_10", cost=0.30, confidence=0.70, second=False),
        candidate_score(20, "juve_09", cost=0.45, confidence=0.55, second=True),
    ]

    candidates = build_identity_candidates(scores, tracks=tracks, min_margin=0.05)
    apply_identity_candidates(tracks, candidates)

    assert 10 not in candidates
    assert candidates[20]["candidate_player_id"] == "juve_10"
    assert tracks["players"][0][1]["player_id"] == "inter_01"
    assert tracks["players"][0][2]["player_id"] == "unknown"
    assert tracks["players"][0][2]["candidate_player_id"] == "juve_10"
    assert identity_candidate_rows(candidates)[0]["track_id"] == 20


def candidate_score(track_id, player_id, cost, confidence, second=False):
    jersey = 9 if second else 10
    return {
        "track_id": track_id,
        "player_id": player_id,
        "player_name": player_id,
        "player_team_id": 2,
        "player_jersey_number": jersey,
        "tracklet_team_id": 2,
        "tracklet_team_confidence": 0.8,
        "tracklet_jersey_number": 10,
        "tracklet_jersey_confidence": 0.6,
        "tracklet_jersey_votes": 2,
        "tracklet_frames": 30,
        "position_prior_distance": 12.0,
        "visual_similarity": 0.5,
        "assignment_gate": {"pass": False, "reason": "insufficient_assignment_evidence"},
        "components": {"base": 0.25},
        "cost": cost,
        "confidence": confidence,
    }


if __name__ == "__main__":
    test_identity_candidates_do_not_overwrite_assigned_tracks()
    print("identity candidate tests passed")
