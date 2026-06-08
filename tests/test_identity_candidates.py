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


def test_identity_candidates_skip_referee_like_tracks():
    tracks = {
        "players": [
            {
                2: {
                    "display_track_id": 20,
                    "player_id": "unknown",
                    "role_detection": "referee_candidate",
                    "referee_like_score": 0.45,
                }
            }
        ]
    }
    scores = [candidate_score(20, "juve_10", cost=0.30, confidence=0.70)]

    candidates = build_identity_candidates(scores, tracks=tracks)
    apply_identity_candidates(tracks, candidates)

    assert 20 not in candidates
    assert "candidate_player_id" not in tracks["players"][0][2]


def test_identity_candidates_ignore_referee_roster_rows():
    tracks = {
        "players": [
            {
                2: {"display_track_id": 20, "player_id": "unknown", "role_detection": "player"}
            }
        ]
    }
    scores = [
        candidate_score(20, "referee_yellow", cost=0.20, confidence=0.80, player_role="referee"),
        candidate_score(20, "juve_10", cost=0.30, confidence=0.70, player_role="player", second=True),
    ]

    candidates = build_identity_candidates(scores, tracks=tracks)

    assert candidates[20]["candidate_player_id"] == "juve_10"


def test_identity_candidates_block_widespread_jersey_attractors():
    tracks = {
        "players": [
            {
                1: {"display_track_id": 20, "player_id": "unknown"},
                2: {"display_track_id": 21, "player_id": "unknown"},
                3: {"display_track_id": 22, "player_id": "unknown"},
                4: {"display_track_id": 30, "player_id": "unknown"},
            }
        ]
    }
    scores = [
        candidate_score(20, "team1_02", cost=0.20, confidence=0.80, team_id=1, jersey=2),
        candidate_score(21, "team1_02", cost=0.20, confidence=0.80, team_id=1, jersey=2),
        candidate_score(22, "team1_02", cost=0.20, confidence=0.80, team_id=1, jersey=2),
        candidate_score(30, "team1_25", cost=0.20, confidence=0.80, team_id=1, jersey=25),
    ]

    candidates = build_identity_candidates(
        scores,
        tracks=tracks,
        max_jersey_display_spread=2,
        display_spread_scope="team",
    )

    assert 20 not in candidates
    assert 21 not in candidates
    assert 22 not in candidates
    assert candidates[30]["candidate_player_id"] == "team1_25"
    assert candidates[30]["candidate_evidence"]["candidate_jersey_display_spread"] == 1


def test_identity_candidates_spread_scope_is_per_team():
    scores = [
        candidate_score(20, "team1_02", cost=0.20, confidence=0.80, team_id=1, jersey=2),
        candidate_score(21, "team1_02", cost=0.20, confidence=0.80, team_id=1, jersey=2),
        candidate_score(30, "team2_02", cost=0.20, confidence=0.80, team_id=2, jersey=2),
        candidate_score(31, "team2_02", cost=0.20, confidence=0.80, team_id=2, jersey=2),
    ]

    candidates = build_identity_candidates(scores, max_jersey_display_spread=2, display_spread_scope="team")

    assert set(candidates) == {20, 21, 30, 31}


def candidate_score(track_id, player_id, cost, confidence, second=False, player_role="player", team_id=2, jersey=None):
    jersey = jersey if jersey is not None else (9 if second else 10)
    return {
        "track_id": track_id,
        "player_id": player_id,
        "player_name": player_id,
        "player_team_id": team_id,
        "player_jersey_number": jersey,
        "player_role": player_role,
        "tracklet_team_id": team_id,
        "tracklet_team_confidence": 0.8,
        "tracklet_jersey_number": jersey,
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
    test_identity_candidates_skip_referee_like_tracks()
    test_identity_candidates_ignore_referee_roster_rows()
    test_identity_candidates_block_widespread_jersey_attractors()
    test_identity_candidates_spread_scope_is_per_team()
    print("identity candidate tests passed")
