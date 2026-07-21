from ft.identity.gate_audit import build_final_identity_states, build_identity_gate_audit


def test_gate_audit_distinguishes_tracklet_and_roster_visual_availability():
    summaries = [{
        "track_id": 7, "display_track_id": 7, "num_frames": 30,
        "visual_embedding": [1.0, 0.0], "mean_pitch_position": [50.0, 30.0],
        "mean_team_confidence": 0.9, "team_id": 1, "jersey_votes": 3,
        "jersey_confidence": 0.19, "jersey_winner_margin": 0.05,
        "raw_jersey_distribution": [{"jersey_number": 8, "confidence": 0.18, "votes": 3}],
    }]
    scores = [{
        "track_id": 7, "player_id": "p8", "player_name": "P8", "cost": "0.4",
        "confidence": "0.6", "visual_similarity": None, "position_prior_distance": 5.0,
        "assignment_gate": {"reason": "insufficient_assignment_evidence", "team_match": True,
                            "team_confidence": 0.9, "visual_similarity": None,
                            "tracklet_frames": 30, "position_prior_distance": 5.0},
    }, {
        "track_id": 7, "player_id": "p9", "player_name": "P9", "cost": "0.55",
        "confidence": "0.45", "visual_similarity": None, "position_prior_distance": None,
        "assignment_gate": {"reason": "insufficient_assignment_evidence"},
    }]
    assignments = {7: {"player_id": "unknown", "identity_status": "unknown", "evidence": {"status": "insufficient_assignment_evidence"}}}
    roster = [{"player_id": "p8", "team_id": 1, "jersey_number": 8, "position_prior": [50.0, 30.0]}]

    report = build_identity_gate_audit(summaries, scores, assignments, roster, {"reliable_jersey_min_candidate_score": 0.20})

    row = report["tracklets"][0]
    assert row["tracklet_visual_available"] is True
    assert row["roster_visual_available"] is False
    assert row["near_miss_reliable_jersey"] is True
    assert abs(row["cost_margin"] - 0.15) < 1e-9
    assert report["summary"]["strong_visual_gate_available"] is False


def test_gate_audit_is_one_row_per_tracklet():
    report = build_identity_gate_audit([], [], {}, [], {})
    assert report["summary"]["tracklets"] == 0
    assert report["tracklets"] == []


def test_gate_audit_reports_post_constraint_partial_state():
    summaries = [{"track_id": 7, "display_track_id": 7, "num_frames": 2}]
    assignments = {
        7: {
            "player_id": "p7",
            "identity_status": "assigned",
            "evidence": {"status": "assigned"},
        }
    }
    tracks = {
        "players": [
            {1: {"display_track_id": 7, "identity_tracklet_id": 7, "player_id": "p7"}},
            {1: {
                "display_track_id": 7,
                "identity_tracklet_id": 7,
                "player_id": "unknown",
                "identity_evidence": {"status": "cleared", "reason": "frame_team_conflict"},
            }},
        ]
    }

    report = build_identity_gate_audit(
        summaries,
        [],
        assignments,
        [],
        final_identity_states=build_final_identity_states(tracks),
    )

    row = report["tracklets"][0]
    assert row["hungarian_assignment_status"] == "assigned"
    assert row["assignment_status"] == "partially_assigned"
    assert row["final_assigned_rows"] == 1
    assert row["final_unknown_rows"] == 1
    assert row["post_assignment_reasons"] == ["frame_team_conflict"]
    assert report["summary"]["changed_after_hungarian_tracklets"] == 1
