from ft.identity.evidence import build_tracklet_evidence
from ft.identity.hungarian import HungarianPlayerIdentifier
from ft.identity.propagation import build_assignment
from ft.identity.identity_graph import CompatibilityEdge, TrackletNode


def test_tracklet_evidence_preserves_raw_and_roster_distributions():
    rows = [
        {
            "track_group": "players",
            "frame": 1,
            "track_id": 10,
            "display_track_id": 10,
            "team_id": 1,
            "team_confidence": 0.9,
            "jersey_number": 71,
            "jersey_confidence": 0.31,
            "jersey_votes": 1,
            "raw_jersey_distribution": [
                {"jersey_number": 17, "confidence": 0.31, "votes": 1},
                {"jersey_number": 71, "confidence": 0.29, "votes": 1},
                {"jersey_number": 77, "confidence": 0.28, "votes": 1},
            ],
            "jersey_distribution": [{"jersey_number": 17, "confidence": 0.31, "votes": 1}],
            "jersey_roster_mass": 0.31,
            "crop_quality": 0.4,
        }
    ]
    roster = [{"player_id": "p17", "team_id": 1, "jersey_number": 17}]

    evidence = build_tracklet_evidence(rows, roster)

    assert evidence[0]["jersey_evidence"]["raw_distribution"][0]["jersey_number"] == 17
    assert {item["jersey_number"] for item in evidence[0]["jersey_evidence"]["raw_distribution"]} == {17, 71, 77}
    assert evidence[0]["jersey_evidence"]["roster_distribution"][0]["jersey_number"] == 17


def test_hungarian_assignment_exposes_identity_contract_fields():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [{"player_id": "p17", "name": "P17", "team_id": 1, "jersey_number": 17}]
    summaries = [
        {
            "track_id": 10,
            "team_id": 1,
            "mean_team_confidence": 0.95,
            "jersey_number": 17,
            "raw_jersey_distribution": [{"jersey_number": 17, "confidence": 0.9, "votes": 3}],
            "jersey_distribution": [{"jersey_number": 17, "confidence": 0.9, "votes": 3}],
            "jersey_confidence": 0.9,
            "jersey_votes": 3,
            "num_frames": 50,
            "mean_crop_quality": 0.4,
            "mean_pitch_position": None,
            "visual_embedding": None,
        }
    ]

    assignments, _scores = identifier.assign(summaries)
    assignment = assignments[10]

    assert assignment["identity_status"] == "assigned"
    assert assignment["identity_confidence"] == assignment["confidence"]
    assert "jersey" in assignment["identity_sources"]
    assert isinstance(assignment["identity_risk_flags"], list)


def test_propagated_assignment_is_marked_as_propagated():
    player = {"player_id": "p17", "name": "P17", "team_id": 1, "jersey_number": 17}
    source = TrackletNode(display_track_id=1, player_id="p17", identity_confidence=0.9)
    target = TrackletNode(display_track_id=2, team_id=1, jersey_number=17, num_frames=40)
    edge = CompatibilityEdge(
        source_id=1,
        target_id=2,
        team_score=1.0,
        jersey_score=1.0,
        temporal_score=0.8,
        spatial_score=0.8,
        appearance_score=0.5,
        team_match=True,
        jersey_match=True,
    )

    assignment = build_assignment(player, source, target, edge, propagation_depth=1)

    assert assignment["identity_status"] == "propagated"
    assert assignment["identity_confidence"] == assignment["confidence"]
    assert assignment["identity_sources"]["propagation"] > 0.0
    assert "propagated" in assignment["identity_risk_flags"]
