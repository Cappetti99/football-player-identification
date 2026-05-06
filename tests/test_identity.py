from ft.features.jersey_ocr import parse_number, vote_numbers
from ft.features.roster_aware_ocr import RosterAwareOCRFilter
from ft.identity.hungarian import (
    HungarianPlayerIdentifier,
    is_non_player_tracklet,
    validate_unique_team_jersey,
)
from ft.identity.roster import normalize_jersey_number


def test_hungarian_assignment_prefers_jersey_over_uncertain_team():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "p7", "name": "P7", "team_id": 2, "jersey_number": 7, "position_prior": None},
        {"player_id": "p9", "name": "P9", "team_id": 1, "jersey_number": 9, "position_prior": None},
    ]
    summaries = [
        {
            "track_id": 4,
            "team_id": 1,
            "mean_team_confidence": 0.25,
            "jersey_number": 7,
            "jersey_confidence": 1.0,
            "jersey_votes": 3,
            "num_frames": 40,
            "mean_crop_quality": 0.4,
            "mean_pitch_position": None,
            "visual_embedding": None,
        }
    ]

    assignments, scores = identifier.assign(summaries)

    assert assignments[4]["player_id"] == "p7"
    assert scores[0]["player_id"] == "p7"


def test_unique_team_jersey_constraint_rejects_duplicate_roster_numbers():
    roster = [
        {"player_id": "a", "team_id": 1, "jersey_number": 7},
        {"player_id": "b", "team_id": 1, "jersey_number": 7},
    ]
    try:
        validate_unique_team_jersey(roster)
    except ValueError as exc:
        assert "duplicate players" in str(exc)
    else:
        raise AssertionError("Expected duplicate team/jersey roster validation to fail")


def test_reliable_jersey_blocks_same_team_wrong_number():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "p7", "name": "P7", "team_id": 1, "jersey_number": 7, "position_prior": None},
        {"player_id": "p9", "name": "P9", "team_id": 1, "jersey_number": 9, "position_prior": None},
    ]
    tracklet = {
        "track_id": 4,
        "team_id": 1,
        "mean_team_confidence": 0.8,
        "jersey_number": 7,
        "jersey_confidence": 0.9,
        "jersey_votes": 3,
        "num_frames": 40,
        "mean_crop_quality": 0.4,
        "mean_pitch_position": None,
        "visual_embedding": None,
    }

    p7 = identifier.cost_details(tracklet, identifier.roster[0])
    p9 = identifier.cost_details(tracklet, identifier.roster[1])

    assert p7["cost"] < p9["cost"]
    assert p9["components"]["team_jersey_constraint"] > 0.0


def test_jersey_numbers_start_at_one():
    assert parse_number("1") == 1
    assert parse_number("00") is None
    assert parse_number("0") is None
    try:
        normalize_jersey_number(0)
    except ValueError as exc:
        assert "expected an integer from 1 to 99" in str(exc)
    else:
        raise AssertionError("Expected jersey_number=0 to be rejected")


def test_number_one_is_soft_goalkeeper_prior():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    tracklet = {
        "track_id": 1,
        "team_id": 1,
        "mean_team_confidence": 0.8,
        "jersey_number": 1,
        "jersey_confidence": 0.9,
        "jersey_votes": 3,
        "num_frames": 40,
        "mean_crop_quality": 0.4,
        "mean_pitch_position": None,
        "visual_embedding": None,
    }
    goalkeeper = {"player_id": "gk", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"}
    outfield = {"player_id": "p1", "team_id": 1, "jersey_number": 1, "role": "player"}

    gk_cost = identifier.cost_details(tracklet, goalkeeper)
    outfield_cost = identifier.cost_details(tracklet, outfield)

    assert gk_cost["raw_cost"] < outfield_cost["raw_cost"]
    assert gk_cost["components"]["goalkeeper_number_one_prior"] < 0.0
    assert outfield_cost["components"]["goalkeeper_number_one_prior"] > 0.0


def test_ocr_vote_requires_raw_confidence_filter():
    detections = [
        {"number": 7, "confidence": 0.01},
        {"number": 9, "confidence": 0.70},
        {"number": 9, "confidence": 0.60},
    ]
    voted = vote_numbers(detections, min_raw_confidence=0.05)
    assert voted["jersey_number"] == 9
    assert voted["votes"] == 2
    assert voted["winner_margin"] > 0.0


def test_summary_preserves_ocr_votes_not_frame_count():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    rows = [
        {
            "frame": frame,
            "track_id": 1,
            "display_track_id": 1,
            "raw_track_id": 1,
            "team_id": 1,
            "team_confidence": 0.8,
            "jersey_number": 7,
            "jersey_confidence": 0.6,
            "jersey_votes": 2,
            "crop_quality": 0.2,
        }
        for frame in range(100)
    ]
    summary = identifier.summarize(rows)[0]
    assert summary["jersey_number"] == 7
    assert summary["jersey_votes"] == 2


def test_roster_aware_ocr_degrades_number_not_in_team_roster():
    roster = [
        {"player_id": "roma_90", "team_id": 1, "jersey_number": 90},
        {"player_id": "verona_31", "team_id": 2, "jersey_number": 31},
    ]
    assignments = {
        1: {"jersey_number": 96, "confidence": 0.9, "votes": 5},
        2: {"jersey_number": 31, "confidence": 0.8, "votes": 3},
    }
    rows = [
        {"display_track_id": 1, "track_id": 1, "team_id": 1},
        {"display_track_id": 2, "track_id": 2, "team_id": 2},
    ]

    filtered, diagnostics = RosterAwareOCRFilter(roster, mode="degrade", confidence_scale=0.5).apply(assignments, rows)

    assert 1 in filtered
    assert 2 in filtered
    assert filtered[1]["confidence"] == 0.45
    assert diagnostics["degraded"]["1"]["status"] == "degraded"


def test_roster_aware_ocr_promotes_valid_alternative():
    roster = [{"player_id": "roma_90", "team_id": 1, "jersey_number": 90}]
    assignments = {
        1: {
            "jersey_number": 96,
            "confidence": 0.5,
            "votes": 5,
            "candidates": [
                {"jersey_number": 96, "confidence": 0.5, "votes": 5},
                {"jersey_number": 90, "confidence": 0.2, "votes": 2},
            ],
        }
    }
    rows = [{"display_track_id": 1, "track_id": 1, "team_id": 1}]

    filtered, diagnostics = RosterAwareOCRFilter(roster, mode="degrade").apply(assignments, rows)

    assert filtered[1]["jersey_number"] == 90
    assert filtered[1]["roster_filter"]["status"] == "distribution_promoted"


def test_hungarian_uses_jersey_distribution_candidate():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "p90", "name": "P90", "team_id": 1, "jersey_number": 90, "position_prior": None},
        {"player_id": "p10", "name": "P10", "team_id": 1, "jersey_number": 10, "position_prior": None},
    ]
    summaries = [
        {
            "track_id": 9,
            "team_id": 1,
            "mean_team_confidence": 0.8,
            "jersey_number": 96,
            "jersey_confidence": 0.4,
            "jersey_votes": 5,
            "jersey_distribution": [
                {"jersey_number": 90, "confidence": 0.55, "votes": 3},
                {"jersey_number": 10, "confidence": 0.10, "votes": 1},
            ],
            "num_frames": 40,
            "mean_crop_quality": 0.4,
            "mean_pitch_position": None,
            "visual_embedding": None,
        }
    ]

    assignments, _ = identifier.assign(summaries)

    assert assignments[9]["player_id"] == "p90"


def test_referee_candidates_are_not_player_identity_summaries():
    assert is_non_player_tracklet(
        [
            {"role_detection": "referee_candidate"},
            {"role_detection": "referee_candidate"},
            {"role_detection": "player"},
        ]
    )
    assert not is_non_player_tracklet(
        [
            {"role_detection": "player"},
            {"role_detection": "player"},
            {"role_detection": "referee_candidate"},
        ]
    )


if __name__ == "__main__":
    test_hungarian_assignment_prefers_jersey_over_uncertain_team()
    test_unique_team_jersey_constraint_rejects_duplicate_roster_numbers()
    test_reliable_jersey_blocks_same_team_wrong_number()
    test_jersey_numbers_start_at_one()
    test_number_one_is_soft_goalkeeper_prior()
    test_ocr_vote_requires_raw_confidence_filter()
    test_summary_preserves_ocr_votes_not_frame_count()
    test_roster_aware_ocr_degrades_number_not_in_team_roster()
    test_roster_aware_ocr_promotes_valid_alternative()
    test_hungarian_uses_jersey_distribution_candidate()
    test_referee_candidates_are_not_player_identity_summaries()
    print("FT identity tests passed")
