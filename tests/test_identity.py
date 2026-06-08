import tempfile
from pathlib import Path

import numpy as np

from ft.features.team import TeamAssigner, normalize_random_states, team_color_ranges_by_team_from_roster
from ft.features.jersey_ocr import (
    aggregate_detections_by_crop,
    apply_crop_quality_vote_weights,
    backend_load_mode,
    diagnostic_key,
    is_ocr_player_row,
    jersey_group_key,
    mmocr_score,
    normalize_mmocr_model_name,
    ocr_decision_diagnostics,
    parse_number,
    preprocess_variants,
    requested_backends,
    vote_numbers,
)
from ft.features.referee import referee_color_ranges_from_roster
from ft.features.goalkeeper import GoalkeeperAppearanceAssigner, goalkeeper_color_ranges_by_team_from_roster
from ft.features.jersey_template import prefer_two_digit_candidates
from ft.features.roster_aware_ocr import RosterAwareOCRFilter
from ft.identity.constraints import enforce_identity_constraints, jersey_rank
from ft.identity.candidates import build_identity_candidates
from ft.identity.hungarian import (
    HungarianPlayerIdentifier,
    apply_assignments,
    is_non_player_tracklet,
    validate_unique_team_jersey,
)
from ft.identity.roster import normalize_jersey_number
from ft.linking.jersey_identity_linker import JerseyIdentityLinker
from ft.linking.tracklet_linker import TrackletLinker
from ft.tracking.yolo_strongsort import Detection, StrongSortTrackerCore
from ft.features.visual import extract_from_crop as extract_visual_from_crop, normalize_embedding_mode
from ft.visualization.overlay import player_label, referee_label


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
            "jersey_votes": 5,
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
        "jersey_votes": 5,
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
    assert parse_number("A17") == 17
    assert parse_number("171") == 17
    try:
        normalize_jersey_number(0)
    except ValueError as exc:
        assert "expected an integer from 1 to 99" in str(exc)
    else:
        raise AssertionError("Expected jersey_number=0 to be rejected")


def test_roster_kit_hint_feeds_referee_colour_ranges(tmp_path):
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(
        """
        [
          {
            "player_id": "referee_yellow",
            "name": "Referee",
            "team_id": null,
            "jersey_number": null,
            "role": "referee",
            "metadata": {
              "kit_hint": {
                "shirt": "fluorescent_yellow",
                "shorts": "black",
                "socks": "fluorescent_yellow"
              }
            }
          }
        ]
        """,
        encoding="utf-8",
    )

    from ft.identity.roster import load_roster

    roster = load_roster(roster_path)
    ranges = referee_color_ranges_from_roster(roster)

    assert roster[0]["metadata"]["kit_color"] == "fluorescent_yellow"
    assert "roster_fluorescent_yellow" in ranges


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
    assert voted["head_confidence"] > 0.5


def test_ocr_digit_confusion_override_prefers_close_alternative():
    detections = [
        {"number": 7, "confidence": 0.60},
        {"number": 7, "confidence": 0.55},
        {"number": 2, "confidence": 0.45},
        {"number": 2, "confidence": 0.35},
    ]

    voted = vote_numbers(
        detections,
        min_raw_confidence=0.05,
        digit_confusion_overrides={"7": {"prefer": 2, "min_alternative_ratio": 0.35}},
    )

    assert voted["jersey_number"] == 2
    assert voted["votes"] == 2
    assert voted["candidates"][0]["jersey_number"] == 7


def test_ocr_digit_confusion_override_keeps_strong_winner():
    detections = [
        {"number": 7, "confidence": 0.90},
        {"number": 7, "confidence": 0.85},
        {"number": 2, "confidence": 0.10},
    ]

    voted = vote_numbers(
        detections,
        min_raw_confidence=0.05,
        digit_confusion_overrides={"7": {"prefer": 2, "min_alternative_ratio": 0.35}},
    )

    assert voted["jersey_number"] == 7


def test_ocr_template_votes_are_weighted_signal():
    detections = [
        {"number": 8, "confidence": 0.60},
        {"number": 18, "confidence": 0.50},
        {"number": 18, "confidence": 0.90, "vote_weight": 0.25, "source": "template"},
    ]

    voted = vote_numbers(detections, min_raw_confidence=0.05)

    assert voted["jersey_number"] == 18
    assert voted["votes"] == 2
    assert voted["candidates"][0]["score"] < 1.0


def test_template_prefers_two_digit_candidates_over_digit_fragments():
    candidates = [
        {"jersey_number": 3, "confidence": 0.9},
        {"jersey_number": 8, "confidence": 0.8},
        {"jersey_number": 38, "confidence": 0.7},
    ]

    filtered = prefer_two_digit_candidates(candidates)

    assert [item["jersey_number"] for item in filtered] == [38]


def test_ocr_aggregates_variants_by_crop_before_voting():
    detections = [
        {"crop_path": "a.jpg", "frame": 1, "variant": "upper_equalized", "number": 38, "confidence": 0.90},
        {"crop_path": "a.jpg", "frame": 1, "variant": "upper_clahe", "number": 38, "confidence": 0.80},
        {"crop_path": "a.jpg", "frame": 1, "variant": "upper_binary", "number": 8, "confidence": 0.99},
        {"crop_path": "b.jpg", "frame": 2, "variant": "upper_equalized", "number": 38, "confidence": 0.70},
    ]

    aggregated = aggregate_detections_by_crop(detections, min_raw_confidence=0.05)
    voted = vote_numbers(aggregated, min_raw_confidence=0.05)

    assert [item["number"] for item in aggregated] == [38, 38]
    assert voted["jersey_number"] == 38
    assert voted["votes"] == 2
    assert voted["total_detections"] == 2


def test_ocr_crop_quality_weighting_downweights_noisy_crops():
    detections = [
        {"crop_path": "low.jpg", "frame": 1, "variant": "upper_equalized", "number": 1, "confidence": 0.95},
        {"crop_path": "low.jpg", "frame": 1, "variant": "upper_clahe", "number": 1, "confidence": 0.90},
        {"crop_path": "high.jpg", "frame": 2, "variant": "upper_equalized", "number": 25, "confidence": 0.50},
    ]
    selected = [
        (0, {"crop_path": "low.jpg", "frame": 1, "crop_quality": 0.10}),
        (0, {"crop_path": "high.jpg", "frame": 2, "crop_quality": 0.90}),
    ]

    weighted = apply_crop_quality_vote_weights(detections, selected, enabled=True, min_weight=0.20)
    aggregated = aggregate_detections_by_crop(weighted, min_raw_confidence=0.05)
    voted = vote_numbers(aggregated, min_raw_confidence=0.05)

    assert voted["jersey_number"] == 25
    low_candidate = next(item for item in aggregated if item["crop_path"] == "low.jpg")
    high_candidate = next(item for item in aggregated if item["crop_path"] == "high.jpg")
    assert low_candidate["crop_quality_vote_weight"] == 0.20
    assert high_candidate["crop_quality_vote_weight"] == 1.0


def test_mmocr_backend_alias_keeps_easyocr_fallback():
    assert requested_backends("mmocr_easyocr") == ["mmocr", "easyocr", "pytesseract"]
    assert requested_backends("mmocr,easyocr") == ["mmocr", "easyocr"]
    assert requested_backends("mmocr_rec+easyocr") == ["mmocr_rec", "easyocr"]
    assert backend_load_mode("mmocr_easyocr") == "combine"
    assert backend_load_mode("mmocr-fallback") == "fallback"
    assert normalize_mmocr_model_name("dbnet_resnet18_fpnc_1200e_icdar2015", task="det") == "dbnet_resnet18_fpnc_1200e_icdar2015"


def test_paddleocr_backend_aliases_are_opt_in():
    assert requested_backends("paddleocr") == ["paddleocr"]
    assert requested_backends("paddle_ocr") == ["paddleocr"]
    assert requested_backends("paddleocr_easyocr") == ["paddleocr", "easyocr", "pytesseract"]


def test_ocr_super_resolution_variants_can_upscale_small_crop():
    try:
        import cv2
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as directory:
        crop_path = Path(directory) / "crop.png"
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[:, 4:6] = 255
        cv2.imwrite(str(crop_path), image)

        baseline = preprocess_variants(crop_path, augment=False)
        upscaled = preprocess_variants(
            crop_path,
            augment=False,
            super_resolution_enabled=True,
            super_resolution_scale=2,
            super_resolution_max_side=20,
        )

    assert baseline
    assert upscaled
    assert upscaled[0][1].shape[0] > baseline[0][1].shape[0]
    assert upscaled[0][1].shape[1] > baseline[0][1].shape[1]


def test_visual_embedding_modes_are_available():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return

    crop = np.zeros((24, 12, 3), dtype=np.uint8)
    crop[:, :6] = (255, 255, 255)

    legacy = extract_visual_from_crop(crop, mode="hsv")
    rich = extract_visual_from_crop(crop, mode="hsv_lab_gradient")

    assert normalize_embedding_mode("legacy") == "hsv"
    assert normalize_embedding_mode("rich") == "hsv_lab_gradient"
    assert legacy is not None and len(legacy) == 30
    assert rich is not None and len(rich) == 42


def test_ocr_diagnostics_explain_direct_only_single_digit_rejection():
    raw = [
        {
            "source": "mmocr",
            "ocr_channel": "direct",
            "crop_path": "a.jpg",
            "variant": "mmocr_torso",
            "number": 6,
            "confidence": 0.9,
        }
    ]
    aggregated = aggregate_detections_by_crop(raw, min_raw_confidence=0.05)
    voted = vote_numbers(aggregated, min_raw_confidence=0.05)
    diagnostics = ocr_decision_diagnostics(raw, aggregated, voted, min_votes=2, min_raw_confidence=0.05)

    assert voted is None
    assert diagnostics["status"] == "only_direct_single_digit_candidates"
    assert diagnostics["voting_rejection_reasons"]["direct_only_single_digit"] == 1


def test_mmocr_score_uses_mean_character_confidence():
    assert abs(mmocr_score([0.8, 0.6]) - 0.7) < 1e-6
    assert mmocr_score(None) == 0.0


def test_ocr_skips_referee_candidate_rows():
    assert not is_ocr_player_row({"role_detection": "referee_candidate"})
    assert not is_ocr_player_row({"semantic_group_id": 5})
    assert is_ocr_player_row({"role_detection": "player", "semantic_group_id": 1})


def test_team_assigner_uses_roster_kit_colors_without_kmeans():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return

    frame = np.full((90, 100, 3), 80, dtype=np.uint8)
    frame[:, :50] = (0, 0, 0)
    frame[:, 50:] = (255, 255, 255)
    tracks = {
        "players": [
            {
                1: {"bbox": [5, 5, 45, 85], "role_detection": "player"},
                2: {"bbox": [55, 5, 95, 85], "role_detection": "player"},
            }
        ]
    }
    roster = [
        {
            "player_id": "team1_10",
            "team_id": 1,
            "role": "player",
            "metadata": {"team_kit_color": "black_blue"},
        },
        {
            "player_id": "team2_11",
            "team_id": 2,
            "role": "player",
            "metadata": {"team_kit_color": "white"},
        },
    ]

    assigner = TeamAssigner(
        min_seed_colors=99,
        min_tracklet_colors=1,
        color_ranges_by_team=team_color_ranges_by_team_from_roster(roster),
    )
    assignments = assigner.fit_apply([frame], tracks)

    assert assignments[1]["team"] == 1
    assert assignments[1]["source"] == "roster_color"
    assert assignments[2]["team"] == 2
    assert tracks["players"][0][1]["frame_team"] == 1
    assert tracks["players"][0][2]["frame_team"] == 2


def test_team_assigner_normalizes_kmeans_random_states():
    assert normalize_random_states(None) == [0, 42, 7, 13]
    assert normalize_random_states(5) == [5]
    assert normalize_random_states("1, 2,3") == [1, 2, 3]


def test_team_assigner_accepts_multiple_kmeans_initializations():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return

    assigner = TeamAssigner(
        min_seed_colors=2,
        min_tracklet_colors=1,
        min_cluster_separation=10.0,
        kmeans_random_states=[0, 42],
    )

    assigner._fit([(0, 0, 0), (255, 255, 255), (5, 5, 5), (250, 250, 250)])

    assert assigner.kmeans is not None
    assert sorted(assigner.team_colors) == [1, 2]


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
    assert filtered[1]["roster_filter"]["status"] == "promoted_alternative"


def test_roster_aware_ocr_distribution_does_not_promote_when_disabled():
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

    filtered, diagnostics = RosterAwareOCRFilter(
        roster,
        mode="degrade",
        promote_roster_candidate=False,
        confidence_scale=0.5,
    ).apply(assignments, rows)

    assert filtered[1]["jersey_number"] == 96
    assert filtered[1]["confidence"] == 0.25
    assert filtered[1]["raw_jersey_distribution"] == [
        {"jersey_number": 96, "confidence": 0.5, "votes": 5},
        {"jersey_number": 90, "confidence": 0.2, "votes": 2},
    ]
    assert filtered[1]["jersey_distribution"] == [{"jersey_number": 90, "confidence": 0.2, "votes": 2}]
    assert filtered[1]["jersey_roster_mass"] == 0.2
    assert filtered[1]["roster_filter"]["status"] == "degraded"


def test_roster_aware_ocr_preserves_dropped_candidate_evidence():
    roster = [{"player_id": "juve_21", "team_id": 2, "jersey_number": 21}]
    assignments = {
        45: {
            "jersey_number": 2,
            "confidence": 0.26,
            "votes": 12,
            "candidates": [
                {"jersey_number": 2, "confidence": 0.26, "votes": 12},
                {"jersey_number": 21, "confidence": 0.23, "votes": 8},
            ],
        }
    }
    rows = [{"display_track_id": 45, "track_id": 45, "team_id": 2}]

    filtered, diagnostics = RosterAwareOCRFilter(
        roster,
        mode="drop",
        promote_roster_candidate=False,
        preserve_dropped_evidence=True,
    ).apply(assignments, rows)

    assert filtered[45]["jersey_number"] is None
    assert filtered[45]["raw_jersey_distribution"] == [
        {"jersey_number": 2, "confidence": 0.26, "votes": 12},
        {"jersey_number": 21, "confidence": 0.23, "votes": 8},
    ]
    assert filtered[45]["jersey_distribution"] == [{"jersey_number": 21, "confidence": 0.23, "votes": 8}]
    assert filtered[45]["roster_filter"]["status"] == "dropped_preserved_evidence"
    assert diagnostics["dropped"]["45"]["reason"] == "number_not_in_team_roster"


def test_hungarian_uses_preserved_dropped_candidate_evidence():
    identifier = HungarianPlayerIdentifier(
        roster_path=None,
        unknown_threshold=0.0,
        reliable_jersey_min_candidate_score=0.20,
    )
    identifier.roster = [
        {"player_id": "juve_21", "name": "P21", "team_id": 2, "jersey_number": 21, "position_prior": None},
    ]
    summaries = [
        {
            "track_id": 45,
            "team_id": 2,
            "mean_team_confidence": 0.9,
            "jersey_number": None,
            "jersey_confidence": 0.0,
            "jersey_votes": 0,
            "raw_jersey_distribution": [
                {"jersey_number": 2, "confidence": 0.26, "votes": 12},
                {"jersey_number": 21, "confidence": 0.23, "votes": 8},
            ],
            "jersey_distribution": [{"jersey_number": 21, "confidence": 0.23, "votes": 8}],
            "preserved_dropped_jersey_evidence": True,
            "num_frames": 800,
            "mean_crop_quality": 0.4,
            "mean_pitch_position": None,
            "visual_embedding": None,
        }
    ]

    assignments, scores = identifier.assign(summaries)
    candidates = build_identity_candidates(
        scores,
        tracks={"players": [{45: {"display_track_id": 45, "player_id": "unknown"}}]},
        min_confidence=0.35,
        max_cost=0.85,
    )

    assert assignments[45]["player_id"] == "unknown"
    assert candidates[45]["candidate_player_id"] == "juve_21"
    assert scores[0]["jersey_score_source"] == "raw_jersey_distribution"
    assert scores[0]["assignment_gate"]["pass"] is False


def test_hungarian_summary_marks_preserved_dropped_evidence():
    identifier = HungarianPlayerIdentifier(roster_path=None)
    summaries = identifier.summarize(
        [
            {
                "frame": 0,
                "track_id": 45,
                "display_track_id": 45,
                "team_id": 2,
                "jersey_number": None,
                "jersey_roster_filter": '{"status": "dropped_preserved_evidence"}',
                "raw_jersey_distribution": [
                    {"jersey_number": 2, "confidence": 0.26, "votes": 12},
                    {"jersey_number": 21, "confidence": 0.23, "votes": 8},
                ],
                "jersey_distribution": [{"jersey_number": 21, "confidence": 0.23, "votes": 8}],
                "position_pitch": None,
                "visual_embedding": None,
                "bbox": [0, 0, 20, 40],
            }
        ]
    )

    assert summaries[0]["preserved_dropped_jersey_evidence"] is True
    assert summaries[0]["raw_jersey_distribution"][0]["jersey_number"] == 2


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


def test_jersey_ocr_segment_group_key_uses_frame_window():
    row_a = {"display_track_id": 7, "track_id": 99, "frame": 499}
    row_b = {"display_track_id": 7, "track_id": 99, "frame": 500}

    assert jersey_group_key(row_a, segment_frames=0) == 7
    assert jersey_group_key(row_a, segment_frames=500) == (7, 0)
    assert jersey_group_key(row_b, segment_frames=500) == (7, 1)
    assert diagnostic_key((7, 1)) == "7:1"


def test_hungarian_identity_tracklet_id_keeps_segment_assignments_local():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "p95", "name": "P95", "team_id": 1, "jersey_number": 95, "position_prior": None},
    ]
    rows = [
        {
            "frame": 10,
            "track_id": 1,
            "display_track_id": 4,
            "identity_tracklet_id": 400000,
            "team_id": 1,
            "team_confidence": 0.9,
            "jersey_number": 95,
            "jersey_confidence": 0.8,
            "jersey_head_confidence": 0.9,
            "jersey_winner_margin": 0.4,
            "jersey_votes": 8,
            "raw_jersey_distribution": [{"jersey_number": 95, "confidence": 0.8, "votes": 8}],
            "jersey_distribution": [{"jersey_number": 95, "confidence": 0.8, "votes": 8}],
            "crop_quality": 0.4,
            "role_detection": "player",
        },
        {
            "frame": 510,
            "track_id": 1,
            "display_track_id": 4,
            "identity_tracklet_id": 400001,
            "team_id": 1,
            "team_confidence": 0.9,
            "jersey_number": None,
            "jersey_votes": 0,
            "crop_quality": 0.4,
            "role_detection": "player",
        },
    ]
    summaries = identifier.summarize(rows)
    assignments, _ = identifier.assign(summaries)
    tracks = {"players": [{1: {"display_track_id": 4, "identity_tracklet_id": 400000}}, {1: {"display_track_id": 4, "identity_tracklet_id": 400001}}]}

    apply_assignments(tracks, assignments)

    assert tracks["players"][0][1]["player_id"] == "p95"
    assert tracks["players"][1][1]["player_id"] == "unknown"


def test_hungarian_prefers_raw_distribution_for_jersey_cost():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "p90", "name": "P90", "team_id": 1, "jersey_number": 90, "position_prior": None},
    ]
    tracklet = {
        "track_id": 9,
        "team_id": 1,
        "mean_team_confidence": 0.8,
        "jersey_number": 96,
        "jersey_confidence": 0.4,
        "jersey_votes": 5,
        "raw_jersey_distribution": [
            {"jersey_number": 96, "confidence": 0.70, "votes": 5},
            {"jersey_number": 90, "confidence": 0.10, "votes": 1},
        ],
        "jersey_distribution": [
            {"jersey_number": 90, "confidence": 0.90, "votes": 3},
        ],
        "num_frames": 40,
        "mean_crop_quality": 0.4,
        "mean_pitch_position": None,
        "visual_embedding": None,
    }

    details = identifier.cost_details(tracklet, identifier.roster[0])

    assert details["jersey_score_source"] == "raw_jersey_distribution"
    assert abs(details["components"]["jersey"] - -0.015) < 1e-9


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


def test_linker_team_gate_blocks_confident_team_mismatch():
    tracks = two_tracklets(
        previous={"team": 1, "team_confidence": 0.9, "visual_embedding": [1.0, 0.0]},
        current={"team": 2, "team_confidence": 0.9, "visual_embedding": [1.0, 0.0]},
    )
    linker = TrackletLinker(max_gap=10, max_distance=20, min_frames=1, team_gate_min_confidence=0.65)

    mapping = linker.apply(tracks)

    assert mapping[1] != mapping[2]
    assert linker.diagnostics["rejection_counts"]["team_gate"] == 1


def test_linker_appearance_gate_blocks_low_similarity():
    tracks = two_tracklets(
        previous={"team": 1, "team_confidence": 0.9, "visual_embedding": [1.0, 0.0]},
        current={"team": 1, "team_confidence": 0.9, "visual_embedding": [0.0, 1.0]},
    )
    linker = TrackletLinker(max_gap=10, max_distance=20, min_frames=1, appearance_min_similarity=0.72)

    mapping = linker.apply(tracks)

    assert mapping[1] != mapping[2]
    assert linker.diagnostics["rejection_counts"]["appearance_gate"] == 1


def test_linker_accepts_distance_gap_when_embeddings_missing():
    tracks = two_tracklets(
        previous={"team": 1, "team_confidence": 0.9},
        current={"team": 1, "team_confidence": 0.9},
    )
    linker = TrackletLinker(max_gap=10, max_distance=20, min_frames=1)

    mapping = linker.apply(tracks)

    assert mapping[1] == mapping[2]
    assert linker.diagnostics["accepted_links"][0]["from_track_id"] == 1


def test_jersey_identity_linker_links_reliable_same_number_continuation():
    tracks = two_tracklets(
        previous={
            "display_track_id": 9,
            "team": 1,
            "team_confidence": 0.9,
            "jersey_number": 6,
            "jersey_votes": 5,
            "jersey_evidence": {"confidence": 0.82, "head_confidence": 0.90, "winner_margin": 0.35, "votes": 5},
        },
        current={
            "display_track_id": 59,
            "team": 1,
            "team_confidence": 0.9,
            "jersey_number": 6,
            "jersey_votes": 5,
            "jersey_evidence": {"confidence": 0.78, "head_confidence": 0.88, "winner_margin": 0.32, "votes": 5},
        },
    )
    rows = [
        {"track_id": 1, "display_track_id": 9, "frame": 0},
        {"track_id": 2, "display_track_id": 59, "frame": 3},
    ]

    diagnostics = JerseyIdentityLinker(max_gap=10, max_distance=20, min_frames=1).apply(tracks, rows=rows)

    assert diagnostics["accepted_links"][0]["from_display_track_id"] == 9
    assert diagnostics["accepted_links"][0]["to_display_track_id"] == 59
    assert tracks["players"][3][2]["display_track_id"] == 9
    assert rows[1]["display_track_id"] == 9
    assert rows[1]["jersey_link_previous_display_track_id"] == 59


def test_jersey_identity_linker_rejects_far_same_number_tracklet():
    tracks = two_tracklets(
        previous={
            "display_track_id": 9,
            "team": 1,
            "team_confidence": 0.9,
            "jersey_number": 6,
            "jersey_votes": 5,
            "jersey_evidence": {"confidence": 0.82, "head_confidence": 0.90, "winner_margin": 0.35, "votes": 5},
            "position": (0.0, 0.0),
        },
        current={
            "display_track_id": 59,
            "team": 1,
            "team_confidence": 0.9,
            "jersey_number": 6,
            "jersey_votes": 5,
            "jersey_evidence": {"confidence": 0.78, "head_confidence": 0.88, "winner_margin": 0.32, "votes": 5},
            "position": (300.0, 300.0),
        },
    )

    diagnostics = JerseyIdentityLinker(max_gap=10, max_distance=20, min_frames=1).apply(tracks)

    assert diagnostics["accepted_links"] == []
    assert tracks["players"][3][2]["display_track_id"] == 59
    assert diagnostics["rejection_counts"]["distance"] >= 1


def test_strongsort_core_uses_appearance_to_keep_identity():
    tracker = StrongSortTrackerCore(
        min_confidence=0.1,
        max_age=5,
        min_hits=1,
        max_center_distance=80.0,
        max_cost=0.90,
        iou_weight=0.20,
        appearance_weight=0.70,
        center_weight=0.10,
        appearance_min_similarity=0.0,
    )
    first = tracker.update(
        [
            Detection(np_box(0, 0, 20, 40), 0.9, "person", [1.0, 0.0]),
            Detection(np_box(80, 0, 100, 40), 0.9, "person", [0.0, 1.0]),
        ]
    )
    ids_by_embedding = {tuple(track.embedding): track.track_id for track in first}

    second = tracker.update(
        [
            Detection(np_box(76, 0, 96, 40), 0.9, "person", [1.0, 0.0]),
            Detection(np_box(4, 0, 24, 40), 0.9, "person", [0.0, 1.0]),
        ]
    )

    assert {tuple(track.embedding): track.track_id for track in second} == ids_by_embedding


def test_position_prior_is_capped():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0, position_prior_max_cost=0.08)
    tracklet = {
        "track_id": 1,
        "team_id": 1,
        "mean_team_confidence": 0.8,
        "jersey_number": None,
        "num_frames": 40,
        "mean_crop_quality": 0.4,
        "mean_pitch_position": [0.0, 0.0],
        "visual_embedding": None,
    }
    player = {"player_id": "p", "team_id": 1, "jersey_number": None, "position_prior": [1000.0, 0.0]}

    details = identifier.cost_details(tracklet, player)

    assert details["components"]["position_prior"] == 0.08
    assert details["position_prior_distance"] == 1000.0


def test_reliable_jersey_beats_position_prior():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0, position_prior_max_cost=0.08)
    tracklet = {
        "track_id": 4,
        "team_id": 1,
        "mean_team_confidence": 0.8,
        "jersey_number": 7,
        "jersey_confidence": 0.9,
        "jersey_votes": 5,
        "num_frames": 40,
        "mean_crop_quality": 0.4,
        "mean_pitch_position": [0.0, 0.0],
        "visual_embedding": None,
    }
    correct_far = {"player_id": "p7", "team_id": 1, "jersey_number": 7, "position_prior": [1000.0, 0.0]}
    wrong_near = {"player_id": "p9", "team_id": 1, "jersey_number": 9, "position_prior": [0.0, 0.0]}

    assert identifier.cost_details(tracklet, correct_far)["cost"] < identifier.cost_details(tracklet, wrong_near)["cost"]


def test_assignment_gate_blocks_weak_non_jersey_assignment():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {
            "player_id": "p10",
            "name": "P10",
            "team_id": 1,
            "jersey_number": 10,
            "position_prior": [40.0, 30.0],
            "visual_embedding": None,
        }
    ]
    summaries = [
        {
            "track_id": 10,
            "team_id": 1,
            "mean_team_confidence": 0.9,
            "jersey_number": None,
            "jersey_confidence": 0.0,
            "jersey_votes": 0,
            "num_frames": 80,
            "mean_crop_quality": 0.5,
            "mean_pitch_position": [40.0, 30.0],
            "visual_embedding": None,
        }
    ]

    assignments, _ = identifier.assign(summaries)

    assert assignments[10]["player_id"] == "unknown"
    assert assignments[10]["evidence"]["assignment_gate"]["reason"] == "insufficient_assignment_evidence"


def test_assignment_gate_allows_strong_team_visual_trajectory():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {
            "player_id": "p10",
            "name": "P10",
            "team_id": 1,
            "jersey_number": 10,
            "position_prior": [40.0, 30.0],
            "visual_embedding": [1.0, 0.0],
        }
    ]
    summaries = [
        {
            "track_id": 10,
            "team_id": 1,
            "mean_team_confidence": 0.9,
            "jersey_number": None,
            "jersey_confidence": 0.0,
            "jersey_votes": 0,
            "num_frames": 80,
            "mean_crop_quality": 0.5,
            "mean_pitch_position": [42.0, 31.0],
            "visual_embedding": [0.99, 0.01],
        }
    ]

    assignments, _ = identifier.assign(summaries)

    assert assignments[10]["player_id"] == "p10"
    assert assignments[10]["evidence"]["assignment_gate"]["reason"] == "strong_team_visual_trajectory"


def test_assignment_gate_allows_singleton_goalkeeper_from_palette():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {
            "player_id": "gk1",
            "name": "GK1",
            "team_id": 1,
            "jersey_number": 1,
            "role": "goalkeeper",
            "position_prior": [18.0, 34.0],
        },
        {
            "player_id": "p6",
            "name": "P6",
            "team_id": 1,
            "jersey_number": 6,
            "role": "player",
            "position_prior": [35.0, 34.0],
        },
    ]
    summaries = [
        {
            "track_id": 6,
            "team_id": 1,
            "mean_team_confidence": 1.0,
            "role_detection": "goalkeeper",
            "semantic_group_id": 3,
            "goalkeeper_palette_match": True,
            "goalkeeper_like_score": 0.58,
            "goalkeeper_like_team": 1,
            "jersey_number": 1,
            "jersey_confidence": 0.23,
            "jersey_votes": 28,
            "jersey_distribution": [
                {"jersey_number": 4, "confidence": 0.25, "votes": 31},
                {"jersey_number": 1, "confidence": 0.42, "votes": 28},
                {"jersey_number": 6, "confidence": 0.36, "votes": 14},
            ],
            "num_frames": 292,
            "mean_crop_quality": 0.25,
            "mean_pitch_position": [18.0, 34.0],
            "visual_embedding": None,
        }
    ]

    assignments, _ = identifier.assign(summaries)

    assert assignments[6]["player_id"] == "gk1"
    assert assignments[6]["evidence"]["assignment_gate"]["reason"] == "goalkeeper_roster_singleton"


def test_assignment_gate_blocks_singleton_goalkeeper_when_roster_has_two_goalkeepers():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"},
        {"player_id": "gk2", "team_id": 1, "jersey_number": 12, "role": "goalkeeper"},
    ]
    summaries = [
        {
            "track_id": 6,
            "team_id": 1,
            "mean_team_confidence": 1.0,
            "role_detection": "goalkeeper",
            "semantic_group_id": 3,
            "goalkeeper_palette_match": True,
            "goalkeeper_like_score": 0.58,
            "goalkeeper_like_team": 1,
            "jersey_number": None,
            "jersey_confidence": 0.0,
            "jersey_votes": 0,
            "num_frames": 292,
            "mean_crop_quality": 0.25,
            "mean_pitch_position": None,
            "visual_embedding": None,
        }
    ]

    assignments, _ = identifier.assign(summaries)

    assert assignments[6]["player_id"] == "unknown"
    assert not assignments[6]["evidence"]["assignment_gate"]["goalkeeper_singleton"]


def test_assignment_gate_blocks_weak_roster_promoted_jersey_candidate():
    identifier = HungarianPlayerIdentifier(roster_path=None, unknown_threshold=0.0)
    identifier.roster = [
        {"player_id": "team2_04", "name": "P4", "team_id": 2, "jersey_number": 4, "role": "player"},
    ]
    summaries = [
        {
            "track_id": 11,
            "team_id": 2,
            "mean_team_confidence": 0.99,
            "jersey_number": 4,
            "jersey_confidence": 0.20,
            "jersey_votes": 24,
            "jersey_distribution": [
                {"jersey_number": 4, "confidence": 0.75, "votes": 24},
            ],
            "jersey_raw_candidates": [
                {"jersey_number": 1, "confidence": 0.42, "votes": 45},
                {"jersey_number": 4, "confidence": 0.20, "votes": 24},
            ],
            "num_frames": 1900,
            "mean_crop_quality": 0.4,
            "mean_pitch_position": None,
            "visual_embedding": None,
        }
    ]

    assignments, _ = identifier.assign(summaries)

    assert assignments[11]["player_id"] == "unknown"
    assert assignments[11]["evidence"]["assignment_gate"]["reason"] == "insufficient_assignment_evidence"


def test_duplicate_jersey_rank_prefers_stronger_ocr_over_identity_confidence():
    weak_promoted = {
        "identity_confidence": 1.0,
        "jersey_evidence": {"confidence": 0.20, "votes": 24, "head_confidence": 0.0, "winner_margin": 0.0},
        "crop_quality": 0.4,
        "bbox": [0, 0, 20, 60],
    }
    strong_ocr = {
        "identity_confidence": 0.8,
        "jersey_evidence": {"confidence": 0.62, "votes": 5, "head_confidence": 0.75, "winner_margin": 0.18},
        "crop_quality": 0.3,
        "bbox": [0, 0, 20, 60],
    }

    assert jersey_rank(strong_ocr) > jersey_rank(weak_promoted)


def test_constraints_clear_duplicate_player_id_in_same_frame():
    roster = [{"player_id": "p7", "team_id": 1, "jersey_number": 7}]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 1,
                    "player_id": "p7",
                    "player_name": "P7",
                    "identity_confidence": 0.9,
                    "team": 1,
                    "jersey_number": 7,
                    "bbox": [0, 0, 20, 40],
                },
                2: {
                    "display_track_id": 2,
                    "player_id": "p7",
                    "player_name": "P7",
                    "identity_confidence": 0.5,
                    "team": 1,
                    "jersey_number": 7,
                    "bbox": [30, 0, 50, 40],
                },
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    assert tracks["players"][0][1]["player_id"] == "p7"
    assert tracks["players"][0][2]["player_id"] == "unknown"
    assert diagnostics["duplicate_player_id_count"] == 1


def test_constraints_clear_invalid_team_jersey():
    roster = [{"player_id": "p7", "team_id": 1, "jersey_number": 7}]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 1,
                    "player_id": "p96",
                    "player_name": "P96",
                    "identity_confidence": 0.8,
                    "team": 1,
                    "jersey_number": 96,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 3,
                    "bbox": [0, 0, 20, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    track = tracks["players"][0][1]
    assert track["jersey_number"] is None
    assert track["player_id"] == "unknown"
    assert diagnostics["invalid_team_jersey_count"] == 1


def test_constraints_clear_duplicate_team_jersey_in_same_frame():
    roster = [
        {"player_id": "gk1", "team_id": 1, "jersey_number": 1},
        {"player_id": "p7", "team_id": 1, "jersey_number": 7},
    ]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 1,
                    "team": 1,
                    "jersey_number": 1,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 4,
                    "jersey_evidence": {"confidence": 0.9, "votes": 4},
                    "bbox": [0, 0, 20, 40],
                },
                2: {
                    "display_track_id": 2,
                    "team": 1,
                    "jersey_number": 1,
                    "jersey_confidence": 0.2,
                    "jersey_votes": 1,
                    "jersey_evidence": {"confidence": 0.2, "votes": 1},
                    "bbox": [30, 0, 50, 40],
                },
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    assert tracks["players"][0][1]["jersey_number"] == 1
    assert tracks["players"][0][2]["jersey_number"] is None
    assert diagnostics["duplicate_team_jersey_count"] == 1


def test_constraints_preserve_same_player_global_team_jersey_fragments():
    roster = [{"player_id": "team1_06", "team_id": 1, "jersey_number": 6, "role": "player"}]
    weak_early = {
        "display_track_id": 9,
        "team": 1,
        "player_id": "team1_06",
        "player_name": "P6",
        "identity_confidence": 1.0,
        "jersey_number": 6,
        "jersey_confidence": 0.24,
        "jersey_votes": 12,
        "jersey_evidence": {"confidence": 0.24, "votes": 12, "head_confidence": 0.30, "winner_margin": 0.02},
        "bbox": [0, 0, 20, 40],
    }
    strong_late = {
        "display_track_id": 59,
        "team": 1,
        "player_id": "team1_06",
        "player_name": "P6",
        "identity_confidence": 0.6,
        "jersey_number": 6,
        "jersey_confidence": 0.68,
        "jersey_votes": 6,
        "jersey_evidence": {"confidence": 0.68, "votes": 6, "head_confidence": 0.82, "winner_margin": 0.31},
        "bbox": [30, 0, 50, 40],
    }
    tracks = {
        "players": [
            {9: dict(weak_early)},
            {9: dict(weak_early)},
            {59: dict(strong_late)},
            {59: dict(strong_late)},
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    assert tracks["players"][0][9]["jersey_number"] == 6
    assert tracks["players"][0][9]["player_id"] == "team1_06"
    assert tracks["players"][2][59]["jersey_number"] == 6
    assert tracks["players"][2][59]["player_id"] == "team1_06"
    assert diagnostics["global_team_jersey_owner_count"] == 0


def test_constraints_clear_competing_global_team_jersey_owner():
    roster = [
        {"player_id": "team1_06", "team_id": 1, "jersey_number": 6, "role": "player"},
        {"player_id": "team1_alt06", "team_id": 1, "jersey_number": 6, "role": "player"},
    ]
    weak_early = {
        "display_track_id": 9,
        "team": 1,
        "player_id": "team1_alt06",
        "player_name": "Alt P6",
        "identity_confidence": 0.9,
        "jersey_number": 6,
        "jersey_confidence": 0.24,
        "jersey_votes": 12,
        "jersey_evidence": {"confidence": 0.24, "votes": 12, "head_confidence": 0.30, "winner_margin": 0.02},
        "bbox": [0, 0, 20, 40],
    }
    strong_late = {
        "display_track_id": 59,
        "team": 1,
        "player_id": "team1_06",
        "player_name": "P6",
        "identity_confidence": 0.6,
        "jersey_number": 6,
        "jersey_confidence": 0.68,
        "jersey_votes": 6,
        "jersey_evidence": {"confidence": 0.68, "votes": 6, "head_confidence": 0.82, "winner_margin": 0.31},
        "bbox": [30, 0, 50, 40],
    }
    tracks = {
        "players": [
            {9: dict(weak_early)},
            {9: dict(weak_early)},
            {59: dict(strong_late)},
            {59: dict(strong_late)},
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    assert tracks["players"][0][9]["jersey_number"] is None
    assert tracks["players"][0][9]["player_id"] == "unknown"
    assert tracks["players"][1][9]["jersey_constraint"]["reason"] == "global_duplicate_team_jersey_owner"
    assert tracks["players"][2][59]["jersey_number"] == 6
    assert tracks["players"][2][59]["player_id"] == "team1_06"
    assert diagnostics["global_team_jersey_owner_count"] == 1
    assert diagnostics["global_team_jersey_owners"][0]["kept_display_track_id"] == 59
    assert diagnostics["global_team_jersey_owners"][0]["cleared_display_track_id"] == 9


def test_constraints_clear_unknown_global_team_jersey_competitor():
    roster = [{"player_id": "team1_06", "team_id": 1, "jersey_number": 6, "role": "player"}]
    unknown_early = {
        "display_track_id": 9,
        "team": 1,
        "player_id": "unknown",
        "identity_confidence": 0.0,
        "jersey_number": 6,
        "jersey_confidence": 0.68,
        "jersey_votes": 6,
        "jersey_evidence": {"confidence": 0.68, "votes": 6, "head_confidence": 0.82, "winner_margin": 0.31},
        "bbox": [0, 0, 20, 40],
    }
    known_late = {
        "display_track_id": 59,
        "team": 1,
        "player_id": "team1_06",
        "player_name": "P6",
        "identity_confidence": 0.6,
        "jersey_number": 6,
        "jersey_confidence": 0.24,
        "jersey_votes": 12,
        "jersey_evidence": {"confidence": 0.24, "votes": 12, "head_confidence": 0.30, "winner_margin": 0.02},
        "bbox": [30, 0, 50, 40],
    }
    tracks = {
        "players": [
            {9: dict(unknown_early)},
            {9: dict(unknown_early)},
            {59: dict(known_late)},
            {59: dict(known_late)},
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    assert tracks["players"][0][9]["jersey_number"] is None
    assert tracks["players"][2][59]["jersey_number"] == 6
    assert tracks["players"][2][59]["player_id"] == "team1_06"
    assert diagnostics["global_team_jersey_owner_count"] == 1
    assert diagnostics["global_team_jersey_owners"][0]["kept_player_id"] == "team1_06"
    assert diagnostics["global_team_jersey_owners"][0]["cleared_player_id"] == "unknown"


def test_constraints_clear_goalkeeper_only_jersey_on_non_goalkeeper():
    roster = [
        {"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"},
        {"player_id": "p7", "team_id": 1, "jersey_number": 7, "role": "player"},
    ]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 1,
                    "team": 1,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 1,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 4,
                    "jersey_evidence": {"confidence": 0.9, "votes": 4},
                    "bbox": [0, 0, 20, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    assert tracks["players"][0][1]["jersey_number"] is None
    assert diagnostics["goalkeeper_only_jersey_count"] == 1


def test_constraints_can_fallback_from_goalkeeper_only_jersey_to_alternate_candidate():
    roster = [
        {"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"},
        {"player_id": "p8", "team_id": 1, "jersey_number": 8, "role": "player"},
    ]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 1,
                    "team": 1,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "player_id": "gk1",
                    "player_name": "GK",
                    "identity_confidence": 0.7,
                    "jersey_number": 1,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 4,
                    "jersey_evidence": {"confidence": 0.9, "votes": 4},
                    "raw_jersey_distribution": [
                        {"jersey_number": 1, "confidence": 0.90, "votes": 4},
                        {"jersey_number": 8, "confidence": 0.20, "votes": 2},
                    ],
                    "bbox": [0, 0, 20, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(
        tracks,
        roster,
        goalkeeper_only_alternate_enabled=True,
        goalkeeper_only_alternate_min_confidence=0.10,
        goalkeeper_only_alternate_min_votes=1,
    )

    track = tracks["players"][0][1]
    assert track["player_id"] == "unknown"
    assert track["jersey_number"] == 8
    assert track["jersey_evidence"]["reason"] == "goalkeeper_only_alternate_jersey"
    assert track["jersey_constraint"]["status"] == "fallback_applied"
    assert diagnostics["goalkeeper_only_jersey_count"] == 1
    assert diagnostics["goalkeeper_only_jersey_alternate_count"] == 1


def test_constraints_do_not_fallback_to_alternate_with_known_global_owner():
    roster = [
        {"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"},
        {"player_id": "p4", "team_id": 1, "jersey_number": 4, "role": "player"},
        {"player_id": "p6", "team_id": 1, "jersey_number": 6, "role": "player"},
    ]
    blocked = {
        "display_track_id": 8,
        "team": 1,
        "semantic_group_id": 1,
        "role_detection": "player",
        "player_id": "unknown",
        "jersey_number": 1,
        "jersey_confidence": 0.9,
        "jersey_votes": 4,
        "raw_jersey_distribution": [
            {"jersey_number": 1, "confidence": 0.90, "votes": 4},
            {"jersey_number": 4, "confidence": 0.35, "votes": 7},
            {"jersey_number": 6, "confidence": 0.12, "votes": 3},
        ],
        "bbox": [0, 0, 20, 40],
    }
    known_owner = {
        "display_track_id": 23,
        "team": 1,
        "semantic_group_id": 1,
        "role_detection": "player",
        "player_id": "p4",
        "player_name": "P4",
        "identity_confidence": 0.9,
        "jersey_number": 4,
        "jersey_confidence": 1.0,
        "jersey_votes": 2,
        "bbox": [30, 0, 50, 40],
    }
    tracks = {"players": [{8: dict(blocked)}, {23: dict(known_owner)}]}

    diagnostics = enforce_identity_constraints(
        tracks,
        roster,
        goalkeeper_only_alternate_enabled=True,
        goalkeeper_only_alternate_min_confidence=0.08,
        goalkeeper_only_alternate_min_votes=1,
        goalkeeper_only_alternate_block_known_owner=True,
    )

    track = tracks["players"][0][8]
    assert track["jersey_number"] is None
    assert track["player_id"] == "unknown"
    assert tracks["players"][1][23]["jersey_number"] == 4
    assert diagnostics["goalkeeper_only_jersey_alternate_count"] == 0
    assert diagnostics["goalkeeper_only_jersey_alternate_rejection_count"] == 1
    assert diagnostics["goalkeeper_only_jersey_alternate_rejections"][0]["reason"] == "alternate_known_owner_conflict"


def test_constraints_clear_non_goalkeeper_jersey_on_goalkeeper():
    roster = [
        {"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"},
        {"player_id": "p7", "team_id": 1, "jersey_number": 7, "role": "player"},
    ]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 38,
                    "team": 1,
                    "semantic_group_id": 3,
                    "role_detection": "goalkeeper",
                    "jersey_number": 7,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 4,
                    "jersey_evidence": {"confidence": 0.9, "votes": 4},
                    "bbox": [0, 0, 20, 40],
                },
                2: {
                    "display_track_id": 7,
                    "team": 1,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 7,
                    "jersey_confidence": 0.8,
                    "jersey_votes": 3,
                    "jersey_evidence": {"confidence": 0.8, "votes": 3},
                    "bbox": [30, 0, 50, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    track = tracks["players"][0][1]
    assert track["jersey_number"] is None
    assert tracks["players"][0][2]["jersey_number"] == 7
    assert track["jersey_constraint"]["reason"] == "non_goalkeeper_jersey_on_goalkeeper"
    assert diagnostics["goalkeeper_invalid_jersey_count"] == 1
    assert diagnostics["duplicate_team_jersey_count"] == 0


def test_constraints_correct_goalkeeper_semantic_group_from_roster():
    roster = [{"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"}]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 1,
                    "player_id": "gk1",
                    "player_name": "GK1",
                    "identity_confidence": 0.8,
                    "team": 1,
                    "jersey_number": 1,
                    "role_detection": "player",
                    "goalkeeper_palette_match": True,
                    "semantic_group_id": 1,
                    "semantic_group": "team1_players",
                    "bbox": [0, 0, 20, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    track = tracks["players"][0][1]
    assert track["semantic_group_id"] == 3
    assert track["semantic_group"] == "team1_goalkeeper"
    assert diagnostics["semantic_group_correction_count"] == 1


def test_constraints_clear_goalkeeper_identity_without_goalkeeper_evidence():
    roster = [{"player_id": "gk1", "team_id": 1, "jersey_number": 1, "role": "goalkeeper"}]
    tracks = {
        "players": [
            {
                1: {
                    "display_track_id": 39,
                    "player_id": "gk1",
                    "player_name": "GK1",
                    "identity_confidence": 0.8,
                    "team": 1,
                    "jersey_number": 1,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 4,
                    "jersey_evidence": {"confidence": 0.9, "votes": 4},
                    "role_detection": "player",
                    "goalkeeper_palette_match": False,
                    "semantic_group_id": 1,
                    "semantic_group": "team1_players",
                    "bbox": [0, 0, 20, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    track = tracks["players"][0][1]
    assert track["player_id"] == "unknown"
    assert track["jersey_number"] is None
    assert track["semantic_group_id"] == 1
    assert diagnostics["goalkeeper_only_jersey_count"] == 1


def test_constraints_clear_identity_on_strong_frame_team_conflict():
    roster = [
        {"player_id": "roma_92", "team_id": 1, "jersey_number": 92, "role": "player"},
        {"player_id": "verona_38", "team_id": 2, "jersey_number": 38, "role": "player"},
    ]
    tracks = {
        "players": [
            {
                23: {
                    "display_track_id": 23,
                    "player_id": "roma_92",
                    "player_name": "Roma 92",
                    "identity_confidence": 0.85,
                    "team": 1,
                    "team_confidence": 0.9,
                    "frame_team": 2,
                    "frame_team_confidence": 0.82,
                    "frame_team_margin": 24.0,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 92,
                    "jersey_confidence": 0.8,
                    "jersey_votes": 4,
                    "jersey_evidence": {"confidence": 0.8, "votes": 4},
                    "bbox": [0, 0, 20, 40],
                }
            }
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster)

    track = tracks["players"][0][23]
    assert track["team"] == 2
    assert track["semantic_group_id"] == 2
    assert track["jersey_number"] is None
    assert track["player_id"] == "unknown"
    assert track["jersey_constraint"]["reason"] == "frame_team_conflict"
    assert diagnostics["frame_team_conflict_count"] == 1


def test_constraints_split_persistent_frame_team_conflict_segment():
    roster = [{"player_id": "roma_92", "team_id": 1, "jersey_number": 92, "role": "player"}]
    tracks = {
        "players": [
            {
                23: {
                    "display_track_id": 23,
                    "team": 1,
                    "team_confidence": 0.9,
                    "frame_team": 2,
                    "frame_team_confidence": 0.9,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 92,
                    "player_id": "roma_92",
                    "bbox": [0, 0, 20, 40],
                }
            },
            {
                23: {
                    "display_track_id": 23,
                    "team": 1,
                    "team_confidence": 0.9,
                    "frame_team": 2,
                    "frame_team_confidence": 0.9,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 92,
                    "player_id": "roma_92",
                    "bbox": [0, 0, 20, 40],
                }
            },
            {
                23: {
                    "display_track_id": 23,
                    "team": 1,
                    "team_confidence": 0.9,
                    "frame_team": 1,
                    "frame_team_confidence": 0.9,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 92,
                    "player_id": "roma_92",
                    "bbox": [0, 0, 20, 40],
                }
            },
        ]
    }

    diagnostics = enforce_identity_constraints(tracks, roster, frame_team_split_min_frames=2)

    new_id = tracks["players"][0][23]["display_track_id"]
    assert new_id != 23
    assert tracks["players"][1][23]["display_track_id"] == new_id
    assert tracks["players"][2][23]["display_track_id"] == 23
    assert tracks["players"][0][23]["previous_display_track_id"] == 23
    assert diagnostics["display_track_split_count"] == 1


def test_constraints_split_merges_short_non_conflict_bridge():
    roster = [{"player_id": "roma_92", "team_id": 1, "jersey_number": 92, "role": "player"}]
    players = []
    for frame in range(8):
        if frame in {0, 1, 4, 5, 6, 7}:
            frame_team = 2
        else:
            frame_team = 1
        players.append(
            {
                23: {
                    "display_track_id": 23,
                    "team": 1,
                    "team_confidence": 0.9,
                    "frame_team": frame_team,
                    "frame_team_confidence": 0.9,
                    "semantic_group_id": 1,
                    "role_detection": "player",
                    "jersey_number": 92,
                    "jersey_confidence": 0.9,
                    "jersey_votes": 4,
                    "player_id": "roma_92",
                    "bbox": [0, 0, 20, 40],
                }
            }
        )
    tracks = {"players": players}

    diagnostics = enforce_identity_constraints(
        tracks,
        roster,
        frame_team_split_min_frames=2,
        frame_team_split_max_gap=2,
    )

    new_id = tracks["players"][0][23]["display_track_id"]
    assert new_id != 23
    assert all(tracks["players"][frame][23]["display_track_id"] == new_id for frame in range(8))
    assert tracks["players"][2][23]["jersey_number"] is None
    assert tracks["players"][2][23]["player_id"] == "unknown"
    assert tracks["players"][2][23]["jersey_constraint"]["reason"] == "persistent_frame_team_conflict_bridge"
    assert diagnostics["display_track_split_count"] == 1
    assert diagnostics["display_track_splits"][0]["bridge_frames"] == 2


def test_referee_color_ranges_can_come_from_roster_metadata():
    roster = [
        {
            "player_id": "ref_yellow",
            "team_id": None,
            "jersey_number": None,
            "role": "referee",
            "metadata": {"kit_color": "yellow"},
        }
    ]

    ranges = referee_color_ranges_from_roster(roster)

    assert "roster_yellow" in ranges
    assert ranges["roster_yellow"]


def test_goalkeeper_color_ranges_can_come_from_roster_metadata():
    roster = [
        {
            "player_id": "team1_gk",
            "team_id": 1,
            "jersey_number": 1,
            "role": "goalkeeper",
            "metadata": {"kit_color": "black"},
        }
    ]

    ranges = goalkeeper_color_ranges_by_team_from_roster(roster)

    assert 1 in ranges
    assert "team1_goalkeeper_black" in ranges[1]


def test_goalkeeper_color_ranges_support_custom_colour_names():
    roster = [
        {
            "player_id": "milan_gk",
            "team_id": 1,
            "jersey_number": 16,
            "role": "goalkeeper",
            "metadata": {"kit_color": "orange"},
        },
        {
            "player_id": "atalanta_gk",
            "team_id": 2,
            "jersey_number": 29,
            "role": "goalkeeper",
            "metadata": {"kit_color": "fluorescent_yellow"},
        },
        {
            "player_id": "monza_gk",
            "team_id": 3,
            "jersey_number": 16,
            "role": "goalkeeper",
            "metadata": {"kit_color": "blue"},
        },
    ]

    ranges = goalkeeper_color_ranges_by_team_from_roster(roster)

    assert "team1_goalkeeper_orange" in ranges[1]
    assert "team2_goalkeeper_fluorescent_yellow" in ranges[2]
    assert "team3_goalkeeper_blue" in ranges[3]


def test_goalkeeper_roster_colour_can_correct_team_when_confident():
    assigner = GoalkeeperAppearanceAssigner(assign_team_from_color=True, team_correction_min_score=0.55)
    track = {"team": 2, "team_confidence": 0.7}
    assigner._apply_team_from_color(
        track,
        {"team_id": 1, "score": 0.58, "color": "team1_goalkeeper_light_blue"},
    )

    assert track["team"] == 1
    assert track["team_evidence"]["source"] == "goalkeeper_roster_color"


def test_overlay_hides_assigned_jersey_without_ocr_evidence():
    label = player_label(
        1,
        {
            "display_track_id": 8,
            "semantic_group_id": 1,
            "team": 1,
            "jersey_number": 10,
            "player_id": "team1_10",
            "identity_confidence": 0.95,
        },
    )

    assert label == "T1 ID8"


def test_overlay_shows_reliable_ocr_jersey():
    label = player_label(
        1,
        {
            "display_track_id": 8,
            "semantic_group_id": 1,
            "team": 1,
            "jersey_number": 10,
            "jersey_evidence": {"confidence": 0.8, "votes": 3},
            "player_id": "unknown",
            "identity_confidence": 0.0,
        },
    )

    assert label == "T1 #10 ID8"


def test_overlay_can_show_high_confidence_player_id_when_enabled():
    label = player_label(
        1,
        {
            "display_track_id": 8,
            "semantic_group_id": 1,
            "team": 1,
            "jersey_number": 10,
            "jersey_evidence": {"confidence": 0.8, "votes": 3},
            "player_id": "team1_10",
            "identity_confidence": 0.95,
        },
        config={"show_player_id": True},
    )

    assert label == "T1 #10 team1_10"


def test_overlay_shows_stable_low_mass_ocr_jersey():
    label = player_label(
        1,
        {
            "display_track_id": 2,
            "semantic_group_id": 2,
            "team": 2,
            "jersey_number": 38,
            "jersey_evidence": {"confidence": 0.425, "winner_margin": 0.226, "votes": 192},
            "player_id": "unknown",
            "identity_confidence": 0.0,
        },
    )

    assert label == "T2 #38 ID2"


def test_overlay_shows_head_confident_crop_aggregated_jersey():
    label = player_label(
        1,
        {
            "display_track_id": 2,
            "semantic_group_id": 2,
            "team": 2,
            "jersey_number": 38,
            "jersey_evidence": {
                "confidence": 0.277,
                "head_confidence": 0.72,
                "winner_margin": 0.169,
                "votes": 12,
            },
            "player_id": "unknown",
            "identity_confidence": 0.0,
        },
    )

    assert label == "T2 #38 ID2"


def test_overlay_can_show_any_ocr_winner_for_diagnostics():
    label = player_label(
        1,
        {
            "display_track_id": 5,
            "semantic_group_id": 2,
            "team": 2,
            "jersey_number": 17,
            "jersey_evidence": {"confidence": 0.05, "head_confidence": 0.20, "votes": 1},
            "player_id": "unknown",
            "identity_confidence": 0.0,
        },
        config={"show_jersey_winner": True},
    )

    assert label == "T2 #17 ID5"


def test_overlay_labels_referee_candidate_without_question_mark():
    label = referee_label(
        {"referee_like_color": "roster_yellow", "referee_like_score": 0.31},
        prefix="ref",
    )

    assert label == "ref roster_yellow 0.31"


def two_tracklets(previous, current):
    base_previous = {"bbox": [0, 0, 10, 20], "position": (10.0, 10.0)}
    base_previous.update(previous)
    base_current = {"bbox": [2, 0, 12, 20], "position": (12.0, 10.0)}
    base_current.update(current)
    return {
        "players": [
            {1: dict(base_previous)},
            {1: dict(base_previous)},
            {},
            {2: dict(base_current)},
            {2: dict(base_current)},
        ]
    }


def np_box(x1, y1, x2, y2):
    import numpy as np

    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


if __name__ == "__main__":
    test_hungarian_assignment_prefers_jersey_over_uncertain_team()
    test_unique_team_jersey_constraint_rejects_duplicate_roster_numbers()
    test_reliable_jersey_blocks_same_team_wrong_number()
    test_jersey_numbers_start_at_one()
    test_number_one_is_soft_goalkeeper_prior()
    test_ocr_vote_requires_raw_confidence_filter()
    test_ocr_digit_confusion_override_prefers_close_alternative()
    test_ocr_digit_confusion_override_keeps_strong_winner()
    test_ocr_template_votes_are_weighted_signal()
    test_template_prefers_two_digit_candidates_over_digit_fragments()
    test_ocr_aggregates_variants_by_crop_before_voting()
    test_ocr_crop_quality_weighting_downweights_noisy_crops()
    test_mmocr_backend_alias_keeps_easyocr_fallback()
    test_paddleocr_backend_aliases_are_opt_in()
    test_ocr_super_resolution_variants_can_upscale_small_crop()
    test_visual_embedding_modes_are_available()
    test_ocr_diagnostics_explain_direct_only_single_digit_rejection()
    test_mmocr_score_uses_mean_character_confidence()
    test_ocr_skips_referee_candidate_rows()
    test_team_assigner_uses_roster_kit_colors_without_kmeans()
    test_team_assigner_normalizes_kmeans_random_states()
    test_team_assigner_accepts_multiple_kmeans_initializations()
    test_summary_preserves_ocr_votes_not_frame_count()
    test_roster_aware_ocr_degrades_number_not_in_team_roster()
    test_roster_aware_ocr_promotes_valid_alternative()
    test_roster_aware_ocr_distribution_does_not_promote_when_disabled()
    test_roster_aware_ocr_preserves_dropped_candidate_evidence()
    test_hungarian_uses_preserved_dropped_candidate_evidence()
    test_hungarian_summary_marks_preserved_dropped_evidence()
    test_hungarian_uses_jersey_distribution_candidate()
    test_jersey_ocr_segment_group_key_uses_frame_window()
    test_hungarian_identity_tracklet_id_keeps_segment_assignments_local()
    test_hungarian_prefers_raw_distribution_for_jersey_cost()
    test_referee_candidates_are_not_player_identity_summaries()
    test_linker_team_gate_blocks_confident_team_mismatch()
    test_linker_appearance_gate_blocks_low_similarity()
    test_linker_accepts_distance_gap_when_embeddings_missing()
    test_jersey_identity_linker_links_reliable_same_number_continuation()
    test_jersey_identity_linker_rejects_far_same_number_tracklet()
    test_strongsort_core_uses_appearance_to_keep_identity()
    test_position_prior_is_capped()
    test_reliable_jersey_beats_position_prior()
    test_assignment_gate_blocks_weak_non_jersey_assignment()
    test_assignment_gate_allows_strong_team_visual_trajectory()
    test_assignment_gate_blocks_weak_roster_promoted_jersey_candidate()
    test_duplicate_jersey_rank_prefers_stronger_ocr_over_identity_confidence()
    test_constraints_clear_duplicate_player_id_in_same_frame()
    test_constraints_clear_invalid_team_jersey()
    test_constraints_clear_duplicate_team_jersey_in_same_frame()
    test_constraints_preserve_same_player_global_team_jersey_fragments()
    test_constraints_clear_competing_global_team_jersey_owner()
    test_constraints_clear_unknown_global_team_jersey_competitor()
    test_constraints_clear_goalkeeper_only_jersey_on_non_goalkeeper()
    test_constraints_can_fallback_from_goalkeeper_only_jersey_to_alternate_candidate()
    test_constraints_do_not_fallback_to_alternate_with_known_global_owner()
    test_constraints_clear_non_goalkeeper_jersey_on_goalkeeper()
    test_constraints_correct_goalkeeper_semantic_group_from_roster()
    test_constraints_clear_goalkeeper_identity_without_goalkeeper_evidence()
    test_constraints_clear_identity_on_strong_frame_team_conflict()
    test_constraints_split_persistent_frame_team_conflict_segment()
    test_constraints_split_merges_short_non_conflict_bridge()
    test_referee_color_ranges_can_come_from_roster_metadata()
    test_goalkeeper_color_ranges_can_come_from_roster_metadata()
    test_goalkeeper_color_ranges_support_custom_colour_names()
    test_goalkeeper_roster_colour_can_correct_team_when_confident()
    test_overlay_hides_assigned_jersey_without_ocr_evidence()
    test_overlay_shows_reliable_ocr_jersey()
    test_overlay_can_show_high_confidence_player_id_when_enabled()
    test_overlay_shows_stable_low_mass_ocr_jersey()
    test_overlay_shows_head_confident_crop_aggregated_jersey()
    test_overlay_can_show_any_ocr_winner_for_diagnostics()
    test_overlay_labels_referee_candidate_without_question_mark()
    print("FT identity tests passed")
