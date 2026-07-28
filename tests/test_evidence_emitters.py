import pytest

from ft.core.emitters import (
    PRODUCER_JERSEY_PRIMARY,
    PRODUCER_JERSEY_REGION_CTC,
    PRODUCER_JERSEY_SECONDARY,
    GROUP_PLAYERS,
    GROUP_REFEREES,
    display_subject,
    jersey_assignment_evidence,
    jersey_assignments_from_evidence,
    jersey_region_ctc_evidence,
    linking_evidence,
    referee_role_evidence,
    team_evidence,
)
from ft.core.evidence import EvidenceKind, SubjectType
from ft.pipeline import _apply_jersey, identity_tracklet_id


CONFIG_HASH = "cfg0"


def assignment(display_id=12, number=10, segment_index=None):
    return {
        "jersey_number": number,
        "confidence": 0.71,
        "head_confidence": 0.83,
        "winner_margin": 0.32,
        "winner_score": 2.1,
        "runner_up_score": 0.4,
        "winner_score_ratio": 5.25,
        "votes": 4,
        "total_detections": 9,
        "candidates": [{"jersey_number": number, "confidence": 0.71, "votes": 4, "score": 2.1}],
        "raw_jersey_distribution": {str(number): 0.71},
        "jersey_distribution": {str(number): 0.9},
        "jersey_roster_mass": 0.9,
        "roster_filter": "applied",
        "full_body_sufficient": True,
        "display_track_id": display_id,
        "segment_index": segment_index,
        "segment_start_frame": None if segment_index is None else segment_index * 500,
        "segment_end_frame": None if segment_index is None else segment_index * 500 + 499,
    }


# --- subject id contract ----------------------------------------------------

@pytest.mark.parametrize("display_id,segment_index", [(12, None), (12, 0), (7, 3)])
def test_subject_id_matches_pipeline_identity_tracklet_id(display_id, segment_index):
    # The emitter must not import the orchestrator, so this test is what pins
    # the two id schemes together.
    subject = display_subject(GROUP_PLAYERS, display_id, segment_index)
    parts = [int(part) for part in subject.split(":")[1:]]
    rebuilt = identity_tracklet_id(parts[0], parts[1] if len(parts) > 1 else None)
    assert rebuilt == identity_tracklet_id(display_id, segment_index)


def test_same_display_id_in_two_groups_stays_distinct():
    # display_track_id restarts per track group: SNGS-025 has a referee 13 and
    # a player 13, and they are different subjects.
    assert display_subject(GROUP_REFEREES, 13) != display_subject(GROUP_PLAYERS, 13)


# --- jersey primary ---------------------------------------------------------

def test_jersey_assignment_evidence_shape():
    rows = jersey_assignment_evidence({12: assignment()}, CONFIG_HASH)
    assert len(rows) == 1
    row = rows[0]
    assert row.subject_type == SubjectType.IDENTITY_TRACKLET
    assert row.subject_id == "players:12"
    assert row.kind == EvidenceKind.JERSEY_NUMBER
    assert row.value == "10"
    assert row.score == pytest.approx(0.71)
    assert row.produced_by == PRODUCER_JERSEY_PRIMARY
    assert row.config_hash == CONFIG_HASH
    assert row.payload["votes"] == 4


def test_tuple_and_segment_keys_produce_distinct_subjects():
    rows = jersey_assignment_evidence(
        {
            (12, 0): assignment(segment_index=0),
            (12, 1): assignment(segment_index=1, number=7),
        },
        CONFIG_HASH,
    )
    assert sorted(row.subject_id for row in rows) == ["players:12:0", "players:12:1"]


def test_competing_recognizers_coexist_on_the_same_subject():
    primary = jersey_assignment_evidence({12: assignment(number=10)}, CONFIG_HASH)
    secondary = jersey_assignment_evidence(
        {12: assignment(number=5)},
        CONFIG_HASH,
        produced_by=PRODUCER_JERSEY_SECONDARY,
        model_sha256="e93894",
    )
    subjects = {row.subject_id for row in primary + secondary}
    producers = {row.produced_by for row in primary + secondary}
    # Same subject, two contradictory values, no arbitration at this layer.
    assert subjects == {"players:12"}
    assert producers == {PRODUCER_JERSEY_PRIMARY, PRODUCER_JERSEY_SECONDARY}
    assert secondary[0].model_sha256 == "e93894"


# --- the sufficiency gate for step 2 ---------------------------------------

def rows_and_tracks(display_id=12, raw_id=12, frames=3):
    rows = [
        {
            "frame": frame,
            "track_id": raw_id,
            "raw_track_id": raw_id,
            "display_track_id": display_id,
            "track_group": "players",
        }
        for frame in range(frames)
    ]
    tracks = {
        "players": [
            {raw_id: {"display_track_id": display_id, "bbox": [0, 0, 10, 20]}}
            for _ in range(frames)
        ]
    }
    return rows, tracks


def test_evidence_reconstructs_jersey_rows_exactly():
    """Evidence must carry everything ``_apply_jersey`` writes into the rows."""
    assignments = {12: assignment(display_id=12), 7: assignment(display_id=7, number=23)}

    expected_rows, expected_tracks = rows_and_tracks()
    _apply_jersey(assignments, expected_rows, expected_tracks)

    evidence = jersey_assignment_evidence(assignments, CONFIG_HASH)
    rebuilt_assignments = jersey_assignments_from_evidence(evidence)
    actual_rows, actual_tracks = rows_and_tracks()
    _apply_jersey(rebuilt_assignments, actual_rows, actual_tracks)

    assert actual_rows == expected_rows
    assert actual_tracks == expected_tracks


def test_reconstruction_preserves_segmented_assignments():
    assignments = {(12, 0): assignment(segment_index=0), (12, 1): assignment(segment_index=1, number=7)}
    rebuilt = jersey_assignments_from_evidence(
        jersey_assignment_evidence(assignments, CONFIG_HASH)
    )
    assert set(rebuilt) == set(assignments)
    assert rebuilt[(12, 1)]["jersey_number"] == 7


def test_abstention_survives_the_roundtrip():
    evidence = jersey_assignment_evidence({12: {**assignment(), "jersey_number": None}}, CONFIG_HASH)
    assert evidence[0].abstained
    assert jersey_assignments_from_evidence(evidence)[12]["jersey_number"] is None


# --- region CTC auditor -----------------------------------------------------

def ctc_diagnostics():
    return {
        "enabled": True,
        "standalone_assignments": {
            "12#0": {
                "display_track_id": 12,
                "jersey_number": 95,
                "confidence": 0.42,
                "winner_margin": 0.11,
                "recognized_frames": 3,
                "frames": [10, 40, 90],
                "top5": [95, 55, 5, 9, 59],
                "applied": False,
            }
        },
        "crops": [
            {
                "display_track_id": 12,
                "frame": 40,
                "crop_path": "/crops/12_40.jpg",
                "crop_sha256": "aa",
                "crop_bytes": 1024,
                "crop_quality": 0.31,
                "selection_score": 0.9,
                "selection_reason": "legibility_top1",
                "selection_rank": 1,
                "detector_confidence": 0.55,
                "detector_checkpoint_sha256": "ec3c4d",
                "region_xyxyn": [0.1, 0.2, 0.3, 0.4],
                "region_width": 24,
                "region_height": 18,
                "box_padding": 0.25,
                "ctc_top1": 95,
                "ctc_top1_log_probability": -0.62,
                "ctc_top5": [95, 55, 5, 9, 59],
            }
        ],
        "configuration": {"ctc_checkpoint_sha256": "11c635"},
    }


def test_ctc_audit_emits_track_and_crop_evidence():
    rows = jersey_region_ctc_evidence(ctc_diagnostics(), CONFIG_HASH)
    tracks = [row for row in rows if row.subject_type == SubjectType.DISPLAY_TRACK]
    crops = [row for row in rows if row.subject_type == SubjectType.CROP]

    assert len(tracks) == 1 and len(crops) == 1
    assert tracks[0].subject_id == "players:12"
    assert tracks[0].payload["scene_segment_id"] == "0"
    assert tracks[0].frame_start == 10 and tracks[0].frame_end == 90
    assert all(row.produced_by == PRODUCER_JERSEY_REGION_CTC for row in rows)
    assert all(row.model_sha256 == "11c635" for row in rows)


def test_crop_evidence_carries_full_provenance():
    crop = [
        row
        for row in jersey_region_ctc_evidence(ctc_diagnostics(), CONFIG_HASH)
        if row.subject_type == SubjectType.CROP
    ][0]
    assert crop.subject_id == "/crops/12_40.jpg"
    assert crop.value == "95"
    for key in ("crop_sha256", "detector_checkpoint_sha256", "region_xyxyn", "box_padding", "ctc_top5"):
        assert key in crop.payload


def test_disabled_auditor_emits_nothing():
    assert jersey_region_ctc_evidence({"enabled": False}, CONFIG_HASH) == []
    assert jersey_region_ctc_evidence(None, CONFIG_HASH) == []


# --- team, role, link -------------------------------------------------------

def test_team_evidence_abstains_on_missing_team():
    rows = team_evidence(
        {
            "1": {"team": 0, "confidence": 0.8, "source": "roster_color", "margin": 0.2},
            "2": {"team": None, "confidence": 0.0, "num_colors": 1},
        },
        CONFIG_HASH,
    )
    by_subject = {row.subject_id: row for row in rows}
    assert by_subject["players:1"].value == "0"
    assert by_subject["players:2"].abstained


def test_referee_evidence_separates_decision_from_candidate():
    rows = referee_role_evidence(
        {
            "referees": {"3": {"score": 0.7, "color": "yellow", "is_referee_palette": True, "num_samples": 5}},
            "players": {"4": {"score": 0.5, "color": "yellow", "is_referee_palette": True, "num_samples": 8}},
        },
        CONFIG_HASH,
    )
    by_subject = {row.subject_id: row for row in rows}
    assert by_subject["referees:3"].value == "referee"
    # A palette match on a player is a signal, not the stage's decision.
    assert by_subject["players:4"].abstained
    assert by_subject["players:4"].payload["is_referee_palette"] is True


def test_linking_evidence_one_row_per_accepted_link():
    rows = linking_evidence(
        {
            "enabled": True,
            "accepted_links": [
                {"from_track_id": 4, "to_track_id": 9, "display_track_id": 4, "gap": 12, "distance": 30.5, "visual_similarity": 0.81}
            ],
        },
        CONFIG_HASH,
    )
    assert len(rows) == 1
    assert rows[0].kind == EvidenceKind.LINK
    assert rows[0].subject_id == "players:4"
    assert rows[0].value == "9"
    assert rows[0].score == pytest.approx(0.81)


def test_disabled_linker_emits_nothing():
    assert linking_evidence({"enabled": False, "status": "disabled"}, CONFIG_HASH) == []


def test_payload_is_lossless_for_unknown_upstream_fields():
    """A field added upstream must survive the round-trip, not be whitelisted away."""
    custom = {**assignment(), "roster_promoted": True, "some_future_field": [1, 2]}
    rebuilt = jersey_assignments_from_evidence(
        jersey_assignment_evidence({12: custom}, CONFIG_HASH)
    )[12]
    rebuilt.pop("legacy_key", None)
    assert rebuilt == custom
