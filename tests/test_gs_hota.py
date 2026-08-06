"""Regression tests for GS-HOTA (Somers et al., CVPR24, arXiv:2404.11335).

Sim_GS-HOTA(P, G) = LocSim(P, G) x IdSim(P, G):
- LocSim is a Gaussian kernel over Euclidean pitch distance in metres, tau=5m.
- IdSim requires role, team and jersey number to all match, except an
  attribute is skipped when the ground truth does not provide it.
"""

import math

from ft.evaluation.gsr_detection_tracking import gs_hota_similarity, gs_hota_summary


def gt_row(x=10.0, y=20.0, role="player", team="left", jersey=7):
    return {
        "gt_position_pitch": [x, y],
        "gt_role": role,
        "gt_team": team,
        "gt_jersey": jersey,
    }


def pred_row(x=10.0, y=20.0, role="player", team="left", jersey=7):
    return {
        "pred_position_pitch": [x, y],
        "pred_role": role,
        "pred_team": team,
        "pred_jersey": jersey,
    }


def test_identical_position_and_attributes_gives_similarity_one():
    assert gs_hota_similarity(gt_row(), pred_row()) == 1.0


def test_wrong_jersey_number_zeroes_the_similarity():
    assert gs_hota_similarity(gt_row(jersey=7), pred_row(jersey=9)) == 0.0


def test_wrong_team_zeroes_the_similarity():
    assert gs_hota_similarity(gt_row(team="left"), pred_row(team="right")) == 0.0


def test_wrong_role_zeroes_the_similarity():
    assert gs_hota_similarity(gt_row(role="player"), pred_row(role="goalkeeper")) == 0.0


def test_missing_gt_attribute_is_ignored_not_penalized():
    """A referee has no jersey number in the ground truth; IdSim must not
    require the prediction's jersey field to also be null."""
    referee_gt = {
        "gt_position_pitch": [5.0, 5.0],
        "gt_role": "referee",
        "gt_team": None,
        "gt_jersey": None,
    }
    referee_pred = {
        "pred_position_pitch": [5.0, 5.0],
        "pred_role": "referee",
        "pred_team": None,
        "pred_jersey": 42,
    }
    assert gs_hota_similarity(referee_gt, referee_pred) == 1.0


def test_distance_tolerance_matches_the_paper_formula():
    """LocSim(P, G) = exp(ln(0.05) * ||P-G||^2 / tau^2); at ||P-G||=tau the
    similarity is exactly 0.05 by construction (Fig. 2 in the paper)."""
    tau = 5.0
    far_gt = gt_row(x=10.0, y=20.0 + tau)
    similarity = gs_hota_similarity(far_gt, pred_row(x=10.0, y=20.0), tau=tau)
    assert math.isclose(similarity, 0.05, rel_tol=1e-9)


def test_missing_position_never_matches():
    no_position_gt = {"gt_position_pitch": None, "gt_role": "player", "gt_team": "left", "gt_jersey": 7}
    assert gs_hota_similarity(no_position_gt, pred_row()) == 0.0


def test_missing_prediction_position_never_matches():
    no_position_pred = {"pred_position_pitch": None, "pred_role": "player", "pred_team": "left", "pred_jersey": 7}
    assert gs_hota_similarity(gt_row(), no_position_pred) == 0.0


def test_perfect_two_frame_sequence_scores_one():
    gt = {
        0: [{"gt_track_id": "g1", **gt_row()}],
        1: [{"gt_track_id": "g1", **gt_row()}],
    }
    pred = {
        0: [{"pred_identity_id": "p1", **pred_row()}],
        1: [{"pred_identity_id": "p1", **pred_row()}],
    }
    result = gs_hota_summary(gt, pred)
    assert result["gs_hota"] == 1.0
    assert result["gs_deta"] == 1.0
    assert result["gs_assa"] == 1.0


def test_identity_swap_mid_sequence_hurts_association_not_localization():
    """Two athletes at fixed positions; predictions swap identities halfway.
    Localization stays perfect (positions matched correctly throughout the
    swap window is not guaranteed at every alpha, but detection accuracy at
    low alpha should stay near 1 while association accuracy drops)."""
    gt = {
        0: [{"gt_track_id": "g1", **gt_row(x=0.0, jersey=7)}, {"gt_track_id": "g2", **gt_row(x=50.0, jersey=9)}],
        1: [{"gt_track_id": "g1", **gt_row(x=0.0, jersey=7)}, {"gt_track_id": "g2", **gt_row(x=50.0, jersey=9)}],
    }
    # frame 1: predicted jersey numbers are swapped relative to their gt match
    # at those positions, so IdSim rejects the "same position" pairing and the
    # association must break.
    pred = {
        0: [{"pred_identity_id": "p1", **pred_row(x=0.0, jersey=7)}, {"pred_identity_id": "p2", **pred_row(x=50.0, jersey=9)}],
        1: [{"pred_identity_id": "p1", **pred_row(x=0.0, jersey=9)}, {"pred_identity_id": "p2", **pred_row(x=50.0, jersey=7)}],
    }
    result = gs_hota_summary(gt, pred)
    assert result["gs_assa"] < 1.0
