import math

from ft.evaluation.gsr_detection_tracking import (
    average_precision,
    detection_summary,
    evaluate_frames,
    hota_summary,
    size_breakdown,
    tracking_summary,
)


BOX = [10.0, 10.0, 50.0, 90.0]


def gt_row(track_id="gt-1", bbox=BOX, role="player"):
    return {"gt_track_id": track_id, "bbox": list(bbox), "gt_role": role}


def pred_row(track_id="pred-1", bbox=BOX, confidence=0.9, role="player"):
    return {
        "pred_identity_id": f"players:{track_id}",
        "raw_pred_identity_id": f"players:{track_id}",
        "bbox": list(bbox),
        "pred_role": role,
        "detection_confidence": confidence,
    }


def test_perfect_detection_and_tracking_have_unit_scores():
    gt = {frame: [gt_row()] for frame in range(3)}
    pred = {frame: [pred_row()] for frame in range(3)}

    evaluation = evaluate_frames(gt, pred, 0.5, continuity_id_key="pred_identity_id")
    detection = detection_summary(evaluation)
    tracking = tracking_summary(evaluation, gt, pred)
    hota = hota_summary(gt, pred)

    assert detection["precision"] == 1.0
    assert detection["recall"] == 1.0
    assert tracking["mota"] == 1.0
    assert tracking["idf1"] == 1.0
    assert tracking["id_switches"] == 0
    assert tracking["fragmentations"] == 0
    assert hota["hota"] == 1.0
    assert hota["deta"] == 1.0
    assert hota["assa"] == 1.0


def test_identity_switch_lowers_idf1_without_changing_detection():
    gt = {frame: [gt_row()] for frame in range(3)}
    pred = {
        0: [pred_row("pred-a")],
        1: [pred_row("pred-b")],
        2: [pred_row("pred-b")],
    }

    evaluation = evaluate_frames(gt, pred, 0.5, continuity_id_key="pred_identity_id")
    detection = detection_summary(evaluation)
    tracking = tracking_summary(evaluation, gt, pred)

    assert detection["f1"] == 1.0
    assert tracking["id_switches"] == 1
    assert math.isclose(tracking["idf1"], 2.0 / 3.0)
    assert math.isclose(tracking["mota"], 2.0 / 3.0)


def test_missed_middle_frame_counts_one_fragmentation():
    gt = {frame: [gt_row()] for frame in range(3)}
    pred = {0: [pred_row()], 2: [pred_row()]}

    evaluation = evaluate_frames(gt, pred, 0.5, continuity_id_key="pred_identity_id")
    tracking = tracking_summary(evaluation, gt, pred)

    assert tracking["fragmentations"] == 1
    assert tracking["id_switches"] == 0
    assert tracking["mostly_tracked"] == 0
    assert tracking["partially_tracked"] == 1


def test_clear_matching_preserves_valid_previous_assignments():
    left = [0.0, 0.0, 20.0, 40.0]
    right = [80.0, 0.0, 100.0, 40.0]
    overlap = [40.0, 0.0, 60.0, 40.0]
    gt = {
        0: [gt_row("g1", left), gt_row("g2", right)],
        1: [gt_row("g1", overlap), gt_row("g2", overlap)],
    }
    pred = {
        0: [pred_row("p1", left), pred_row("p2", right)],
        1: [pred_row("p2", overlap), pred_row("p1", overlap)],
    }

    independent = evaluate_frames(gt, pred, 0.5)
    clear = evaluate_frames(
        gt, pred, 0.5, continuity_id_key="pred_identity_id"
    )

    assert tracking_summary(independent, gt, pred)["id_switches"] == 2
    assert tracking_summary(clear, gt, pred)["id_switches"] == 0


def test_average_precision_uses_detector_confidence_order():
    gt = {0: [gt_row()]}
    false_box = [200.0, 200.0, 240.0, 280.0]
    bad_order = {
        0: [pred_row("true", confidence=0.4)],
        1: [pred_row("false", bbox=false_box, confidence=0.9)],
    }
    good_order = {
        0: [pred_row("true", confidence=0.9)],
        1: [pred_row("false", bbox=false_box, confidence=0.4)],
    }

    bad = average_precision(gt, bad_order, 0.5)
    good = average_precision(gt, good_order, 0.5)

    assert bad["available"] is True
    assert math.isclose(bad["ap"], 0.5)
    assert good["ap"] == 1.0


def test_ap_is_explicitly_unavailable_for_legacy_artifacts():
    gt = {0: [gt_row()]}
    pred = {0: [pred_row(confidence=None)]}

    result = average_precision(gt, pred, 0.5)

    assert result["available"] is False
    assert result["reason"] == "detection_confidence_missing"
    assert result["ap"] is None


def test_bbox_size_recall_uses_declared_relative_bins():
    gt = {
        0: [gt_row("small", [0, 0, 20, 20])],
        1: [gt_row("medium", [0, 0, 100, 100])],
        2: [gt_row("large", [0, 0, 200, 200])],
    }
    pred = {
        0: [pred_row("small", [0, 0, 20, 20])],
        2: [pred_row("large", [0, 0, 200, 200])],
    }

    evaluation = evaluate_frames(gt, pred, 0.5)
    result = size_breakdown(evaluation, image_width=1000, image_height=1000)

    assert result["small"]["recall"] == 1.0
    assert result["medium"]["recall"] == 0.0
    assert result["large"]["recall"] == 1.0
