from ft.evaluation.identity_benchmark import (
    adjudicate,
    assign_second_review,
    bbox_iou,
    compare_runs,
    evaluate_identity_units,
    group_identity_units,
    identity_metrics,
    match_anchors,
    merge_identity_units,
    promotion_gate,
    validate_annotation,
    wilson_interval,
)


def player_row(frame, bbox, display=1, player="unknown", confidence=0.0, status="unknown"):
    return {
        "track_group": "players",
        "frame": frame,
        "raw_track_id": display,
        "display_track_id": display,
        "identity_tracklet_id": display,
        "scene_segment_id": 0,
        "bbox": bbox,
        "crop_path": f"/tmp/{display}_{frame}.jpg",
        "crop_quality": 0.5,
        "player_id": player,
        "identity_confidence": confidence,
        "identity_status": status,
        "identity_evidence": {},
    }


def test_bbox_iou_and_run_independent_unit_merge_are_deterministic():
    first = [
        player_row(frame, [10, 10, 30, 50], display=1)
        for frame in range(6)
    ]
    second = [
        player_row(frame, [11, 10, 31, 50], display=99)
        for frame in range(6)
    ]
    units = (
        group_identity_units(first, "run_a", "Video", "test")
        + group_identity_units(second, "run_b", "Video", "test")
    )
    forward = merge_identity_units(units, anchors_per_unit=4)
    backward = merge_identity_units(list(reversed(units)), anchors_per_unit=4)

    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert len(forward) == 1
    assert forward[0]["item_id"] == backward[0]["item_id"]
    assert len(forward[0]["members"]) == 2


def test_second_review_sampling_is_seeded_and_stratified():
    items = [
        {
            "item_id": f"u{index}",
            "item_type": "identity",
            "video_id": "V",
            "strata": ["unknown" if index < 5 else "direct_assigned"],
        }
        for index in range(10)
    ]
    first = assign_second_review([dict(row) for row in items], fraction=0.2, seed=7)
    second = assign_second_review(list(reversed([dict(row) for row in items])), fraction=0.2, seed=7)
    assert {
        row["item_id"] for row in first if row["second_review_required"]
    } == {
        row["item_id"] for row in second if row["second_review_required"]
    }
    assert sum(row["second_review_required"] for row in first) == 2


def test_anchor_matching_metrics_and_delta_use_determinable_unknowns():
    rows = [player_row(frame, [10, 10, 30, 50]) for frame in range(4)]
    unit = merge_identity_units(
        group_identity_units(rows, "reference", "Video", "test"),
        anchors_per_unit=4,
    )[0]
    manifest = {"identity_units": [unit]}
    gt = [{
        "item_id": unit["item_id"],
        "item_type": "identity",
        "annotation_status": "determinate",
        "gt_player_id": "p7",
    }]
    baseline_rows = {"Video": [player_row(frame, [10, 10, 30, 50]) for frame in range(4)]}
    candidate_rows = {
        "Video": [
            player_row(frame, [10, 10, 30, 50], player="p7", confidence=0.9, status="assigned")
            for frame in range(4)
        ]
    }
    baseline = evaluate_identity_units(manifest, gt, baseline_rows)
    candidate = evaluate_identity_units(manifest, gt, candidate_rows)
    metrics = identity_metrics(candidate)
    delta = compare_runs(baseline, candidate, gt)

    assert metrics["identity_precision_unit"] == 1.0
    assert metrics["correct_coverage"] == 1.0
    assert delta["new_decisions"] == 1
    assert delta["new_false_positives"] == 0


def test_anchor_matching_rejects_missing_and_ambiguous_detections():
    anchors = [
        {"frame": 0, "bbox": [10, 10, 30, 50]},
        {"frame": 1, "bbox": [10, 10, 30, 50]},
    ]
    candidates = [
        player_row(0, [10, 10, 30, 50], display=1),
        player_row(0, [10, 10, 30, 50], display=2),
    ]
    matches = match_anchors(anchors, candidates)
    assert matches[0]["ambiguous"] is True
    assert matches[0]["matched"] is False
    assert matches[1]["matched"] is False


def test_adjudication_requires_resolution_and_reports_agreement():
    base = {
        "item_id": "u1",
        "item_type": "identity",
        "video_id": "V",
        "split": "test",
        "second_review_required": "true",
        "annotation_status": "determinate",
        "gt_team_id": "1",
        "gt_jersey_number": "7",
        "jersey_visibility": "full",
        "uncertainty_reason": "",
        "notes": "",
    }
    reviewer_a = [{**base, "gt_player_id": "p7"}]
    reviewer_b = [{**base, "gt_player_id": "p8"}]
    final, disagreements, report = adjudicate(reviewer_a, reviewer_b)
    assert final == []
    assert len(disagreements) == 1
    assert report["exact_identity_agreement"] == 0.0

    final, disagreements, _ = adjudicate(
        reviewer_a,
        reviewer_b,
        [{**base, "gt_player_id": "p7"}],
    )
    assert len(final) == 1
    assert len(disagreements) == 1


def test_wilson_zero_error_bound_and_fail_closed_promotion():
    lower, upper = wilson_interval(100, 100)
    assert lower < 1.0
    assert upper == 1.0
    gate = promotion_gate(
        {"identity_precision_unit": 1.0, "correct_coverage": 0.8},
        {"identity_precision_unit": 1.0, "correct_coverage": 0.9},
        {"new_false_positives": 0, "new_indeterminate": 1},
        hashes_match=True,
    )
    assert gate["status"] == "inconclusive"


def test_annotation_validation_requires_reasons_and_visibility():
    base = {"item_type": "identity", "gt_player_id": "", "jersey_visibility": ""}
    assert "uncertainty_reason" in validate_annotation({
        **base, "annotation_status": "exclude", "uncertainty_reason": "",
    })
    assert "jersey_visibility" in validate_annotation({
        **base, "annotation_status": "not_determinable", "uncertainty_reason": "blur",
    })
    assert validate_annotation({
        **base,
        "annotation_status": "not_determinable",
        "uncertainty_reason": "blur",
        "jersey_visibility": "not_visible",
    }) is None
