from ft.tracking.yolo_bytetrack import YoloByteTracker


def make_tracker():
    tracker = YoloByteTracker.__new__(YoloByteTracker)
    tracker.ball_max_area_ratio = 0.01
    tracker.ball_size_penalty = 0.0
    tracker.ball_temporal_consistency = True
    tracker.ball_temporal_max_distance = 60.0
    tracker.ball_temporal_max_distance_cap = 60.0
    tracker.ball_temporal_distance_penalty = 0.50
    tracker.ball_temporal_reject_outliers = True
    tracker.ball_min_acquisition_confidence = 0.05
    tracker.ball_low_confidence_max_distance = 30.0
    tracker.ball_temporal_min_confidence_after_miss = 0.05
    tracker.ball_temporal_miss_reset = 12
    tracker.ball_kalman = None
    tracker.ball_kalman_enabled = False
    tracker.ball_kalman_high_speed_threshold = 30.0
    tracker.ball_kalman_high_speed_area_multiplier = 3.0
    tracker._last_ball_bbox_size = None
    tracker.reset_ball_state()
    return tracker


def test_ball_selection_penalizes_large_temporal_jump():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([104.0, 104.0, 116.0, 116.0], 0.30),
        ([490.0, 490.0, 510.0, 510.0], 0.60),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected["bbox"] == candidates[0][0]
    assert selected["temporal_distance"] < 20.0
    assert selected["score"] < selected["base_score"]


def test_ball_selection_reset_removes_temporal_penalty():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    tracker.reset_ball_state()
    candidates = [
        ([104.0, 104.0, 116.0, 116.0], 0.30),
        ([490.0, 490.0, 510.0, 510.0], 0.60),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected["bbox"] == candidates[1][0]
    assert selected["temporal_distance"] is None


def test_ball_selection_is_stateless_when_temporal_consistency_disabled():
    tracker = make_tracker()
    tracker.ball_temporal_consistency = False
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([104.0, 104.0, 116.0, 116.0], 0.30),
        ([490.0, 490.0, 510.0, 510.0], 0.60),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected["bbox"] == candidates[1][0]
    assert selected["temporal_distance"] is None


def test_ball_selection_rejects_temporal_outlier_when_no_near_candidate():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([490.0, 490.0, 510.0, 510.0], 0.90),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected is None
    assert tracker._last_ball_center == [100.0, 100.0]


def test_ball_selection_caps_allowed_distance_after_missed_frames():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([205.0, 95.0, 215.0, 105.0], 0.40),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=13)

    assert selected is None


def test_ball_selection_requires_confidence_after_missed_frames():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([112.0, 95.0, 122.0, 105.0], 0.004),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=13)

    assert selected is None


def test_ball_selection_requires_min_confidence_for_acquisition():
    tracker = make_tracker()
    candidates = [
        ([104.0, 104.0, 116.0, 116.0], 0.004),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected is None


def test_ball_selection_allows_low_confidence_when_temporally_consistent():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([104.0, 104.0, 116.0, 116.0], 0.004),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected is not None
    assert selected["bbox"] == candidates[0][0]


def test_ball_selection_rejects_low_confidence_temporal_jump():
    tracker = make_tracker()
    tracker._last_ball_center = [100.0, 100.0]
    tracker._last_ball_frame = 10
    candidates = [
        ([195.0, 95.0, 205.0, 105.0], 0.004),
    ]

    selected = tracker.select_ball(candidates, frame_area=1_000_000.0, frame_num=11)

    assert selected is None


def test_ball_interpolation_preserves_detected_metadata():
    frames = [
        {1: {"bbox": [0.0, 0.0, 10.0, 10.0], "confidence": 0.40, "score": 0.25}},
        {},
        {1: {"bbox": [20.0, 20.0, 30.0, 30.0], "confidence": 0.50, "score": 0.35}},
    ]

    interpolated = YoloByteTracker.interpolate_ball(frames)

    assert interpolated[0][1]["confidence"] == 0.40
    assert interpolated[0][1]["interpolated"] is False
    assert interpolated[1][1]["interpolated"] is True
    assert "confidence" not in interpolated[1][1]
    assert interpolated[2][1]["score"] == 0.35
    assert interpolated[2][1]["interpolated"] is False


def test_ball_interpolation_preserves_kalman_prediction_metadata():
    frames = [
        {1: {"bbox": [0.0, 0.0, 10.0, 10.0], "confidence": 0.40}},
        {1: {"bbox": [10.0, 0.0, 20.0, 10.0], "interpolated": True, "kalman_predicted": True}},
    ]

    interpolated = YoloByteTracker.interpolate_ball(frames)

    assert interpolated[0][1]["interpolated"] is False
    assert interpolated[1][1]["interpolated"] is True
    assert interpolated[1][1]["kalman_predicted"] is True


def test_ball_selection_accepts_larger_blur_box_with_dynamic_area_ratio():
    tracker = make_tracker()
    candidate = ([0.0, 0.0, 120.0, 120.0], 0.40)

    assert tracker.select_ball([candidate], frame_area=1_000_000.0, frame_num=1) is None
    selected = tracker.select_ball([candidate], frame_area=1_000_000.0, frame_num=1, max_area_ratio=0.02)

    assert selected is not None
    assert selected["bbox"] == candidate[0]


def test_ball_selection_uses_kalman_prediction_to_reject_far_candidate():
    tracker = make_tracker()
    candidates = [
        ([104.0, 104.0, 116.0, 116.0], 0.20),
        ([900.0, 390.0, 920.0, 410.0], 0.80),
    ]

    selected = tracker.select_ball(
        candidates,
        frame_area=1_000_000.0,
        frame_num=20,
        predicted_center=[110.0, 110.0],
        predicted_gap=1,
    )

    assert selected["bbox"] == candidates[0][0]
    assert selected["temporal_distance"] == 0.0


if __name__ == "__main__":
    test_ball_selection_penalizes_large_temporal_jump()
    test_ball_selection_reset_removes_temporal_penalty()
    test_ball_selection_is_stateless_when_temporal_consistency_disabled()
    test_ball_selection_rejects_temporal_outlier_when_no_near_candidate()
    test_ball_selection_caps_allowed_distance_after_missed_frames()
    test_ball_selection_requires_confidence_after_missed_frames()
    test_ball_selection_requires_min_confidence_for_acquisition()
    test_ball_selection_allows_low_confidence_when_temporally_consistent()
    test_ball_selection_rejects_low_confidence_temporal_jump()
    test_ball_interpolation_preserves_detected_metadata()
    test_ball_interpolation_preserves_kalman_prediction_metadata()
    test_ball_selection_accepts_larger_blur_box_with_dynamic_area_ratio()
    test_ball_selection_uses_kalman_prediction_to_reject_far_candidate()
    print("ball tracking tests passed")
