from ft.tracking.ball_kalman import BallKalmanTracker


def test_ball_kalman_initializes_and_updates_position():
    tracker = BallKalmanTracker(max_lost_frames=2)
    assert tracker.predict() is None

    assert tracker.update([10.0, 20.0]) == [10.0, 20.0]
    predicted = tracker.predict()
    assert predicted is not None

    updated = tracker.update([13.0, 24.0])
    assert abs(updated[0] - 13.0) < 1.0
    assert abs(updated[1] - 24.0) < 1.0
    assert tracker.is_valid()


def test_ball_kalman_expires_after_lost_window():
    tracker = BallKalmanTracker(max_lost_frames=1)
    tracker.init([0.0, 0.0])
    assert tracker.is_valid()
    tracker.predict()
    assert tracker.is_valid()
    tracker.predict()
    assert not tracker.is_valid()


if __name__ == "__main__":
    test_ball_kalman_initializes_and_updates_position()
    test_ball_kalman_expires_after_lost_window()
    print("ball kalman tests passed")
