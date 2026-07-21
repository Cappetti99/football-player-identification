from ft.tracking.observations import normalize_ultralytics_result, role_for_class


class FakeBoxes:
    def __init__(self):
        self.xyxy = [[1, 2, 11, 22], [5, 6, 9, 10], [0, 0, 4, 4]]
        self.cls = [0, 1, 2]
        self.conf = [0.9, 0.4, 0.8]

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    names = {0: "player", 1: "ball", 2: "advertisement"}
    boxes = FakeBoxes()
    orig_shape = (1080, 1920)


def test_ultralytics_results_become_detection_level_records():
    frame = normalize_ultralytics_result(FakeResult(), frame=17)
    assert frame.frame == 17
    assert frame.image_shape == (1080, 1920)
    assert len(frame.observations) == 2
    player, ball = frame.observations
    assert player.role == "player"
    assert player.athlete_payload() == {
        "bbox": [1.0, 2.0, 11.0, 22.0],
        "detection_confidence": 0.9,
        "class_name": "player",
        "role_detection": "player",
    }
    assert ball.is_ball
    assert not ball.is_athlete


def test_role_mapping_is_centralized():
    assert role_for_class("goalkeeper") == "goalkeeper"
    assert role_for_class("referee") == "referee"
    assert role_for_class("person") == "player"
    assert role_for_class("sports ball") == "ball"
