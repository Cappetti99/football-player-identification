from scripts.train_yolo_gsr_full import (
    bbox_to_yolo,
    infer_split,
    map_role_to_class_id,
)


def test_gsr_official_splits_remain_separate():
    assert infer_split("/dataset/SoccerNet-GSR/train/SNGS-001/Labels-GameState.json") == "train"
    assert infer_split("/dataset/SoccerNet-GSR/valid/SNGS-002/Labels-GameState.json") == "val"
    assert infer_split("/dataset/SoccerNet-GSR/test/SNGS-003/Labels-GameState.json") == "test"
    assert infer_split("/dataset/SoccerNet-GSR/challenge/SNGS-004/Labels-GameState.json") == "test"


def test_person_ball_mapping_collapses_semantic_roles():
    assert map_role_to_class_id("player", "person_ball") == 0
    assert map_role_to_class_id("goalkeeper", "person_ball") == 0
    assert map_role_to_class_id("referee", "person_ball") == 0
    assert map_role_to_class_id("ball", "person_ball") == 1
    assert map_role_to_class_id("pitch", "person_ball") is None


def test_bbox_conversion_rejects_invalid_and_normalizes_valid_boxes():
    assert bbox_to_yolo({}, 1920, 1080, 4.0) is None
    assert bbox_to_yolo({"x": 10, "y": 20, "w": 1, "h": 1}, 1920, 1080, 4.0) is None

    converted = bbox_to_yolo(
        {"x": 100, "y": 50, "w": 200, "h": 100},
        1000,
        500,
        4.0,
    )

    assert converted == [0.2, 0.2, 0.2, 0.2]
