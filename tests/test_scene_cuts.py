import numpy as np

from ft.utils.scene_cuts import (
    annotate_tracks_with_scene_segments,
    detect_scene_cuts,
    scene_segments,
)


def test_scene_cut_detector_finds_hard_colour_change():
    frames = []
    for _ in range(4):
        frames.append(np.full((80, 120, 3), (0, 130, 0), dtype=np.uint8))
    for _ in range(4):
        frames.append(np.full((80, 120, 3), (220, 220, 220), dtype=np.uint8))

    diagnostics = detect_scene_cuts(
        frames,
        enabled=True,
        threshold=0.25,
        min_gap=2,
        crop_top_fraction=0.0,
        crop_bottom_fraction=0.0,
        crop_left_fraction=0.0,
        crop_right_fraction=0.0,
    )

    assert diagnostics["status"] == "ok"
    assert diagnostics["cut_frames"] == [4]
    assert diagnostics["segments"] == [
        {"segment_index": 0, "start_frame": 0, "end_frame": 3, "num_frames": 4},
        {"segment_index": 1, "start_frame": 4, "end_frame": 7, "num_frames": 4},
    ]


def test_scene_segment_annotation_marks_cut_boundary():
    tracks = {
        "players": [
            {1: {"bbox": [0, 0, 10, 10]}},
            {1: {"bbox": [0, 0, 10, 10]}},
            {2: {"bbox": [0, 0, 10, 10]}},
        ],
        "referees": [{}, {}, {}],
        "ball": [{}, {}, {}],
    }

    annotate_tracks_with_scene_segments(
        tracks,
        [2],
        scene_cut_lookup={2: {"score": 1.2, "type": "hard_cut"}},
    )

    assert tracks["players"][0][1]["scene_segment_id"] == 0
    assert tracks["players"][2][2]["scene_segment_id"] == 1
    assert tracks["players"][2][2]["scene_cut_boundary"] is True
    assert tracks["players"][2][2]["scene_cut_score"] == 1.2


def test_scene_segments_ignore_invalid_cut_frames():
    assert scene_segments(5, [-1, 0, 3, 5, 99]) == [
        {"segment_index": 0, "start_frame": 0, "end_frame": 2, "num_frames": 3},
        {"segment_index": 1, "start_frame": 3, "end_frame": 4, "num_frames": 2},
    ]


if __name__ == "__main__":
    test_scene_cut_detector_finds_hard_colour_change()
    test_scene_segment_annotation_marks_cut_boundary()
    test_scene_segments_ignore_invalid_cut_frames()
    print("scene cut tests passed")
