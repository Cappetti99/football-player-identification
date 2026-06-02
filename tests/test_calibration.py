import json

import numpy as np

from ft.calibration.pitch_transform import PitchTransform
from ft.calibration.tvcalib_adapter import (
    load_tvcalib_homography_map,
    tvcalib_homography_to_pitch,
)


def test_tvcalib_centered_homography_shifts_to_ft_pitch():
    h = tvcalib_homography_to_pitch(np.eye(3), pitch_length=105.0, pitch_width=68.0)

    point = np.asarray([0.0, 0.0, 1.0])
    transformed = h @ point

    assert transformed[:2].tolist() == [52.5, 34.0]


def test_tvcalib_jsonl_homography_map_uses_image_id_frame_suffix(tmp_path):
    path = tmp_path / "per_sample_output.json"
    records = [
        {"image_ids": "frame_000010.jpg", "homography": np.eye(3).tolist()},
        {"image_ids": "frame_000020.jpg", "homography": (np.eye(3) * 2.0).tolist()},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    homographies = load_tvcalib_homography_map(path, coordinate_system="ft", frame_offset=10)

    assert sorted(homographies) == [0, 10]
    assert np.allclose(homographies[0], np.eye(3))
    assert np.allclose(homographies[10], np.eye(3))


def test_pitch_transform_uses_nearest_tvcalib_frame_for_tracks():
    calibrator = PitchTransform(
        homographies_by_frame={
            0: np.eye(3, dtype=np.float32),
            10: np.asarray([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        },
        source="tvcalib:test",
        nearest_frame=True,
        max_frame_gap=5,
    )
    tracks = {
        "players": [
            {1: {"position": [1.0, 2.0], "bbox": [0, 0, 1, 1]}},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {2: {"position": [1.0, 2.0], "bbox": [0, 0, 1, 1]}},
        ]
    }

    calibrator.apply_tracks(tracks)

    assert tracks["players"][0][1]["position_pitch"] == [1.0, 2.0]
    assert tracks["players"][8][2]["position_pitch"] == [11.0, 2.0]
    assert calibrator.diagnostics()["mode"] == "per_frame"
