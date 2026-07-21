import pytest

from scripts.build_sjn210k_datasets import (
    localization_policy,
    parse_label_line,
    quadrilateral_bbox,
)


def test_sjn_blank_second_digit_and_quadrilateral_are_parsed():
    one = parse_label_line("0.jpg 5 10", "train", 1)
    assert one["text"] == "5"
    assert one["second_digit"] is None
    two = parse_label_line(
        "1.jpg 1 0 0.1 0.2 0.1 0.4 0.8 0.4 0.8 0.2", "train", 2
    )
    assert two["text"] == "10"
    assert len(two["quad"]) == 8


def test_localization_policy_clips_moderate_and_rejects_severe_excess():
    status, clipped, excess = localization_policy(
        [-0.02, 0.2, 0.1, 0.4, 0.8, 0.4, 1.03, 0.2], 0.05
    )
    assert status == "clipped"
    assert excess == pytest.approx(0.03)
    assert min(clipped) == 0.0 and max(clipped) == 1.0
    status, clipped, excess = localization_policy(
        [-0.06, 0.2, 0.1, 0.4, 0.8, 0.4, 1.0, 0.2], 0.05
    )
    assert status == "excluded_excess"
    assert clipped is None


def test_quadrilateral_bbox_is_axis_aligned_and_normalized():
    assert quadrilateral_bbox([0.1, 0.2, 0.2, 0.8, 0.9, 0.7, 0.8, 0.1]) == pytest.approx(
        (0.5, 0.45, 0.8, 0.7)
    )
