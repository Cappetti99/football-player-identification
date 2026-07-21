import pytest

from scripts.evaluate_jersey_number_region_ctc import padded_box, row_key


def test_region_padding_is_clipped_to_image():
    assert padded_box((0.0, 0.1, 0.5, 0.9), 0.1) == pytest.approx(
        (0.0, 0.02, 0.55, 0.98)
    )


def test_region_surface_key_uses_sequence_track_and_frame():
    assert row_key({"sequence": "A", "gt_track_id": 7, "frame": "12"}) == (
        "A", "7", 12
    )
