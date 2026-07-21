import pytest

from scripts.build_jersey_number_region_ctc_dataset import padded_box


def test_region_ctc_padding_is_clipped():
    assert padded_box((0.0, 0.1, 0.5, 0.9), 0.1) == pytest.approx(
        (0.0, 0.02, 0.55, 0.98)
    )
