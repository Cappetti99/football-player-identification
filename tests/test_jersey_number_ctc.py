import math

import pytest

from ft.features.jersey_number_ctc import aggregate_frames, encode_text
from scripts.build_jersey_number_ctc_dataset import limit_tracks, normalized_text


def test_numeric_text_validation_and_normalization():
    assert encode_text("10") == [1, 0]
    assert normalized_text({"gt_jersey": "07", "transcription": "7"}) == "7"
    with pytest.raises(ValueError):
        encode_text("123")
    with pytest.raises(ValueError):
        normalized_text({"audit_id": "x", "gt_jersey": "10", "transcription": "11"})


def test_probabilistic_aggregation_uses_readability_weights():
    first = {"10": math.log(0.9), "11": math.log(0.1)}
    second = {"10": math.log(0.1), "11": math.log(0.9)}
    result = aggregate_frames([first, second], weights=[1.0, 0.1])
    assert result["prediction"] == 10
    assert result["confidence"] > 0.5
    assert result["margin"] > 0


def test_track_limiting_is_temporally_distributed():
    rows = [
        {"sequence": "A", "gt_track_id": "1", "frame": frame}
        for frame in range(10)
    ]
    selected = limit_tracks(rows, 3)
    assert [row["frame"] for row in selected] == [0, 4, 9]
