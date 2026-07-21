import json

import pytest

from ft.features.jersey_multiframe_digits import (
    masked_softmax,
    number_log_probabilities,
    number_targets,
)
from scripts.build_jersey_multiframe_digit_dataset import (
    group_track_bags,
    temporal_sample,
    validate_source_manifest,
)
from scripts.train_jersey_multiframe_digits import optional_tens_loss, validate_manifest
from scripts.evaluate_jersey_multiframe_digits import paired_comparison


def test_masked_attention_ignores_padding():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 1.0, 99.0], [2.0, 0.0, -5.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    result = masked_softmax(logits, mask)
    assert torch.allclose(result.sum(dim=1), torch.ones(2))
    assert result[0, 2] == 0
    assert result[1].tolist() == [1.0, 0.0, 0.0]


def test_digit_heads_map_to_numbers_zero_through_99():
    torch = pytest.importorskip("torch")
    outputs = {
        "length_logits": torch.tensor([[-5.0, 5.0], [5.0, -5.0]]),
        "tens_logits": torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 9.0],
            [0.0] * 10,
        ]),
        "units_logits": torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
    }
    probabilities = number_log_probabilities(outputs)
    assert probabilities.shape == (2, 100)
    assert probabilities.argmax(dim=1).tolist() == [95, 2]
    assert torch.allclose(probabilities.logsumexp(dim=1), torch.zeros(2), atol=1e-6)


def test_number_targets_are_interpretable():
    lengths, tens, units = number_targets([2, 10, 95])
    assert lengths.tolist() == [0, 1, 1]
    assert tens.tolist() == [-100, 1, 9]
    assert units.tolist() == [2, 0, 5]


def test_tens_loss_is_finite_for_one_digit_only_batch():
    torch = pytest.importorskip("torch")
    logits = torch.randn(3, 10, requires_grad=True)
    loss = optional_tens_loss(logits, torch.tensor([-100, -100, -100]))
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
    loss.backward()


def test_grouping_is_temporal_and_rejects_inconsistent_tracks():
    rows = [
        {"sequence": "A", "gt_track_id": "1", "frame": frame, "image": f"{frame}.jpg", "text": "95"}
        for frame in range(10)
    ]
    bag = group_track_bags(rows, 3)[0]
    assert [row["frame"] for row in bag["frames"]] == [0, 4, 9]
    bad = rows + [{"sequence": "A", "gt_track_id": "1", "frame": 11, "image": "x.jpg", "text": "55"}]
    with pytest.raises(ValueError, match="inconsistent"):
        group_track_bags(bad, 3)


def test_manifests_reject_frozen_or_overlapping_sequences():
    with pytest.raises(ValueError, match="frozen"):
        validate_source_manifest({"format": "jersey_numeric_ctc_v1", "frozen_sequences_observed": ["X"]})
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest({
            "format": "jersey_multiframe_digits_v1",
            "train_sequences": ["A"],
            "validation_sequences": ["A"],
            "frozen_sequences_observed": [],
        })


def test_paired_comparison_counts_regressions_and_recoveries():
    baseline = [
        {"sequence": "A", "gt_track_id": "1", "correct": "True"},
        {"sequence": "A", "gt_track_id": "2", "correct": "False"},
    ]
    candidate = [
        {"sequence": "A", "gt_track_id": "1", "correct": False},
        {"sequence": "A", "gt_track_id": "2", "correct": True},
    ]
    result = paired_comparison(baseline, candidate)
    assert result["correct_to_wrong"] == 1
    assert result["wrong_to_correct"] == 1
    assert result["net_correct"] == 0
