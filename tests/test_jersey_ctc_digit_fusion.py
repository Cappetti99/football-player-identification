import math

from ft.features.jersey_number_ctc import (
    CANDIDATES,
    digit_factorized_log_scores,
    interpolate_track_scores,
)
from scripts.evaluate_jersey_ctc_digit_fusion import select_weight


def distribution(winner, runner_up=None):
    probabilities = {candidate: 1e-9 for candidate in CANDIDATES}
    probabilities[str(winner)] = 0.8
    if runner_up is not None:
        probabilities[str(runner_up)] = 0.19
    total = sum(probabilities.values())
    return {key: math.log(value / total) for key, value in probabilities.items()}


def test_zero_digit_weight_exactly_preserves_whole_number_winner():
    scores = [distribution(95, 55), distribution(95, 55)]
    result = interpolate_track_scores(scores, 0.0)
    assert result["prediction"] == 95


def test_factorized_scores_are_normalized():
    result = digit_factorized_log_scores(distribution(95, 55))
    assert abs(sum(math.exp(value) for value in result.values()) - 1.0) < 1e-8


def test_train_selection_prefers_smaller_weight_on_ties():
    selected = select_weight([
        {"digit_weight": 0.5, "accuracy": 0.8, "top5_rate": 0.9,
         "paired_to_zero": {"correct_to_wrong": 0}},
        {"digit_weight": 0.0, "accuracy": 0.8, "top5_rate": 0.9,
         "paired_to_zero": {"correct_to_wrong": 0}},
    ])
    assert selected["digit_weight"] == 0.0


def test_selection_rejects_a_more_accurate_weight_with_regressions():
    selected = select_weight([
        {"digit_weight": 0.5, "accuracy": 0.9, "top5_rate": 1.0,
         "paired_to_zero": {"correct_to_wrong": 1}},
        {"digit_weight": 0.0, "accuracy": 0.8, "top5_rate": 0.9,
         "paired_to_zero": {"correct_to_wrong": 0}},
    ], max_regressions=0)
    assert selected["digit_weight"] == 0.0
