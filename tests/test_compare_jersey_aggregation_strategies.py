"""Regression test for the aggregation-strategy comparison tool.

Fixes a synthetic adversarial case: 3 frames weakly favor the correct number,
2 frames strongly (overconfidently) favor a wrong one. The current production
aggregation (weighted sum of log-probabilities) is fooled by the two
overconfident frames even though the correct number wins by frame count.
This is the failure mode already documented for Inter-Juve ("9" read as "5",
overconfident). median and majority_vote should recover the correct answer
here; if a future change to aggregate_frames or these strategies breaks that,
this test catches it.
"""
from scripts.compare_jersey_aggregation_strategies import (
    aggregate_baseline,
    aggregate_majority_vote,
    aggregate_median,
    aggregate_trimmed_sum,
)


def _dist(winner, second, margin):
    base = {"7": -5.0, "5": -5.0, "17": -5.0, "55": -5.0}
    base[winner] = -0.05
    base[second] = -0.05 - margin
    return base


SCORES = [
    _dist("7", "17", 0.1),
    _dist("7", "17", 0.1),
    _dist("7", "55", 0.1),
    _dist("5", "55", 0.01),
    _dist("5", "55", 0.001),
]
WEIGHTS = [0.5, 0.5, 0.5, 0.99, 0.999]


def test_baseline_is_fooled_by_the_overconfident_minority():
    result = aggregate_baseline(SCORES, WEIGHTS)
    # "55" wins, not "7": it is the runner-up in three different frames
    # (frames 1-3), so it accumulates enough log-probability mass across the
    # sum to beat the true plurality winner "7" -- a second illustration of
    # the same underlying problem (weighted log-sum can be won by a candidate
    # that never actually tops a single frame).
    assert result["prediction"] == 55


def test_median_recovers_the_plurality_winner():
    result = aggregate_median(SCORES, WEIGHTS)
    assert result["prediction"] == 7


def test_majority_vote_recovers_the_plurality_winner():
    result = aggregate_majority_vote(SCORES, WEIGHTS)
    assert result["prediction"] == 7


def test_trimmed_sum_drop_one_is_not_enough_for_two_bad_frames():
    result = aggregate_trimmed_sum(SCORES, WEIGHTS, drop_worst=1)
    assert result["prediction"] == 55


def test_trimmed_sum_drop_two_recovers_the_plurality_winner():
    result = aggregate_trimmed_sum(SCORES, WEIGHTS, drop_worst=2)
    assert result["prediction"] == 7
