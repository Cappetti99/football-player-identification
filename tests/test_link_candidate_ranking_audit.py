from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_gsr_link_candidate_ranking.py"
SPEC = spec_from_file_location("link_candidate_audit", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_soft_pitch_cost_can_correct_ambiguous_baseline_winner():
    rows = [
        candidate(1, baseline_score=80, speed=24, correct=False),
        candidate(2, baseline_score=84, speed=6, correct=True),
    ]

    baseline = MODULE.evaluate(rows, weight=0.0)
    reranked = MODULE.evaluate(rows, weight=0.1)

    assert baseline["recall_at_1"] == 0.0
    assert reranked["recall_at_1"] == 1.0
    assert reranked["baseline_wrong_to_correct"] == 1
    assert reranked["baseline_correct_to_wrong"] == 0


def test_missing_pitch_keeps_baseline_order():
    rows = [
        candidate(1, baseline_score=80, speed=None, correct=True),
        candidate(2, baseline_score=84, speed=6, correct=False),
    ]

    result = MODULE.evaluate(rows, weight=1.0)

    assert result["recall_at_1"] == 1.0
    assert result["net_winner_corrections"] == 0


def candidate(source, baseline_score, speed, correct):
    return {
        "sequence": "SNGS-test",
        "from_track_id": source,
        "to_track_id": 9,
        "gap": 2,
        "baseline_score": baseline_score,
        "required_speed_mps": speed,
        "correct_link_offline": correct,
    }
