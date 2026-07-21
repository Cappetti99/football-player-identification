import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_jersey_thesis_benchmark.py"
SPEC = importlib.util.spec_from_file_location("jersey_thesis_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AUDIT_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_jersey_ctc_qualitative_audit.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("jersey_ctc_qualitative", AUDIT_SCRIPT)
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)


def rows(method, values):
    output = []
    for sequence, correct, assigned, top5 in values:
        output.append({
            "split": "development",
            "method": method,
            "sequence": sequence,
            "gt_track_id": "1",
            "eval_track_id": "1",
            "gt_jersey": 10,
            "pred_jersey": 10 if correct else (11 if assigned else ""),
            "assigned": int(assigned),
            "correct": int(correct),
            "wrong": int(assigned and not correct),
            "abstained": int(not assigned),
            "gt_in_selected_candidates": int(top5),
            "votes": "",
            "winner_margin": "",
            "confidence": "",
        })
    return output


def test_aggregate_and_bootstrap_are_sequence_level_and_deterministic():
    tracks = rows("baseline", [("A", 0, 0, 0), ("B", 1, 1, 1)])
    tracks += rows("candidate", [("A", 1, 1, 1), ("B", 1, 1, 1)])
    per_sequence = MODULE.build_per_sequence(tracks)
    aggregate = MODULE.build_aggregate(tracks, per_sequence)
    candidate = next(row for row in aggregate if row["method"] == "candidate")
    assert candidate["correct"] == 2
    assert candidate["coverage"] == 1.0
    first = MODULE.build_bootstrap(tracks, "baseline", samples=200, seed=7)
    second = MODULE.build_bootstrap(tracks, "baseline", samples=200, seed=7)
    assert first == second
    assert first["unit"] == "sequence"
    assert first["comparisons"]["candidate"]["accuracy_all"]["mean_delta"] > 0


def test_transition_names_regressions_and_recoveries():
    assert MODULE.transition(10, 11, 10) == "correct_to_wrong"
    assert MODULE.transition(None, 10, 10) == "recovered_correct"
    assert MODULE.transition(None, 11, 10) == "new_wrong_emission"
    assert MODULE.transition(11, None, 10) == "wrong_to_abstention"


def test_transition_summary_reports_paired_recoveries():
    transitions = [
        {"candidate": "new", "transition": "recovered_correct"}
        for _ in range(45)
    ] + [
        {"candidate": "new", "transition": "correct_to_wrong"}
        for _ in range(5)
    ]
    summary = MODULE.build_transition_summary(transitions)["comparisons"]["new"]
    assert summary["net_correct_gain"] == 40
    assert summary["discordant_tracks"] == 50
    assert summary["paired_binomial_p_value"] < 1e-8


def test_qualitative_audit_indexes_selected_crops_by_eval_id():
    payload = {
        "tracklets": {
            "x": {
                "display_track_id": 12,
                "selected_crops": [{"crop_path": "example.jpg"}],
            }
        }
    }
    assert AUDIT.crops_by_eval_id(payload)["12"][0]["crop_path"] == "example.jpg"
