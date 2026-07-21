#!/usr/bin/env python3
"""Compare frozen GSR jersey-selector OCR runs track by track."""

import argparse
import csv
import json
from pathlib import Path


EMPTY = {None, "", "None", "null", "-1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = {"baseline": Path(args.baseline), "candidate": Path(args.candidate)}
    predictions = {name: load_predictions(root / "predictions.csv") for name, root in roots.items()}
    if set(predictions["baseline"]) != set(predictions["candidate"]):
        missing_candidate = sorted(set(predictions["baseline"]) - set(predictions["candidate"]))
        missing_baseline = sorted(set(predictions["candidate"]) - set(predictions["baseline"]))
        raise ValueError(
            f"run surfaces differ: missing_candidate={missing_candidate[:20]} "
            f"missing_baseline={missing_baseline[:20]}"
        )
    availability = {
        name: correct_candidate_availability(root / "ocr_diagnostics.json", rows)
        for name, (root, rows) in (
            (label, (roots[label], predictions[label])) for label in roots
        )
    }
    summaries = {
        name: summarize(rows, availability[name]) for name, rows in predictions.items()
    }
    deltas = compare(predictions["baseline"], predictions["candidate"])
    gates = {
        "correct_not_lower": summaries["candidate"]["correct"] >= summaries["baseline"]["correct"],
        "wrong_not_higher": summaries["candidate"]["wrong"] <= summaries["baseline"]["wrong"],
        "coverage_not_lower": summaries["candidate"]["coverage"] >= summaries["baseline"]["coverage"],
        "zero_correct_to_wrong": not any(row["transition"] == "correct_to_wrong" for row in deltas),
        "top5_correct_candidate_not_lower": (
            summaries["candidate"]["correct_candidate_available"]
            >= summaries["baseline"]["correct_candidate_available"]
        ),
    }
    result = {
        "baseline": str(roots["baseline"]),
        "candidate": str(roots["candidate"]),
        "summaries": summaries,
        "deltas": deltas,
        "promotion_gates": gates,
        "passes_all_gates": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def load_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["gt_track_id"]))
        if key in output:
            raise ValueError(f"duplicate prediction key: {key}")
        output[key] = {
            **row,
            "eval_track_id": str(row["eval_track_id"]),
            "gt": integer(row["gt_jersey_number"]),
            "pred": integer(row["pred_jersey_number"]),
        }
    return output


def correct_candidate_availability(path, predictions):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_eval = {row["eval_track_id"]: row for row in predictions.values()}
    output = {}
    for diagnostic in (payload.get("tracklets") or {}).values():
        eval_id = str(diagnostic.get("display_track_id"))
        prediction = by_eval.get(eval_id)
        if prediction is None:
            continue
        numbers = {
            integer(row.get("number"))
            for row in diagnostic.get("detections", [])
            if integer(row.get("number")) is not None
        }
        key = (prediction["sequence"], prediction["gt_track_id"])
        output[key] = prediction["gt"] in numbers
    return output


def summarize(rows, availability):
    values = list(rows.values())
    assigned = [row for row in values if row["pred"] is not None]
    correct = [row for row in assigned if row["pred"] == row["gt"]]
    wrong = [row for row in assigned if row["pred"] != row["gt"]]
    available = sum(bool(availability.get(key)) for key in rows)
    return {
        "tracklets": len(values),
        "assigned": len(assigned),
        "correct": len(correct),
        "wrong": len(wrong),
        "coverage": ratio(len(assigned), len(values)),
        "accuracy_assigned": ratio(len(correct), len(assigned)),
        "accuracy_all": ratio(len(correct), len(values)),
        "correct_candidate_available": available,
        "correct_candidate_available_rate": ratio(available, len(values)),
    }


def compare(baseline, candidate):
    output = []
    for key in sorted(baseline):
        before = baseline[key]["pred"]
        after = candidate[key]["pred"]
        truth = baseline[key]["gt"]
        if truth != candidate[key]["gt"]:
            raise ValueError(f"ground truth mismatch for {key}")
        if before == after:
            continue
        output.append({
            "sequence": key[0],
            "gt_track_id": key[1],
            "gt_jersey": truth,
            "baseline": before,
            "candidate": after,
            "transition": transition(before, after, truth),
        })
    return output


def transition(before, after, truth):
    before_correct = before is not None and before == truth
    after_correct = after is not None and after == truth
    if before_correct and after_correct:
        return "correct_changed_correct"
    if before_correct and after is None:
        return "correct_to_abstention"
    if before_correct and not after_correct:
        return "correct_to_wrong"
    if not before_correct and after_correct:
        return "recovered_correct"
    if before is None and after is not None:
        return "new_wrong_emission"
    if before is not None and after is None:
        return "wrong_to_abstention"
    return "wrong_to_wrong"


def integer(value):
    if value in EMPTY:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
