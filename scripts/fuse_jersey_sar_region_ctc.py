#!/usr/bin/env python3
"""Conservatively fuse primary SAR and jersey-region CTC predictions."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EMPTY = {None, "", "None", "null", "nan", "-1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="SAR run directory or predictions.csv")
    parser.add_argument("--candidate", required=True, help="region CTC run directory or predictions.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-min-confidence", type=float, default=0.90)
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.candidate_min_confidence <= 1.0:
        parser.error("--candidate-min-confidence must be between 0 and 1")

    baseline_path = prediction_path(args.baseline)
    candidate_path = prediction_path(args.candidate)
    baseline = read_predictions(baseline_path)
    candidate = read_predictions(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError(surface_mismatch(baseline, candidate))

    fused, decisions = fuse_rows(
        baseline,
        candidate,
        candidate_min_confidence=args.candidate_min_confidence,
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "predictions.csv", fused)
    write_csv(output / "fusion_provenance.csv", decisions)

    metrics = summarize(fused, decisions)
    manifest = {
        "method": "conservative_sar_region_ctc_fusion_v1",
        "rule": {
            "candidate_min_confidence": args.candidate_min_confidence,
            "candidate_may_override_assigned_baseline": True,
            "candidate_may_fill_baseline_abstention": False,
            "agreement_keeps_baseline": True,
        },
        "inputs": {
            "baseline_predictions": artifact(baseline_path),
            "candidate_predictions": artifact(candidate_path),
            "ctc_checkpoint": artifact(Path(args.ctc_checkpoint).resolve()),
            "detector_checkpoint": artifact(Path(args.detector_checkpoint).resolve()),
        },
        "metrics": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def prediction_path(value):
    path = Path(value).resolve()
    return path / "predictions.csv" if path.is_dir() else path


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        key = str(row["sequence"]), str(row["gt_track_id"])
        if key in output:
            raise ValueError(f"duplicate prediction key: {key}")
        output[key] = row
    return output


def fuse_rows(baseline, candidate, candidate_min_confidence):
    fused, decisions = [], []
    for key in sorted(baseline):
        before_row = baseline[key]
        candidate_row = candidate[key]
        truth = integer(before_row.get("gt_jersey_number"))
        if truth != integer(candidate_row.get("gt_jersey_number")):
            raise ValueError(f"ground truth mismatch for {key}")
        before = integer(before_row.get("pred_jersey_number"))
        proposed = integer(candidate_row.get("pred_jersey_number"))
        confidence = floating(candidate_row.get("confidence"))

        if before is None:
            chosen, source, reason = None, "baseline", "baseline_abstention_preserved"
        elif proposed is None:
            chosen, source, reason = before, "baseline", "candidate_abstained"
        elif proposed == before:
            chosen, source, reason = before, "agreement", "recognizers_agree"
        elif confidence >= candidate_min_confidence:
            chosen, source, reason = proposed, "region_ctc", "high_confidence_override"
        else:
            chosen, source, reason = before, "baseline", "candidate_below_threshold"

        row = dict(before_row)
        row.update({
            "pred_jersey_number": "" if chosen is None else chosen,
            "assigned": chosen is not None,
            "correct": chosen is not None and chosen == truth,
            "fusion_source": source,
            "fusion_reason": reason,
            "fusion_candidate_confidence": confidence,
            "fusion_candidate_margin": floating(candidate_row.get("winner_margin")),
            "fusion_candidate_recognized_frames": integer(candidate_row.get("recognized_frames")) or 0,
            "fusion_candidate_min_confidence": candidate_min_confidence,
        })
        fused.append(row)
        decisions.append({
            "sequence": key[0],
            "gt_track_id": key[1],
            "gt_jersey_number": truth,
            "baseline_prediction": "" if before is None else before,
            "candidate_prediction": "" if proposed is None else proposed,
            "fused_prediction": "" if chosen is None else chosen,
            "candidate_confidence": confidence,
            "candidate_margin": floating(candidate_row.get("winner_margin")),
            "candidate_recognized_frames": integer(candidate_row.get("recognized_frames")) or 0,
            "source": source,
            "reason": reason,
            "transition": transition(before, chosen, truth),
        })
    return fused, decisions


def summarize(rows, decisions):
    assigned = [row for row in rows if boolean(row.get("assigned"))]
    correct = [row for row in assigned if boolean(row.get("correct"))]
    transitions = Counter(row["transition"] for row in decisions)
    sources = Counter(row["source"] for row in decisions)
    baseline_summary = decision_summary(decisions, "baseline_prediction")
    candidate_summary = decision_summary(decisions, "candidate_prediction")
    return {
        "tracklets": len(rows),
        "assigned": len(assigned),
        "correct": len(correct),
        "wrong": len(assigned) - len(correct),
        "coverage": ratio(len(assigned), len(rows)),
        "accuracy_assigned": ratio(len(correct), len(assigned)),
        "accuracy_all": ratio(len(correct), len(rows)),
        "overrides": sources["region_ctc"],
        "sources": dict(sources),
        "transitions": dict(transitions),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "net_correct_gain_vs_baseline": len(correct) - baseline_summary["correct"],
        "zero_correct_to_wrong": transitions["correct_to_wrong"] == 0,
        "zero_new_wrong_emissions": transitions["new_wrong_emission"] == 0,
    }


def decision_summary(rows, prediction_field):
    predictions = [integer(row.get(prediction_field)) for row in rows]
    truths = [integer(row.get("gt_jersey_number")) for row in rows]
    assigned = sum(value is not None for value in predictions)
    correct = sum(value is not None and value == truth for value, truth in zip(predictions, truths))
    return {
        "assigned": assigned,
        "correct": correct,
        "wrong": assigned - correct,
        "coverage": ratio(assigned, len(rows)),
        "accuracy_all": ratio(correct, len(rows)),
        "accuracy_assigned": ratio(correct, assigned),
    }


def transition(before, after, truth):
    before_correct = before is not None and before == truth
    after_correct = after is not None and after == truth
    if before == after:
        return "unchanged_correct" if before_correct else "unchanged"
    if before_correct and not after_correct:
        return "correct_to_wrong" if after is not None else "correct_to_abstention"
    if after_correct:
        return "recovered_correct"
    if before is None and after is not None:
        return "new_wrong_emission"
    if before is not None and after is None:
        return "wrong_to_abstention"
    return "wrong_to_wrong"


def artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def surface_mismatch(baseline, candidate):
    missing_candidate = sorted(set(baseline) - set(candidate))[:20]
    missing_baseline = sorted(set(candidate) - set(baseline))[:20]
    return f"prediction surfaces differ: missing_candidate={missing_candidate} missing_baseline={missing_baseline}"


def integer(value):
    if value in EMPTY:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def floating(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def boolean(value):
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def ratio(a, b):
    return a / b if b else 0.0


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
