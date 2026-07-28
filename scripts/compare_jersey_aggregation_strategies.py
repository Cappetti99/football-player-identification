#!/usr/bin/env python3
"""Compare multi-frame aggregation strategies on already-computed CTC scores.

Motivation: the frozen/test evaluation shows a persistent gap between
GT-in-top5 (67.31% on test) and top-1 accuracy (57.11%). The correct number
is often present in a track's per-frame candidate distributions but does not
win the current aggregation (a weighted sum of log-probabilities across
frames, i.e. a product-of-experts). A single overconfident, wrong frame can
dominate that sum even when the correct candidate is the plurality winner
across the other frames (the "9 confused as 5, overconfident" failure mode
already documented for Inter-Juve).

This script takes the raw per-frame scores dumped by
`evaluate_jersey_number_region_ctc_ocr_run.py --dump-raw-scores` and replays
four aggregation strategies against the *same* per-frame CTC outputs, with
zero new model inference:

  baseline        the current production aggregation (ft.features.jersey_number_ctc.aggregate_frames)
  median          per-candidate median log-probability across frames
  majority_vote   plurality of each frame's own top-1 candidate, ties broken
                  by summed log-probability of the tied candidates
  trimmed_sum     weighted sum like baseline, but drops each candidate's
                  single lowest (most adversarial) per-frame log-probability
                  before summing -- directly targets the one-bad-frame failure

Must be pointed at the frozen or development raw-scores dump, never at the
locked GSR test surface, mirroring the same rule already enforced by
audit_jersey_number_region_detector_coverage.py.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.jersey_number_ctc import aggregate_frames  # noqa: E402

FORBIDDEN_SURFACE_MARKERS = ("gsr_test", "shared_surface")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scores", required=True, help="output of --dump-raw-scores")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-test-surface",
        action="store_true",
        help="required to point this at a path containing gsr_test/shared_surface markers",
    )
    args = parser.parse_args()

    if not args.allow_test_surface and any(
        marker in args.raw_scores for marker in FORBIDDEN_SURFACE_MARKERS
    ):
        parser.error(
            f"{args.raw_scores} looks like the locked GSR test surface. Comparing "
            "aggregation strategies there is a re-evaluation of the frozen test, "
            "which the project rules forbid. Point --raw-scores at the frozen or "
            "development dump instead, or pass --allow-test-surface if this is "
            "intentional and already approved."
        )

    tracks = json.loads(Path(args.raw_scores).read_text())
    if not tracks:
        raise RuntimeError("no tracks found in raw-scores file")

    strategies = {
        "baseline": aggregate_baseline,
        "median": aggregate_median,
        "majority_vote": aggregate_majority_vote,
        "trimmed_sum": aggregate_trimmed_sum,
    }

    rows = []
    for key, entry in tracks.items():
        scores = entry["scores"]
        weights = entry["weights"]
        truth = entry.get("gt_jersey_number")
        row = {
            "track": key,
            "sequence": entry.get("sequence"),
            "gt_track_id": entry.get("gt_track_id"),
            "gt_jersey_number": truth,
            "recognized_frames": len(scores),
        }
        for name, strategy in strategies.items():
            result = strategy(scores, weights)
            assigned = result["prediction"] is not None
            row[f"{name}_prediction"] = result["prediction"]
            row[f"{name}_confidence"] = result["confidence"]
            row[f"{name}_assigned"] = assigned
            row[f"{name}_correct"] = assigned and truth is not None and int(result["prediction"]) == int(truth)
            row[f"{name}_gt_in_top5"] = (
                truth is not None and str(truth) in dict(list(result["scores"].items())[:5])
            )
        rows.append(row)

    summary = {name: summarize(rows, name) for name in strategies}
    transitions = pairwise_transitions(rows, "baseline", strategies)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "aggregation_comparison.csv", rows)
    (output / "aggregation_comparison_summary.json").write_text(
        json.dumps({"summary": summary, "transitions_vs_baseline": transitions}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "transitions_vs_baseline": transitions}, indent=2))
    print(f"\ncsv={output / 'aggregation_comparison.csv'}")
    print(f"summary={output / 'aggregation_comparison_summary.json'}")


def aggregate_baseline(scores, weights):
    return aggregate_frames(scores, weights)


def aggregate_median(scores, weights):
    if not scores:
        return {"prediction": None, "confidence": 0.0, "margin": 0.0, "scores": {}}
    candidates = sorted(set.intersection(*(set(s) for s in scores)))
    medians = {candidate: median([float(s[candidate]) for s in scores]) for candidate in candidates}
    return rank_to_result(medians)


def aggregate_majority_vote(scores, weights):
    if not scores:
        return {"prediction": None, "confidence": 0.0, "margin": 0.0, "scores": {}}
    votes = {}
    logprob_sums = {}
    for frame_scores in scores:
        top1 = max(frame_scores.items(), key=lambda item: item[1])[0]
        votes[top1] = votes.get(top1, 0) + 1
    for frame_scores in scores:
        for candidate, value in frame_scores.items():
            logprob_sums[candidate] = logprob_sums.get(candidate, 0.0) + float(value)
    total_votes = sum(votes.values())
    # Rank by (vote share, mean log-probability) so every candidate seen in
    # any frame's distribution gets a well-defined position for gt_in_top5,
    # not just the ones that won at least one frame's local top-1.
    all_candidates = set.intersection(*(set(s) for s in scores))
    ranked = {
        candidate: (
            votes.get(candidate, 0) / total_votes if total_votes else 0.0,
            logprob_sums.get(candidate, float("-inf")) / len(scores),
        )
        for candidate in all_candidates
    }
    ordering = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
    winner, (vote_share, _) = ordering[0]
    runner_up_share = ordering[1][1][0] if len(ordering) > 1 else 0.0
    return {
        "prediction": int(winner),
        "confidence": vote_share,
        "margin": vote_share - runner_up_share,
        "scores": {candidate: value[0] for candidate, value in ordering},
    }


def aggregate_trimmed_sum(scores, weights, drop_worst=1):
    if not scores:
        return {"prediction": None, "confidence": 0.0, "margin": 0.0, "scores": {}}
    weights = weights or [1.0] * len(scores)
    candidates = sorted(set.intersection(*(set(s) for s in scores)))
    totals = {}
    for candidate in candidates:
        weighted = sorted(
            max(0.05, float(weight)) * float(frame_scores[candidate])
            for frame_scores, weight in zip(scores, weights)
        )
        kept = weighted[drop_worst:] if len(weighted) > drop_worst else weighted
        totals[candidate] = sum(kept)
    return rank_to_result(totals)


def rank_to_result(totals):
    ordering = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    winner, best = ordering[0]
    runner_up = ordering[1][1] if len(ordering) > 1 else 0.0
    return {
        "prediction": int(winner),
        "confidence": best,
        "margin": best - runner_up,
        "scores": dict(ordering),
    }


def median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def summarize(rows, name):
    assigned = [row for row in rows if row[f"{name}_assigned"]]
    correct = [row for row in assigned if row[f"{name}_correct"]]
    gt_top5 = [row for row in rows if row[f"{name}_gt_in_top5"]]
    return {
        "tracklets": len(rows),
        "assigned": len(assigned),
        "coverage": ratio(len(assigned), len(rows)),
        "correct": len(correct),
        "accuracy_assigned": ratio(len(correct), len(assigned)),
        "accuracy_all": ratio(len(correct), len(rows)),
        "gt_in_top5_rate": ratio(len(gt_top5), len(rows)),
    }


def pairwise_transitions(rows, baseline_name, strategies):
    transitions = {}
    for name in strategies:
        if name == baseline_name:
            continue
        wrong_to_correct = sum(
            1 for row in rows if not row[f"{baseline_name}_correct"] and row[f"{name}_correct"]
        )
        correct_to_wrong = sum(
            1 for row in rows if row[f"{baseline_name}_correct"] and not row[f"{name}_correct"]
        )
        transitions[name] = {
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_gain": wrong_to_correct - correct_to_wrong,
        }
    return transitions


def ratio(a, b):
    return a / b if b else 0.0


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
