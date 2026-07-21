#!/usr/bin/env python3
"""Offline ablation of top-k legibility-filtered jersey voting on GSR."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--oracle-tracklets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-frame-gap", type=int, default=0)
    args = parser.parse_args()

    crops = read_csv(Path(args.scores))
    tracklets = json.loads(Path(args.oracle_tracklets).read_text())
    truth = {
        str(row["gt_track_id"]): integer(row.get("gt_jersey"))
        for row in tracklets if integer(row.get("gt_jersey")) is not None
    }
    current = {
        str(row["gt_track_id"]): integer(row.get("current_prediction"))
        for row in tracklets if integer(row.get("gt_jersey")) is not None
    }
    grouped = defaultdict(list)
    for row in crops:
        track = str(row.get("gt_track_id"))
        if track not in truth:
            continue
        grouped[track].append(normalize(row))

    policies = {}
    decisions = []
    for top_k in args.top_k:
        for policy in ("confidence", "legibility_confidence", "agreement"):
            name = f"top{top_k}_{policy}"
            correct = emitted = 0
            for track, gt_jersey in truth.items():
                selected = select_crops(
                    grouped.get(track, []), top_k,
                    min_score=args.min_score,
                    min_frame_gap=args.min_frame_gap,
                )
                prediction, margin, votes = vote(selected, policy)
                emitted += int(prediction is not None)
                correct += int(prediction == gt_jersey)
                decisions.append({
                    "policy": name,
                    "gt_track_id": track,
                    "gt_jersey": gt_jersey,
                    "current_prediction": current.get(track),
                    "prediction": prediction,
                    "correct": prediction == gt_jersey,
                    "selected_crops": len(selected),
                    "selected_frames": [row["frame"] for row in selected],
                    "selected_winners": [row["winner"] for row in selected],
                    "margin": margin,
                    "winner_votes": votes,
                })
            policies[name] = {
                "correct": correct,
                "total": len(truth),
                "accuracy": ratio(correct, len(truth)),
                "emitted": emitted,
                "coverage": ratio(emitted, len(truth)),
                "delta_correct_vs_current": correct - sum(current.get(t) == j for t, j in truth.items()),
            }

    summary = {
        "visible_gt_tracklets": len(truth),
        "current_correct": sum(current.get(track) == jersey for track, jersey in truth.items()),
        "current_accuracy": ratio(sum(current.get(track) == jersey for track, jersey in truth.items()), len(truth)),
        "min_legibility_score": args.min_score,
        "min_frame_gap": args.min_frame_gap,
        "policies": policies,
        "best_policy": max(policies, key=lambda name: policies[name]["correct"]) if policies else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "decisions": decisions}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def select_crops(rows, limit, min_score, min_frame_gap):
    ranked = sorted(rows, key=lambda row: row["legibility_score"], reverse=True)
    selected = []
    for row in ranked:
        if row["legibility_score"] < min_score:
            continue
        if min_frame_gap and any(abs(row["frame"] - other["frame"]) < min_frame_gap for other in selected):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def vote(rows, policy):
    scores = defaultdict(float)
    counts = Counter()
    for row in rows:
        winner = row["winner"]
        if winner is None:
            continue
        if policy == "confidence":
            weight = row["winner_confidence"]
        elif policy == "legibility_confidence":
            weight = row["legibility_score"] * row["winner_confidence"]
        else:
            weight = row["legibility_score"] * row["winner_confidence"] * max(1, row["agreement"])
        scores[winner] += max(0.001, weight)
        counts[winner] += 1
    if not scores:
        return None, None, 0
    ranked = sorted(scores.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
    winner, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(scores.values())
    return winner, (winner_score - runner_up) / total if total else None, counts[winner]


def normalize(row):
    return {
        "frame": integer(row.get("frame")) or 0,
        "winner": integer(row.get("winner")),
        "winner_confidence": floating(row.get("winner_confidence")),
        "legibility_score": floating(row.get("legibility_score")),
        "agreement": integer(row.get("agreement")) or 0,
    }


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def integer(value):
    try: return int(float(value))
    except (TypeError, ValueError): return None


def floating(value):
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def ratio(a, b): return float(a / b) if b else None


if __name__ == "__main__":
    main()
