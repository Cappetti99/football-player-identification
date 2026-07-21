#!/usr/bin/env python3
"""Evaluate FT linker candidate ranking with offline-only GSR identity labels."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[0, .05, .1, .2, .5, 1.0])
    parser.add_argument("--baseline-distance-scale", type=float, default=160.0)
    parser.add_argument("--free-speed-mps", type=float, default=8.0)
    args = parser.parse_args()

    artifact_root = Path(args.artifacts_root)
    evaluation_root = Path(args.evaluation_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    discovered = []
    skipped = []
    for sequence_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        linking_path = sequence_dir / "metadata" / f"{sequence}_linking.json"
        matches_path = evaluation_root / sequence / "frame_matches.csv"
        if not linking_path.is_file() or not matches_path.is_file():
            skipped.append({
                "sequence": sequence,
                "linking_exists": linking_path.is_file(),
                "matches_exists": matches_path.is_file(),
            })
            continue
        discovered.append(sequence)
        gt_by_raw = read_gt_majority(matches_path)
        diagnostics = json.loads(linking_path.read_text(encoding="utf-8"))
        for row in diagnostics.get("candidates", []):
            source = int(row["from_track_id"])
            target = int(row["to_track_id"])
            source_gt, target_gt = gt_by_raw.get(source), gt_by_raw.get(target)
            item = dict(row)
            item.update({
                "sequence": sequence,
                "from_gt_track_id_offline": source_gt,
                "to_gt_track_id_offline": target_gt,
                "correct_link_offline": (
                    None if source_gt is None or target_gt is None else source_gt == target_gt
                ),
                "gt_usage": "offline_evaluation_only",
            })
            candidates.append(item)

    if not discovered:
        raise RuntimeError(
            "No completed sequences found. Expected exact <sequence>_linking.json "
            f"and frame_matches.csv pairs; skipped={skipped}"
        )
    if not candidates:
        raise RuntimeError(
            "Completed sequences were found but linking diagnostics contain no candidates. "
            "Confirm the benchmark was run after syncing the candidate-diagnostics linker; "
            f"sequences={discovered}"
        )

    write_csv(output / "candidates.csv", candidates)
    sweep = [
        evaluate(candidates, weight, args.baseline_distance_scale, args.free_speed_mps)
        for weight in args.weights
    ]
    write_csv(output / "weight_sweep.csv", sweep)
    payload = {
        "mode": "offline_candidate_ranking_audit",
        "mutates_tracking": False,
        "gt_usage": "offline evaluation only; absent from candidate generation and scores",
        "candidate_rows": len(candidates),
        "sequence_count": len({row["sequence"] for row in candidates}),
        "discovered_sequences": discovered,
        "skipped_sequences": skipped,
        "baseline_distance_scale": args.baseline_distance_scale,
        "free_speed_mps": args.free_speed_mps,
        "weight_sweep": sweep,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def read_gt_majority(path):
    votes = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = integer(row.get("raw_pred_track_id") or row.get("raw_track_id"))
            gt = row.get("gt_track_id")
            if raw is not None and gt not in {None, "", "None", "null"}:
                votes[raw][str(gt)] += 1
    return {raw: counts.most_common(1)[0][0] for raw, counts in votes.items() if counts}


def evaluate(rows, weight, distance_scale=160.0, free_speed=8.0):
    groups = defaultdict(list)
    for row in rows:
        if row.get("correct_link_offline") is None:
            continue
        item = dict(row)
        baseline = number(row.get("baseline_score")) / distance_scale
        speed = optional_number(row.get("required_speed_mps"))
        pitch_penalty = 0.0 if speed is None else max(0.0, speed - free_speed) / free_speed
        item["soft_score"] = baseline + float(weight) * pitch_penalty
        groups[(row["sequence"], int(row["to_track_id"]))].append(item)

    evaluable = correct_available = recall1 = recall3 = recall5 = 0
    reciprocal_rank = 0.0
    baseline_correct_to_wrong = baseline_wrong_to_correct = 0
    for candidates in groups.values():
        evaluable += 1
        baseline_order = sorted(candidates, key=baseline_sort_key)
        order = sorted(candidates, key=lambda row: (row["soft_score"],) + baseline_sort_key(row))
        correct_ranks = [rank for rank, row in enumerate(order, 1) if row["correct_link_offline"]]
        if correct_ranks:
            correct_available += 1
            best = min(correct_ranks)
            recall1 += best <= 1
            recall3 += best <= 3
            recall5 += best <= 5
            reciprocal_rank += 1.0 / best
        baseline_correct = bool(baseline_order[0]["correct_link_offline"])
        reranked_correct = bool(order[0]["correct_link_offline"])
        baseline_correct_to_wrong += baseline_correct and not reranked_correct
        baseline_wrong_to_correct += not baseline_correct and reranked_correct
    return {
        "pitch_weight": float(weight),
        "evaluable_targets": evaluable,
        "correct_candidate_available": correct_available,
        "oracle_coverage": ratio(correct_available, evaluable),
        "recall_at_1": ratio(recall1, evaluable),
        "recall_at_3": ratio(recall3, evaluable),
        "recall_at_5": ratio(recall5, evaluable),
        "mrr": ratio(reciprocal_rank, evaluable),
        "baseline_wrong_to_correct": baseline_wrong_to_correct,
        "baseline_correct_to_wrong": baseline_correct_to_wrong,
        "net_winner_corrections": baseline_wrong_to_correct - baseline_correct_to_wrong,
    }


def baseline_sort_key(row):
    return (
        number(row.get("baseline_score")),
        integer(row.get("gap")) or 0,
        integer(row.get("from_track_id")) or 0,
    )


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["sequence"])
        writer.writeheader()
        writer.writerows(rows)


def integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def number(value):
    return float(value or 0.0)


def optional_number(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def ratio(a, b):
    return a / b if b else None


if __name__ == "__main__":
    main()
