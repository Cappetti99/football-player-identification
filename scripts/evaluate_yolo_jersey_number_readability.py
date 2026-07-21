#!/usr/bin/env python3
"""Evaluate YOLO number-readability classification and within-track ranking."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    args = parser.parse_args()

    manifest_path = Path(args.dataset_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target") != "jersey_number_readability":
        raise ValueError("dataset target is not jersey_number_readability")
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("dataset observes frozen sequences")
    rows = read_validation_rows(manifest_path.parent / "dataset.csv")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for ranking evaluation") from exc
    model = YOLO(args.model)
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    readable_index = next(
        (int(index) for index, name in items if str(name) == "number_readable"), None
    )
    if readable_index is None:
        raise ValueError(f"model classes do not contain number_readable: {names}")

    paths = [row["dataset_path"] for row in rows]
    predictions = model.predict(paths, batch=args.batch, device=args.device, verbose=False)
    for row, prediction in zip(rows, predictions):
        row["frame"] = int(row["frame"])
        row["is_readable"] = row["class_name"] == "number_readable"
        row["readability_score"] = float(prediction.probs.data[readable_index].item())

    metrics = evaluate(rows, args.top_k, args.min_frame_gap)
    metrics.update({
        "model": str(Path(args.model).resolve()),
        "dataset_manifest": str(manifest_path),
        "validation_sequences": manifest["validation_sequences"],
        "frozen_sequences_observed": manifest.get("frozen_sequences_observed") or [],
        "score_semantics": "p_number_readable",
        "top_k": args.top_k,
        "min_frame_gap": args.min_frame_gap,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_scores(output.with_suffix(".scores.csv"), rows)
    print(json.dumps(metrics, indent=2))


def read_validation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row.get("part") == "val"]
    if not rows:
        raise ValueError("dataset has no validation rows")
    return rows


def evaluate(rows, top_k, min_frame_gap):
    labels = [bool(row["is_readable"]) for row in rows]
    scores = [float(row["readability_score"]) for row in rows]
    predicted = [score >= 0.5 for score in scores]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["sequence"], row["gt_track_id"])].append(row)
    eligible = [group for group in grouped.values() if any(row["is_readable"] for row in group)]
    hits = sum(topk_hit(group, top_k, min_frame_gap) for group in eligible)
    ordered = sorted(scores)
    return {
        "validation_images": len(rows),
        "readable_images": sum(labels),
        "unreadable_images": len(rows) - sum(labels),
        "accuracy_at_0_5": sum(a == b for a, b in zip(labels, predicted)) / len(rows),
        "readable_precision_at_0_5": ratio(
            sum(label and pred for label, pred in zip(labels, predicted)), sum(predicted)
        ),
        "readable_recall_at_0_5": ratio(
            sum(label and pred for label, pred in zip(labels, predicted)), sum(labels)
        ),
        "average_precision": average_precision(labels, scores),
        "eligible_tracklets": len(eligible),
        "topk_hit_tracklets": hits,
        "topk_hit_rate": ratio(hits, len(eligible)),
        "score_min": ordered[0],
        "score_median": ordered[len(ordered) // 2],
        "score_max": ordered[-1],
    }


def topk_hit(rows, top_k, min_frame_gap):
    ranked = sorted(
        rows,
        key=lambda row: (-float(row["readability_score"]), row["frame"], row["crop_path"]),
    )
    selected = []
    for row in ranked:
        if all(abs(row["frame"] - other["frame"]) >= min_frame_gap for other in selected):
            selected.append(row)
            if len(selected) >= top_k:
                break
    return any(row["is_readable"] for row in selected)


def average_precision(labels, scores):
    positives = sum(labels)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, labels), reverse=True)
    correct = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            correct += 1
            total += correct / rank
    return total / positives


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_scores(path, rows):
    fields = [
        "audit_id", "sequence", "gt_track_id", "frame", "crop_path",
        "class_name", "readability_score",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


if __name__ == "__main__":
    main()
