#!/usr/bin/env python3
"""Evaluate a YOLO clean-back classifier as a within-track ranker."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_frame_selector import (
    composite_selection_score,
    crop_observation_scores,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--clean-back-weight", type=float, default=0.70)
    parser.add_argument("--sharpness-weight", type=float, default=0.15)
    parser.add_argument("--size-weight", type=float, default=0.05)
    parser.add_argument("--crop-quality-weight", type=float, default=0.10)
    parser.add_argument("--sharpness-scale", type=float, default=100.0)
    parser.add_argument("--size-scale", type=float, default=160.0)
    args = parser.parse_args()
    manifest_path = Path(args.dataset_manifest).resolve()
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(metadata.get("validation_sequences") or []) & set(metadata.get("frozen_validation_sequences") or []):
        raise ValueError("dataset validation overlaps frozen sequences")
    rows = read_validation_rows(manifest_path.parent / "dataset.csv")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for ranking evaluation") from exc
    model = YOLO(args.model)
    names = model.names
    name_items = names.items() if isinstance(names, dict) else enumerate(names)
    clean_index = next((int(index) for index, name in name_items if name == "clean_back"), None)
    if clean_index is None: raise ValueError(f"model classes do not contain clean_back: {names}")
    paths = [row["dataset_path"] for row in rows]
    predictions = model.predict(paths, batch=args.batch, device=args.device, verbose=False)
    for row, prediction in zip(rows, predictions):
        clean_back = float(prediction.probs.data[clean_index].item())
        sharpness, size = crop_observation_scores(
            row["dataset_path"], args.sharpness_scale, args.size_scale
        )
        row["clean_back_score"] = clean_back
        row["sharpness_score"] = sharpness
        row["size_score"] = size
        crop_quality = floating(row.get("crop_quality"))
        row["crop_quality"] = crop_quality
        row["utility_score"] = composite_selection_score(
            clean_back, sharpness, size, crop_quality,
            {
                "clean_back": args.clean_back_weight,
                "sharpness": args.sharpness_weight,
                "size": args.size_weight,
                "crop_quality": args.crop_quality_weight,
            },
        )
        row["is_labeled_positive"] = row["class_name"] == "clean_back"
        row["frame"] = int(row["frame"])
    metrics = metrics_for(rows, args.top_k, args.min_frame_gap)
    metrics.update({
        "model": str(Path(args.model).resolve()),
        "validation_sequences": metadata["validation_sequences"],
        "frozen_sequences_observed": sorted(
            set(metadata["validation_sequences"]) & set(metadata.get("frozen_validation_sequences") or [])
        ),
        "score_semantics": "weighted_clean_back_sharpness_size_crop_quality",
        "score_weights": {
            "clean_back": args.clean_back_weight,
            "sharpness": args.sharpness_weight,
            "size": args.size_weight,
            "crop_quality": args.crop_quality_weight,
        },
    })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def read_validation_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row.get("part") == "val"]
    if not rows: raise ValueError("dataset has no validation rows")
    return rows


def metrics_for(rows, top_k, min_frame_gap):
    labels = [row["is_labeled_positive"] for row in rows]
    scores = [row["utility_score"] for row in rows]
    grouped = defaultdict(list)
    for row in rows: grouped[(row["sequence"], row["gt_track_id"])].append(row)
    eligible = [group for group in grouped.values() if any(r["is_labeled_positive"] for r in group)]
    hits = sum(topk_hit(group, "utility_score", top_k, min_frame_gap) for group in eligible)
    ordered = sorted(scores)
    return {
        "validation_images": len(rows),
        "clean_back_images": sum(labels),
        "average_precision": average_precision(labels, scores),
        "eligible_tracklets": len(eligible),
        "topk_hit_tracklets": hits,
        "topk_hit_rate": hits / len(eligible) if eligible else 0.0,
        "score_min": ordered[0], "score_median": ordered[len(ordered)//2], "score_max": ordered[-1],
        "top_k": top_k, "min_frame_gap": min_frame_gap,
    }


def average_precision(labels, scores):
    positives = sum(bool(value) for value in labels)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    correct = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            correct += 1
            precision_sum += correct / rank
    return precision_sum / positives


def topk_hit(rows, score_key, top_k, min_frame_gap):
    ranked = sorted(
        rows,
        key=lambda row: (-float(row[score_key]), row["frame"], row.get("crop_path", "")),
    )
    selected = []
    for row in ranked:
        if all(abs(row["frame"] - other["frame"]) >= min_frame_gap for other in selected):
            selected.append(row)
            if len(selected) >= top_k:
                break
    return any(row["is_labeled_positive"] for row in selected)


def floating(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__": main()
