#!/usr/bin/env python3
"""Evaluate multi-frame digit recognition and optionally compare paired CTC rows."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ft.features.jersey_multiframe_digits import (
    ARCHITECTURE,
    build_multiframe_digit_recognizer,
    number_log_probabilities,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--part", choices=["validation"], default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-track-predictions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from scripts.train_jersey_multiframe_digits import JerseyMultiFrameDataset, collate_bags

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("unexpected multi-frame checkpoint architecture")
    root = Path(args.dataset).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("evaluation dataset observes frozen sequences")
    namespace = argparse.Namespace(
        image_height=metadata["image_size"][0], image_width=metadata["image_size"][1],
        frames_per_track=metadata["frames_per_track"],
    )
    dataset = JerseyMultiFrameDataset(root / f"{args.part}.jsonl", namespace, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_bags)
    model = build_multiframe_digit_recognizer(pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for images, mask, jerseys, source_rows in loader:
            outputs = model(images.to(args.device), mask.to(args.device))
            probabilities = number_log_probabilities(outputs).exp().cpu()
            attention = outputs["attention"].cpu()
            length_probabilities = outputs["length_logits"].softmax(dim=1).cpu()
            tens_probabilities = outputs["tens_logits"].softmax(dim=1).cpu()
            units_probabilities = outputs["units_logits"].softmax(dim=1).cpu()
            for index, source in enumerate(source_rows):
                ranking = sorted(enumerate(probabilities[index].tolist()), key=lambda item: (-item[1], item[0]))
                prediction, confidence = ranking[0]
                runner_up = ranking[1][1]
                truth = int(jerseys[index])
                valid_frames = int(mask[index].sum())
                rows.append({
                    "sequence": source["sequence"],
                    "gt_track_id": source["gt_track_id"],
                    "gt_jersey": truth,
                    "prediction": prediction,
                    "correct": prediction == truth,
                    "confidence": confidence,
                    "margin": confidence - runner_up,
                    "entropy": entropy(probabilities[index].tolist()),
                    "frames": valid_frames,
                    "frame_ids": json.dumps([frame["frame"] for frame in source["frames"][:valid_frames]]),
                    "attention": json.dumps(attention[index, :valid_frames].tolist()),
                    "length_probabilities": json.dumps(length_probabilities[index].tolist()),
                    "tens_probabilities": json.dumps(tens_probabilities[index].tolist()),
                    "units_probabilities": json.dumps(units_probabilities[index].tolist()),
                    "top5": json.dumps(ranking[:5]),
                    "gt_in_top5": any(number == truth for number, _ in ranking[:5]),
                })
    metrics = summarize(rows)
    metrics.update({
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "dataset": str(root), "part": args.part,
    })
    if args.baseline_track_predictions:
        baseline = read_csv(args.baseline_track_predictions)
        metrics["paired_comparison"] = paired_comparison(baseline, rows)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "track_predictions.csv", rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def summarize(rows):
    correct = sum(bool(row["correct"]) for row in rows)
    top5 = sum(bool(row["gt_in_top5"]) for row in rows)
    return {
        "tracklets": len(rows),
        "track_correct": correct,
        "track_accuracy": ratio(correct, len(rows)),
        "track_gt_in_top5": top5,
        "track_gt_in_top5_rate": ratio(top5, len(rows)),
        "mean_confidence": ratio(sum(float(row["confidence"]) for row in rows), len(rows)),
        "expected_calibration_error_10_bin": expected_calibration_error(rows, 10),
        "five_nine_errors": sum(is_five_nine_error(row) for row in rows),
    }


def paired_comparison(baseline_rows, candidate_rows):
    baseline = {(str(row["sequence"]), str(row["gt_track_id"])): row for row in baseline_rows}
    candidate = {(str(row["sequence"]), str(row["gt_track_id"])): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate do not contain the same tracks")
    transitions = {name: 0 for name in (
        "correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"
    )}
    for key in baseline:
        left = truthy(baseline[key]["correct"])
        right = truthy(candidate[key]["correct"])
        transitions[("correct" if left else "wrong") + "_to_" + ("correct" if right else "wrong")] += 1
    return {
        **transitions,
        "net_correct": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
    }


def expected_calibration_error(rows, bins):
    total = max(1, len(rows))
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [row for row in rows if lower <= float(row["confidence"]) <= upper
                    and (index == bins - 1 or float(row["confidence"]) < upper)]
        if selected:
            accuracy = sum(bool(row["correct"]) for row in selected) / len(selected)
            confidence = sum(float(row["confidence"]) for row in selected) / len(selected)
            result += len(selected) / total * abs(accuracy - confidence)
    return result


def is_five_nine_error(row):
    truth, prediction = str(row["gt_jersey"]), str(row["prediction"])
    return len(truth) == len(prediction) and sum(
        left != right and {left, right} == {"5", "9"}
        for left, right in zip(truth, prediction)
    ) == 1 and sum(left != right for left, right in zip(truth, prediction)) == 1


def entropy(probabilities):
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities)


def truthy(value):
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def ratio(left, right):
    return left / right if right else 0.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
