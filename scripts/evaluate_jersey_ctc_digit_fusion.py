#!/usr/bin/env python3
"""Select digit-level CTC fusion on train and evaluate sequence-disjoint validation."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
    interpolate_track_scores,
)


DEFAULT_WEIGHTS = tuple(index / 10 for index in range(11))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--digit-weight", action="append", type=float, dest="digit_weights")
    parser.add_argument("--selection-part", choices=["train", "validation"], default="train")
    parser.add_argument("--max-selection-regressions", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    weights = tuple(args.digit_weights or DEFAULT_WEIGHTS)
    if not weights or any(not 0 <= value <= 1 for value in weights):
        raise ValueError("digit weights must be between 0 and 1")
    if args.max_selection_regressions < 0:
        raise ValueError("--max-selection-regressions must be non-negative")

    import torch
    from torch.utils.data import DataLoader
    from scripts.train_jersey_number_ctc import JerseyCTCDataset, collate

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") != "resnet18_bilstm_numeric_ctc_v1":
        raise ValueError("unexpected CTC checkpoint architecture")
    root = Path(args.dataset).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("dataset observes frozen sequences")
    model = build_numeric_crnn(pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    namespace = argparse.Namespace(image_height=metadata["image_size"][0], image_width=metadata["image_size"][1])
    tracks = {}
    for part in ("train", "validation"):
        dataset = JerseyCTCDataset(root / f"{part}.jsonl", namespace, augment=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        tracks[part] = infer_tracks(model, loader, args.device)

    selection_tracks = tracks[args.selection_part]
    selection_baseline = prediction_rows(selection_tracks, 0.0)
    sweep = [
        evaluate_weight(selection_tracks, value, selection_baseline)
        for value in sorted(set(weights))
    ]
    selected = select_weight(sweep, args.max_selection_regressions)
    validation_rows = prediction_rows(tracks["validation"], selected["digit_weight"])
    baseline_rows = prediction_rows(tracks["validation"], 0.0)
    comparison = compare_rows(baseline_rows, validation_rows)
    metrics = {
        "selected_digit_weight": selected["digit_weight"],
        "selection_part": args.selection_part,
        "max_selection_regressions": args.max_selection_regressions,
        f"weight_sweep_{args.selection_part}": sweep,
        "validation_baseline": summarize(baseline_rows),
        "validation_fusion": summarize(validation_rows),
        "paired_comparison": comparison,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dataset": str(root),
        "dataset_manifest_sha256": sha256(manifest_path),
        "frozen_sequences_observed": [],
    }
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "baseline_track_predictions.csv", baseline_rows)
    write_csv(output / "fusion_track_predictions.csv", validation_rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def infer_tracks(model, loader, device):
    import torch

    tracks = defaultdict(lambda: {"scores": [], "truth": None, "frames": []})
    with torch.no_grad():
        for images, _, _, rows in loader:
            logits = model(images.to(device)).cpu()
            for index, row in enumerate(rows):
                key = (str(row["sequence"]), str(row["gt_track_id"]))
                tracks[key]["scores"].append(candidate_log_probabilities(logits[:, index, :]))
                tracks[key]["truth"] = int(row["text"])
                tracks[key]["frames"].append(int(row["frame"]))
    return dict(tracks)


def select_weight(sweep, max_regressions=0):
    eligible = [
        row for row in sweep
        if row["paired_to_zero"]["correct_to_wrong"] <= max_regressions
    ]
    if not eligible:
        raise RuntimeError("no digit weight satisfies the regression constraint")
    # Accuracy is primary; top-5 is secondary; smaller weight preserves CTC on ties.
    return max(eligible, key=lambda row: (row["accuracy"], row["top5_rate"], -row["digit_weight"]))


def evaluate_weight(tracks, weight, baseline_rows=None):
    rows = prediction_rows(tracks, weight)
    summary = summarize(rows)
    baseline_rows = baseline_rows or prediction_rows(tracks, 0.0)
    return {"digit_weight": weight, "correct": summary["correct"], "accuracy": summary["accuracy"],
            "top5": summary["top5"], "top5_rate": summary["top5_rate"],
            "paired_to_zero": compare_rows(baseline_rows, rows)}


def prediction_rows(tracks, weight):
    rows = []
    for key, track in sorted(tracks.items()):
        result = interpolate_track_scores(track["scores"], weight)
        top5 = list(result["scores"].items())[:5]
        rows.append({
            "sequence": key[0], "gt_track_id": key[1], "gt_jersey": track["truth"],
            "prediction": result["prediction"], "correct": result["prediction"] == track["truth"],
            "confidence": result["confidence"], "margin": result["margin"],
            "frames": json.dumps(track["frames"]), "digit_weight": weight,
            "top5": json.dumps(top5), "gt_in_top5": str(track["truth"]) in dict(top5),
        })
    return rows


def summarize(rows):
    correct = sum(bool(row["correct"]) for row in rows)
    top5 = sum(bool(row["gt_in_top5"]) for row in rows)
    return {"tracklets": len(rows), "correct": correct, "accuracy": ratio(correct, len(rows)),
            "top5": top5, "top5_rate": ratio(top5, len(rows))}


def compare_rows(baseline, candidate):
    left = {(row["sequence"], row["gt_track_id"]): row for row in baseline}
    right = {(row["sequence"], row["gt_track_id"]): row for row in candidate}
    if set(left) != set(right):
        raise ValueError("paired rows do not match")
    transitions = {name: 0 for name in (
        "correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"
    )}
    for key in left:
        source = "correct" if left[key]["correct"] else "wrong"
        target = "correct" if right[key]["correct"] else "wrong"
        transitions[f"{source}_to_{target}"] += 1
    transitions["net_correct"] = transitions["wrong_to_correct"] - transitions["correct_to_wrong"]
    return transitions


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


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
