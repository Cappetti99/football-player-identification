#!/usr/bin/env python3
"""Evaluate numeric CTC at crop and probabilistic track level."""

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
    greedy_decode,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--part", choices=["validation"], default="validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from scripts.train_jersey_number_ctc import JerseyCTCDataset, collate

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") != "resnet18_bilstm_numeric_ctc_v1":
        raise ValueError("unexpected numeric CTC checkpoint architecture")
    root = Path(args.dataset).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("evaluation dataset observes frozen sequences")
    parameters = metadata["training_parameters"]
    namespace = argparse.Namespace(
        image_height=metadata["image_size"][0], image_width=metadata["image_size"][1]
    )
    dataset = JerseyCTCDataset(root / f"{args.part}.jsonl", namespace, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_numeric_crnn(pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    crop_rows = []
    tracks = defaultdict(lambda: {"scores": [], "truth": None, "frames": []})
    with torch.no_grad():
        for images, _, _, rows in loader:
            logits = model(images.to(args.device)).cpu()
            for index, row in enumerate(rows):
                crop_logits = logits[:, index, :]
                prediction, confidence = greedy_decode(crop_logits)
                scores = candidate_log_probabilities(crop_logits)
                key = (row["sequence"], row["gt_track_id"])
                tracks[key]["scores"].append(scores)
                tracks[key]["truth"] = int(row["text"])
                tracks[key]["frames"].append(row["frame"])
                crop_rows.append({
                    "sequence": key[0], "gt_track_id": key[1], "frame": row["frame"],
                    "gt_jersey": row["text"], "prediction": prediction,
                    "confidence": confidence, "correct": prediction == row["text"],
                    "image": row["image"],
                })
    track_rows = []
    for key, track in sorted(tracks.items()):
        result = aggregate_frames(track["scores"])
        ranking = list(result["scores"].items())[:5]
        track_rows.append({
            "sequence": key[0], "gt_track_id": key[1], "gt_jersey": track["truth"],
            "prediction": result["prediction"], "correct": result["prediction"] == track["truth"],
            "confidence": result["confidence"], "margin": result["margin"],
            "frames": len(track["frames"]), "top5": json.dumps(ranking),
            "gt_in_top5": str(track["truth"]) in dict(ranking),
        })
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "crop_predictions.csv", crop_rows)
    write_csv(output / "track_predictions.csv", track_rows)
    metrics = {
        "crops": len(crop_rows),
        "crop_correct": sum(row["correct"] for row in crop_rows),
        "crop_accuracy": ratio(sum(row["correct"] for row in crop_rows), len(crop_rows)),
        "tracklets": len(track_rows),
        "track_correct": sum(row["correct"] for row in track_rows),
        "track_accuracy": ratio(sum(row["correct"] for row in track_rows), len(track_rows)),
        "track_gt_in_top5": sum(row["gt_in_top5"] for row in track_rows),
        "track_gt_in_top5_rate": ratio(sum(row["gt_in_top5"] for row in track_rows), len(track_rows)),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "dataset": str(root), "training_parameters": parameters,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ratio(a, b):
    return a / b if b else 0.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
