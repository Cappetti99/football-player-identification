#!/usr/bin/env python3
"""Evaluate an OCR-usability checkpoint at a frozen threshold."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.train_ocr_usable_ranker import (
    CropDataset,
    OCRUsableResNet34,
    pairwise_auc,
    threshold_metrics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    rows = normalize_rows(read_csv(Path(args.dataset)))
    allowed_splits = {normalize_split(value) for value in args.splits}
    rows = [row for row in rows if row["split"] in allowed_splits]
    if not rows:
        raise ValueError(f"no rows for requested splits: {sorted(allowed_splits)}")
    missing = [row["crop_path"] for row in rows if not Path(row["crop_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"crop missing, first example: {missing[0]}")

    device = torch.device(args.device)
    model = OCRUsableResNet34()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    loader = DataLoader(
        CropDataset(rows, train=False), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    scores = [None] * len(rows)
    with torch.inference_mode():
        for images, _, indices in loader:
            values = model(images.to(device)).reshape(-1).cpu().tolist()
            for index, value in zip(indices.tolist(), values):
                scores[index] = float(value)

    scored_rows = [
        {**row, "ocr_usable_score": score, "emitted": score >= args.threshold}
        for row, score in zip(rows, scores)
    ]
    by_sequence = defaultdict(list)
    for row in scored_rows:
        by_sequence[row["sequence"]].append(row)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": args.threshold,
        "threshold_policy": "frozen_external_input",
        "splits": sorted(allowed_splits),
        "overall": summarize(scored_rows, args.threshold),
        "sequences": {
            sequence: summarize(items, args.threshold)
            for sequence, items in sorted(by_sequence.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output.with_suffix(".csv"), scored_rows)
    print(json.dumps(summary, indent=2))


def summarize(rows, threshold):
    scores = [row["ocr_usable_score"] for row in rows]
    labels = [int(row["ocr_usable"]) for row in rows]
    metrics = threshold_metrics(scores, labels, threshold)
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "rows": len(rows),
        "positive": positives,
        "negative": negatives,
        "roc_auc": pairwise_auc(scores, labels),
        **metrics,
        "precision": (
            metrics["true_positives"] / metrics["emitted"]
            if metrics["emitted"] else None
        ),
    }


def normalize_rows(rows):
    return [{
        **row,
        "sequence": str(row.get("sequence") or ""),
        "split": normalize_split(row.get("split")),
        "crop_path": str(row.get("crop_path") or ""),
        "ocr_usable": str(row.get("ocr_usable")).lower() in {"1", "true", "yes"},
    } for row in rows]


def normalize_split(value):
    value = str(value or "").strip().lower()
    return {"val": "validation", "valid": "validation"}.get(value, value)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
