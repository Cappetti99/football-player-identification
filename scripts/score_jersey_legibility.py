#!/usr/bin/env python3
"""Score FT player crops with the SoccerNet ResNet34 legibility classifier."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class LegibilityClassifier34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(weights=None)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

    def forward(self, images):
        return torch.sigmoid(self.model_ft(images))


class CropDataset(Dataset):
    def __init__(self, rows, crop_root):
        self.rows = rows
        self.crop_root = Path(crop_root)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["crop_path"])
        if not path.is_absolute():
            path = self.crop_root / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index, str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--crop-root", default=".")
    parser.add_argument("--labels")
    parser.add_argument("--oracle-tracklets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = read_csv(Path(args.manifest))
    apply_oracle_tracklets(rows, args.oracle_tracklets)
    labels = load_labels(args.labels)
    for row in rows:
        row["manual_label"] = labels.get(row.get("crop_id"), "")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = LegibilityClassifier34()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    dataset = CropDataset(rows, args.crop_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    with torch.inference_mode():
        for images, indices, paths in loader:
            scores = model(images.to(device)).reshape(-1).cpu().tolist()
            for index, path, score in zip(indices.tolist(), paths, scores):
                rows[index]["resolved_crop_path"] = path
                rows[index]["legibility_score"] = float(score)
                rows[index]["predicted_readable"] = bool(score >= args.threshold)

    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "crops": len(rows),
        "threshold": args.threshold,
        "manual_evaluation": evaluate_manual_labels(rows, args.threshold),
        "ocr_ranking": evaluate_ocr_ranking(rows),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "scores.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def evaluate_manual_labels(rows, threshold):
    binary = [row for row in rows if row["manual_label"] in {"readable", "not_readable"}]
    if not binary:
        return {"samples": 0}
    tp = sum(row["manual_label"] == "readable" and row["legibility_score"] >= threshold for row in binary)
    fp = sum(row["manual_label"] == "not_readable" and row["legibility_score"] >= threshold for row in binary)
    fn = sum(row["manual_label"] == "readable" and row["legibility_score"] < threshold for row in binary)
    tn = sum(row["manual_label"] == "not_readable" and row["legibility_score"] < threshold for row in binary)
    positives = [row["legibility_score"] for row in binary if row["manual_label"] == "readable"]
    negatives = [row["legibility_score"] for row in binary if row["manual_label"] == "not_readable"]
    return {
        "samples": len(binary),
        "readable": len(positives),
        "not_readable": len(negatives),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": ratio(tp + tn, len(binary)),
        "roc_auc": pairwise_auc(positives, negatives),
        "mean_score_readable": mean(positives),
        "mean_score_not_readable": mean(negatives),
        "partial": summarize_partial(rows),
    }


def evaluate_ocr_ranking(rows):
    groups = defaultdict(list)
    for row in rows:
        if integer(row.get("gt_jersey")) is None:
            continue
        groups[str(row.get("gt_track_id"))].append(row)
    decisions = []
    for track, items in sorted(groups.items(), key=lambda item: int(item[0])):
        ranked = sorted(items, key=lambda row: row["legibility_score"], reverse=True)
        correct = lambda row: integer(row.get("winner")) == integer(row.get("gt_jersey"))
        decisions.append({
            "gt_track_id": track,
            "top1_correct": bool(ranked and correct(ranked[0])),
            "top3_contains_correct": any(correct(row) for row in ranked[:3]),
            "top5_contains_correct": any(correct(row) for row in ranked[:5]),
            "best_frame": integer(ranked[0].get("frame")) if ranked else None,
            "best_score": ranked[0]["legibility_score"] if ranked else None,
            "best_prediction": integer(ranked[0].get("winner")) if ranked else None,
        })
    total = len(decisions)
    return {
        "tracklets": total,
        "top1_correct": sum(row["top1_correct"] for row in decisions),
        "top1_accuracy": ratio(sum(row["top1_correct"] for row in decisions), total),
        "top3_contains_correct": sum(row["top3_contains_correct"] for row in decisions),
        "top3_recall": ratio(sum(row["top3_contains_correct"] for row in decisions), total),
        "top5_contains_correct": sum(row["top5_contains_correct"] for row in decisions),
        "top5_recall": ratio(sum(row["top5_contains_correct"] for row in decisions), total),
        "decisions": decisions,
    }


def summarize_partial(rows):
    scores = [row["legibility_score"] for row in rows if row["manual_label"] == "partial"]
    return {"samples": len(scores), "mean_score": mean(scores)}


def pairwise_auc(positives, negatives):
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives
    )
    return float(wins / (len(positives) * len(negatives)))


def load_labels(path):
    if not path:
        return {}
    return {row["crop_id"]: row["label"] for row in read_csv(Path(path))}


def apply_oracle_tracklets(rows, path):
    if not path:
        return
    tracklets = json.loads(Path(path).read_text())
    by_track = {str(row["gt_track_id"]): row for row in tracklets}
    for row in rows:
        oracle = by_track.get(str(row.get("gt_track_id")))
        if not oracle:
            continue
        row.setdefault("gt_jersey", oracle.get("gt_jersey"))
        row.setdefault("current_prediction", oracle.get("current_prediction"))
        row.setdefault("current_correct", oracle.get("current_correct"))


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def integer(value):
    try: return int(float(value))
    except (TypeError, ValueError): return None


def ratio(a, b): return float(a / b) if b else None
def mean(values): return float(sum(values) / len(values)) if values else None


if __name__ == "__main__":
    main()
