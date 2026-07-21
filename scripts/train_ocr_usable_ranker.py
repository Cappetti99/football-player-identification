#!/usr/bin/env python3
"""Fine-tune the legibility ResNet34 for crop-level OCR usability."""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


class OCRUsableResNet34(nn.Module):
    """Architecture-compatible with the operational legibility selector."""

    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet34(weights=None)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

    def forward(self, images):
        return torch.sigmoid(self.model_ft(images))


class CropDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        augment = [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ] if train else [transforms.Resize((224, 224))]
        self.transform = transforms.Compose(augment + [
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
        with Image.open(row["crop_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(float(row["ocr_usable"])), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--validation-splits", nargs="+", default=["validation"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    rows = normalize_rows(read_csv(Path(args.dataset)))
    train_splits = {normalize_split(value) for value in args.train_splits}
    validation_splits = {normalize_split(value) for value in args.validation_splits}
    forbidden = train_splits & validation_splits
    if forbidden:
        raise ValueError(f"train/validation splits overlap: {sorted(forbidden)}")
    if "test" in train_splits or "test" in validation_splits:
        raise ValueError("test split is frozen and cannot be used for training or calibration")

    train_rows = [row for row in rows if row["split"] in train_splits]
    validation_rows = [row for row in rows if row["split"] in validation_splits]
    validate_partitions(train_rows, validation_rows)
    device = torch.device(args.device)

    model = OCRUsableResNet34()
    state = torch.load(args.initial_checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    train_dataset = CropDataset(train_rows, train=True)
    validation_dataset = CropDataset(validation_rows, train=False)
    sampler = balanced_sampler(train_rows)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.BCELoss()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    initial_validation = evaluate(model, validation_loader, loss_fn, device)
    initial_threshold = choose_zero_fp_threshold(
        initial_validation["scores"], initial_validation["labels"]
    )
    initial_row = {
        "epoch": 0,
        "train_loss": None,
        "validation_loss": initial_validation["loss"],
        "roc_auc": pairwise_auc(initial_validation["scores"], initial_validation["labels"]),
        "zero_fp_threshold": initial_threshold,
        **threshold_metrics(
            initial_validation["scores"], initial_validation["labels"], initial_threshold
        ),
    }
    initial_objective = (
        initial_row["true_positives"], initial_row["roc_auc"] or 0.0,
        -initial_validation["loss"],
    )
    best = {"objective": initial_objective, "epoch": 0, "metrics": initial_row}
    history = [initial_row]
    torch.save(model.state_dict(), output / "best_ocr_usable_resnet34.pth")
    print(json.dumps(initial_row), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        validation = evaluate(model, validation_loader, loss_fn, device)
        threshold = choose_zero_fp_threshold(validation["scores"], validation["labels"])
        metrics = threshold_metrics(validation["scores"], validation["labels"], threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "roc_auc": pairwise_auc(validation["scores"], validation["labels"]),
            "zero_fp_threshold": threshold,
            **metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        objective = (metrics["true_positives"], row["roc_auc"] or 0.0, -validation["loss"])
        if best is None or objective > best["objective"]:
            best = {"objective": objective, "epoch": epoch, "metrics": row}
            torch.save(model.state_dict(), output / "best_ocr_usable_resnet34.pth")
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    metadata = {
        "architecture": "ResNet34 -> 1 sigmoid",
        "target": "P(any OCR candidate exactly matches visible GT jersey)",
        "initial_checkpoint": str(args.initial_checkpoint),
        "train_splits": sorted(train_splits),
        "validation_splits": sorted(validation_splits),
        "train_sequences": sorted({row["sequence"] for row in train_rows}),
        "validation_sequences": sorted({row["sequence"] for row in validation_rows}),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "seed": args.seed,
        "best": best,
        "history": history,
        "checkpoint": str(output / "best_ocr_usable_resnet34.pth"),
    }
    (output / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    samples = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(images).reshape(-1)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        samples += len(labels)
    return total_loss / samples if samples else None


def evaluate(model, loader, loss_fn, device):
    model.eval()
    scores, labels = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, batch_labels, _ in loader:
            images, batch_labels = images.to(device), batch_labels.to(device)
            batch_scores = model(images).reshape(-1)
            total_loss += float(loss_fn(batch_scores, batch_labels).item()) * len(batch_labels)
            scores.extend(batch_scores.cpu().tolist())
            labels.extend(int(value) for value in batch_labels.cpu().tolist())
    return {"scores": scores, "labels": labels, "loss": total_loss / len(labels) if labels else None}


def choose_zero_fp_threshold(scores, labels):
    negatives = [score for score, label in zip(scores, labels) if not label]
    return max(negatives) + 1e-7 if negatives else 0.5


def threshold_metrics(scores, labels, threshold):
    predictions = [score >= threshold for score in scores]
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    fp = sum(pred and not label for pred, label in zip(predictions, labels))
    positives = sum(labels)
    recall = tp / positives if positives else None
    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "recall": recall,
        "zero_fp_recall": recall if fp == 0 else None,
        "emitted": sum(predictions),
    }


def pairwise_auc(scores, labels):
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def balanced_sampler(rows):
    positives = sum(row["ocr_usable"] for row in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError("training requires both positive and negative OCR-usability labels")
    weights = [1.0 / (positives if row["ocr_usable"] else negatives) for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


def validate_partitions(train_rows, validation_rows):
    if not train_rows or not validation_rows:
        raise ValueError("training and validation partitions must both be non-empty")
    overlap = {row["sequence"] for row in train_rows} & {row["sequence"] for row in validation_rows}
    if overlap:
        raise ValueError(f"sequence leakage between training and validation: {sorted(overlap)}")
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        missing = [row["crop_path"] for row in rows if not Path(row["crop_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{name} crop missing, first example: {missing[0]}")


def normalize_rows(rows):
    output = []
    for row in rows:
        output.append({
            **row,
            "split": normalize_split(row.get("split")),
            "sequence": str(row.get("sequence") or ""),
            "crop_path": str(row.get("crop_path") or ""),
            "ocr_usable": str(row.get("ocr_usable")).lower() in {"1", "true", "yes"},
        })
    return output


def normalize_split(value):
    value = str(value or "").strip().lower()
    return {"val": "validation", "valid": "validation"}.get(value, value)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
