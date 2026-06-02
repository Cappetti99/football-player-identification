#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROI_REGIONS = {
    "full_body": (0.00, 1.00, 0.00, 1.00),
    "upper_back": (0.18, 0.55, 0.20, 0.80),
    "center_back": (0.20, 0.50, 0.25, 0.75),
    "torso": (0.12, 0.62, 0.15, 0.85),
    "number_band": (0.24, 0.48, 0.18, 0.82),
}


def main():
    parser = argparse.ArgumentParser(description="Train a compact CNN jersey-number classifier on SoccerNet-GSR crops.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gt-jsonl", required=True, type=Path)
    parser.add_argument("--train-sequence", action="append", required=True)
    parser.add_argument("--eval-sequence", action="append", required=True)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--same-sequence-split", choices=["none", "even-odd", "first-second"], default="none")
    parser.add_argument("--restrict-to-eval-labels", action="store_true")
    parser.add_argument("--roi", choices=sorted(ROI_REGIONS), default="torso")
    parser.add_argument("--size", default="64x128", help="Network input size WxH")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-per-class", type=int, default=1600)
    parser.add_argument("--max-eval-per-class", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    width, height = parse_size(args.size)
    train_sequences = set(args.train_sequence)
    eval_sequences = set(args.eval_sequence)
    train_records = load_records(args.gt_jsonl, train_sequences)
    eval_records = load_records(args.gt_jsonl, eval_sequences)
    if train_sequences == eval_sequences and args.same_sequence_split != "none":
        train_records, eval_records = split_same_sequence_records(train_records, args.same_sequence_split)

    if args.restrict_to_eval_labels:
        eval_labels = {jersey_number(row) for row in eval_records if jersey_number(row) is not None}
        train_records = [row for row in train_records if jersey_number(row) in eval_labels]
        eval_records = [row for row in eval_records if jersey_number(row) in eval_labels]

    class_labels = sorted({jersey_number(row) for row in train_records if jersey_number(row) is not None})
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    if not class_labels:
        raise RuntimeError("No train labels found")

    train_dataset = JerseyDataset(
        args.dataset_root,
        train_records,
        label_to_index,
        roi=args.roi,
        width=width,
        height=height,
        split_override=args.train_split,
        max_per_class=args.max_train_per_class,
        augment=True,
    )
    eval_dataset = JerseyDataset(
        args.dataset_root,
        eval_records,
        label_to_index,
        roi=args.roi,
        width=width,
        height=height,
        split_override=args.eval_split,
        max_per_class=args.max_eval_per_class,
        augment=False,
        drop_unknown_labels=True,
    )
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise RuntimeError(f"Empty dataset: train={len(train_dataset)} eval={len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallJerseyCNN(num_classes=len(class_labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best = None
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, eval_loader, device, class_labels)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "eval_accuracy": metrics["accuracy"],
            "eval_top3_accuracy": metrics["top3_accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or metrics["accuracy"] > best["metrics"]["accuracy"]:
            best = {"epoch": epoch, "metrics": metrics}
            save_checkpoint(args.checkpoint, model, class_labels, args, best)

    final = {
        "best_epoch": best["epoch"],
        "best_metrics": best["metrics"],
        "history": history,
        "train_sequences": sorted(train_sequences),
        "eval_sequences": sorted(eval_sequences),
        "same_sequence_split": args.same_sequence_split,
        "restrict_to_eval_labels": args.restrict_to_eval_labels,
        "roi": args.roi,
        "size": {"width": width, "height": height},
        "class_labels": class_labels,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "train_class_counts": train_dataset.class_counts(),
        "eval_class_counts": eval_dataset.class_counts(),
        "missing_train_images": train_dataset.missing_images,
        "missing_eval_images": eval_dataset.missing_images,
        "checkpoint": str(args.checkpoint),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final["best_metrics"], indent=2), flush=True)


class SmallJerseyCNN:
    def __new__(cls, num_classes):
        import torch
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )


class JerseyDataset:
    def __init__(
        self,
        dataset_root,
        records,
        label_to_index,
        roi,
        width,
        height,
        split_override=None,
        max_per_class=None,
        augment=False,
        drop_unknown_labels=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.label_to_index = label_to_index
        self.roi = roi
        self.width = int(width)
        self.height = int(height)
        self.split_override = split_override
        self.augment = bool(augment)
        self.rows = []
        self.missing_images = 0
        counts = Counter()
        for row in records:
            label = jersey_number(row)
            if label not in label_to_index:
                if drop_unknown_labels:
                    continue
                raise ValueError(f"Label {label} is not in train label set")
            if max_per_class and counts[label] >= max_per_class:
                continue
            self.rows.append(row)
            counts[label] += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        import torch

        row = self.rows[index]
        image_path = resolve_frame_path(
            self.dataset_root,
            self.split_override or row.get("split"),
            row["sequence"],
            int(row["frame"]),
        )
        image = cv2.imread(str(image_path))
        if image is None:
            self.missing_images += 1
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            crop = image
        else:
            crop = crop_bbox(image, row.get("bbox_image") or row.get("bbox"))
            crop = crop_roi(crop, ROI_REGIONS[self.roi], jitter=0.04 if self.augment else 0.0)
            if crop is None:
                crop = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        tensor = preprocess_crop(cv2, crop, self.width, self.height, augment=self.augment)
        label = self.label_to_index[int(jersey_number(row))]
        return torch.from_numpy(tensor), torch.tensor(label, dtype=torch.long)

    def class_counts(self):
        counts = Counter()
        for row in self.rows:
            label = int(jersey_number(row))
            if label in self.label_to_index:
                counts[str(label)] += 1
        return counts.most_common()


def train_one_epoch(model, loader, optimizer, criterion, device):
    import torch

    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(labels.numel())
        total += int(labels.numel())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    return total_loss / max(1, total), correct / max(1, total)


def evaluate(model, loader, device, class_labels):
    import torch

    model.eval()
    all_labels = []
    all_predictions = []
    all_top3 = []
    all_top1_probs = []
    all_margins = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            top_values, top_indices = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(top_indices[:, 0].numpy().tolist())
            all_top3.extend(top_indices.numpy().tolist())
            all_top1_probs.extend(top_values[:, 0].numpy().tolist())
            if top_values.shape[1] > 1:
                all_margins.extend((top_values[:, 0] - top_values[:, 1]).numpy().tolist())
            else:
                all_margins.extend(top_values[:, 0].numpy().tolist())

    labels = np.asarray(all_labels, dtype=np.int32)
    predictions = np.asarray(all_predictions, dtype=np.int32)
    top1_probs = np.asarray(all_top1_probs, dtype=np.float32)
    margins = np.asarray(all_margins, dtype=np.float32)
    correct = predictions == labels
    top3_correct = np.asarray([label in top for label, top in zip(labels.tolist(), all_top3)], dtype=bool)
    confusion = Counter(
        (str(class_labels[int(gt)]), str(class_labels[int(pred)]))
        for gt, pred in zip(labels.tolist(), predictions.tolist())
    )
    wrong = Counter(
        (str(class_labels[int(gt)]), str(class_labels[int(pred)]))
        for gt, pred in zip(labels.tolist(), predictions.tolist())
        if gt != pred
    )
    return {
        "accuracy": float(correct.mean()) if len(labels) else 0.0,
        "top3_accuracy": float(top3_correct.mean()) if len(labels) else 0.0,
        "correct": int(correct.sum()),
        "total": int(len(labels)),
        "top_confusions": confusion.most_common(30),
        "top_wrong": wrong.most_common(30),
        "selective_accuracy": selective_accuracy(correct, margins, top1_probs),
    }


def selective_accuracy(correct, margins, top1_probs):
    rows = []
    for threshold in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        mask = margins >= threshold
        rows.append(
            {
                "margin_threshold": float(threshold),
                "coverage": float(mask.mean()) if len(mask) else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
                "selected": int(mask.sum()),
            }
        )
    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
        mask = top1_probs >= threshold
        rows.append(
            {
                "score_threshold": float(threshold),
                "coverage": float(mask.mean()) if len(mask) else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
                "selected": int(mask.sum()),
            }
        )
    return rows


def save_checkpoint(path, model, class_labels, args, best):
    import torch

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_labels": class_labels,
            "args": vars(args),
            "best": best,
        },
        path,
    )


def load_records(path, sequences):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sequence") not in sequences:
                continue
            if str(row.get("role") or "").lower() not in {"player", "goalkeeper"}:
                continue
            if jersey_number(row) is None:
                continue
            rows.append(row)
    return rows


def split_same_sequence_records(records, mode):
    ordered = sorted(records, key=lambda row: (str(row.get("sequence")), int(row.get("frame", 0)), str(row.get("track_id", ""))))
    if mode == "even-odd":
        return (
            [row for row in ordered if int(row.get("frame", 0)) % 2 == 0],
            [row for row in ordered if int(row.get("frame", 0)) % 2 == 1],
        )
    by_sequence = defaultdict(list)
    for row in ordered:
        by_sequence[str(row.get("sequence"))].append(row)
    train = []
    eval_rows = []
    for seq_rows in by_sequence.values():
        frames = sorted({int(row.get("frame", 0)) for row in seq_rows})
        if not frames:
            continue
        midpoint = frames[len(frames) // 2]
        train.extend(row for row in seq_rows if int(row.get("frame", 0)) <= midpoint)
        eval_rows.extend(row for row in seq_rows if int(row.get("frame", 0)) > midpoint)
    return train, eval_rows


def resolve_frame_path(dataset_root, split, sequence, frame):
    image_dir = resolve_sequence_image_dir(dataset_root, split, sequence)
    candidates = [
        image_dir / f"{frame:06d}.jpg",
        image_dir / f"{frame:06d}.png",
        image_dir / f"{frame:08d}.jpg",
        image_dir / f"frame_{frame:06d}.jpg",
        image_dir / f"{frame}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_sequence_image_dir(dataset_root, split, sequence):
    split = str(split)
    split_candidates = [split]
    aliases = {
        "val": ["valid", "validation"],
        "valid": ["val", "validation"],
        "validation": ["valid", "val"],
    }
    split_candidates.extend(item for item in aliases.get(split.lower(), []) if item not in split_candidates)
    for candidate_split in split_candidates:
        image_dir = Path(dataset_root) / candidate_split / sequence / "img1"
        if image_dir.exists():
            return image_dir
    return Path(dataset_root) / split / sequence / "img1"


def crop_bbox(image, bbox):
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def crop_roi(crop, region, jitter=0.0):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    top_r, bottom_r, left_r, right_r = region
    if jitter > 0:
        top_r += random.uniform(-jitter, jitter)
        bottom_r += random.uniform(-jitter, jitter)
        left_r += random.uniform(-jitter, jitter)
        right_r += random.uniform(-jitter, jitter)
    top_r, bottom_r = max(0.0, top_r), min(1.0, bottom_r)
    left_r, right_r = max(0.0, left_r), min(1.0, right_r)
    top = int(h * top_r)
    bottom = max(top + 1, int(h * bottom_r))
    left = int(w * left_r)
    right = max(left + 1, int(w * right_r))
    roi = crop[top:bottom, left:right]
    return roi if roi.size else None


def preprocess_crop(cv2, crop, width, height, augment=False):
    crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    if augment:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-20, 20)
        crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        if random.random() < 0.25:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    tensor = gray.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor[None, :, :]


def jersey_number(row):
    value = row.get("jersey_number") or row.get("jersey") or row.get("number")
    if value in (None, "", "None", "unknown"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 99 else None


def parse_size(value):
    width, height = str(value).lower().split("x", 1)
    return int(width), int(height)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


if __name__ == "__main__":
    main()
