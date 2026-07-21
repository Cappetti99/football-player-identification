#!/usr/bin/env python3
"""Train a numeric CRNN-CTC and select checkpoints by track-level accuracy."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    BLANK_INDEX,
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
    encode_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    validate_args(args)
    seed_everything(args.seed)

    import torch
    from torch.utils.data import DataLoader

    root = Path(args.dataset).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_dataset_manifest(manifest)
    train_dataset = JerseyCTCDataset(root / "train.jsonl", args, augment=True)
    validation_dataset = JerseyCTCDataset(root / "validation.jsonl", args, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate, generator=generator,
            worker_init_fn=seed_worker,
        ),
        "validation": DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
        ),
    }
    model = build_numeric_crnn(pretrained=not bool(args.initial_checkpoint)).to(args.device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu")
        if initial.get("metadata", {}).get("architecture") != "resnet18_bilstm_numeric_ctc_v1":
            raise ValueError("unexpected initial CTC checkpoint architecture")
        model.load_state_dict(initial["state_dict"])
    criterion = torch.nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_backbone_epochs
        for parameter in model.features.parameters():
            parameter.requires_grad = not frozen
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, args.device)
        validation_metrics = run_epoch(model, loaders["validation"], criterion, None, args.device)
        track_metrics = evaluate_tracks(model, loaders["validation"], args.device)
        record = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_tracks": track_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if track_metrics["accuracy"] > best_accuracy:
            best_accuracy = track_metrics["accuracy"]
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "metadata": {
                    "architecture": "resnet18_bilstm_numeric_ctc_v1",
                    "alphabet": "0123456789",
                    "blank_index": BLANK_INDEX,
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                    "image_size": [args.image_height, args.image_width],
                    "epoch": epoch,
                    "validation_track_accuracy": best_accuracy,
                    "dataset_manifest": manifest,
                    "dataset_manifest_sha256": sha256(manifest_path),
                    "training_parameters": vars(args),
                    "initial_checkpoint": (
                        str(Path(args.initial_checkpoint).resolve())
                        if args.initial_checkpoint else None
                    ),
                    "initial_checkpoint_sha256": (
                        sha256(args.initial_checkpoint) if args.initial_checkpoint else None
                    ),
                    "seed": args.seed,
                },
            }, output)
        if epoch - best_epoch >= args.early_stopping:
            break
    history_path = output.with_suffix(output.suffix + ".history.json")
    history_path.write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_validation_track_accuracy": best_accuracy,
        "epochs": history,
    }, indent=2) + "\n")
    print(f"checkpoint={output} validation_track_accuracy={best_accuracy:.6f}")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JerseyCTCDataset:
    def __init__(self, path, args, augment):
        from torchvision import transforms

        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        operations = [transforms.Resize((args.image_height, args.image_width))]
        if augment:
            operations.extend([
                transforms.ColorJitter(0.25, 0.25, 0.20, 0.05),
                transforms.RandomGrayscale(p=0.08),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            ])
        operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
            if row.get("crop_box") is not None:
                width, height = image.size
                xmin, ymin, xmax, ymax = row["crop_box"]
                image = image.crop((
                    round(xmin * width), round(ymin * height),
                    round(xmax * width), round(ymax * height),
                ))
            tensor = self.transform(image)
        return tensor, encode_text(row["text"]), row


def collate(batch):
    import torch

    images, targets, rows = zip(*batch)
    lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    return torch.stack(images), torch.tensor([value for target in targets for value in target]), lengths, rows


def run_epoch(model, loader, criterion, optimizer, device):
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0
    for images, targets, target_lengths, _ in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            input_lengths = torch.full(
                (images.shape[0],), logits.shape[0], dtype=torch.long
            )
            loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss) * images.shape[0]
        samples += images.shape[0]
    return {"loss": total_loss / max(1, samples), "images": samples}


def evaluate_tracks(model, loader, device):
    import torch

    model.eval()
    tracks = defaultdict(lambda: {"scores": [], "truth": None})
    with torch.no_grad():
        for images, _, _, rows in loader:
            logits = model(images.to(device)).cpu()
            for index, row in enumerate(rows):
                key = (row["sequence"], row["gt_track_id"])
                tracks[key]["scores"].append(candidate_log_probabilities(logits[:, index, :]))
                tracks[key]["truth"] = int(row["text"])
    correct = 0
    for track in tracks.values():
        result = aggregate_frames(track["scores"])
        correct += result["prediction"] == track["truth"]
    return {"tracklets": len(tracks), "correct": correct, "accuracy": correct / max(1, len(tracks))}


def validate_args(args):
    numeric = [args.epochs, args.early_stopping, args.batch_size, args.image_height, args.image_width]
    if min(numeric) <= 0 or args.freeze_backbone_epochs < 0 or args.num_workers < 0:
        raise ValueError("invalid numeric training argument")


def validate_dataset_manifest(manifest):
    dataset_format = manifest.get("format")
    if dataset_format == "jersey_numeric_ctc_v1":
        if manifest.get("frozen_sequences_observed"):
            raise ValueError("CTC dataset observes frozen sequences")
        return dataset_format
    if dataset_format == "jersey_numeric_ctc_sjn210k_v1":
        if manifest.get("official_split_preserved") is not True:
            raise ValueError("SJN CTC dataset does not preserve the official split")
        if manifest.get("test_used_for_gradient_updates") is not False:
            raise ValueError("SJN CTC dataset permits test gradient updates")
        return dataset_format
    raise ValueError("unexpected CTC dataset format")


def seed_everything(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    import torch

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
