#!/usr/bin/env python3
"""Train a supervised attention-based multi-frame jersey digit recognizer."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from ft.features.jersey_multiframe_digits import (
    ARCHITECTURE,
    build_multiframe_digit_recognizer,
    load_ctc_encoder,
    number_log_probabilities,
    number_targets,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-ctc-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping", type=int, default=6)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-track", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frame-aux-weight", type=float, default=0.20)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    validate_args(args)
    seed_everything(args.seed)

    import torch
    from torch.utils.data import DataLoader

    root = Path(args.dataset).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    datasets = {
        "train": JerseyMultiFrameDataset(root / "train.jsonl", args, True),
        "validation": JerseyMultiFrameDataset(root / "validation.jsonl", args, False),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            collate_fn=collate_bags,
            generator=generator if name == "train" else None,
            worker_init_fn=seed_worker if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }

    model = build_multiframe_digit_recognizer(pretrained=False)
    initial_path = Path(args.initial_ctc_checkpoint).resolve()
    initial = torch.load(initial_path, map_location="cpu")
    loaded_encoder_tensors = load_ctc_encoder(model, initial)
    model.to(args.device)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.backbone_learning_rate},
        {"params": model.attention.parameters(), "lr": args.learning_rate},
        {"params": model.length_head.parameters(), "lr": args.learning_rate},
        {"params": model.tens_head.parameters(), "lr": args.learning_rate},
        {"params": model.units_head.parameters(), "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    history, best_accuracy, best_epoch = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_backbone_epochs
        for parameter in model.features.parameters():
            parameter.requires_grad = not frozen
        train = run_epoch(model, loaders["train"], optimizer, args)
        validation = run_epoch(model, loaders["validation"], None, args)
        record = {"epoch": epoch, "backbone_frozen": frozen, "train": train, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["number_accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["number_accuracy"], epoch
            torch.save({
                "state_dict": model.state_dict(),
                "metadata": {
                    "architecture": ARCHITECTURE,
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                    "image_size": [args.image_height, args.image_width],
                    "frames_per_track": args.frames_per_track,
                    "epoch": epoch,
                    "validation_number_accuracy": best_accuracy,
                    "dataset_manifest": manifest,
                    "dataset_manifest_sha256": sha256(manifest_path),
                    "training_parameters": vars(args),
                    "initial_ctc_checkpoint": str(initial_path),
                    "initial_ctc_checkpoint_sha256": sha256(initial_path),
                    "loaded_encoder_tensors": loaded_encoder_tensors,
                    "seed": args.seed,
                },
            }, output)
        if epoch - best_epoch >= args.early_stopping:
            break
    output.with_suffix(output.suffix + ".history.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_validation_number_accuracy": best_accuracy,
        "epochs": history,
    }, indent=2) + "\n")
    print(f"checkpoint={output} validation_number_accuracy={best_accuracy:.6f}")


class JerseyMultiFrameDataset:
    def __init__(self, path, args, augment):
        from torchvision import transforms

        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.frames_per_track = args.frames_per_track
        self.augment = augment
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
        frames = sample_frames(row["frames"], self.frames_per_track, self.augment)
        tensors = []
        for frame in frames:
            with Image.open(frame["image"]) as image:
                image = image.convert("RGB")
                if frame.get("crop_box") is not None:
                    width, height = image.size
                    xmin, ymin, xmax, ymax = frame["crop_box"]
                    image = image.crop((round(xmin*width), round(ymin*height), round(xmax*width), round(ymax*height)))
                tensors.append(self.transform(image))
        return tensors, int(row["jersey"]), row


def sample_frames(frames, maximum, random_sample):
    frames = sorted(frames, key=lambda row: int(row["frame"]))
    if len(frames) <= maximum:
        return frames
    if random_sample:
        return sorted(random.sample(frames, maximum), key=lambda row: int(row["frame"]))
    if maximum == 1:
        return [frames[len(frames)//2]]
    return [frames[round(index*(len(frames)-1)/(maximum-1))] for index in range(maximum)]


def collate_bags(batch):
    import torch

    tensors, jerseys, rows = zip(*batch)
    maximum = max(len(values) for values in tensors)
    shape = tensors[0][0].shape
    images = torch.zeros((len(batch), maximum, *shape), dtype=tensors[0][0].dtype)
    mask = torch.zeros((len(batch), maximum), dtype=torch.bool)
    for bag_index, values in enumerate(tensors):
        for frame_index, tensor in enumerate(values):
            images[bag_index, frame_index] = tensor
            mask[bag_index, frame_index] = True
    return images, mask, torch.tensor(jerseys, dtype=torch.long), rows


def recognition_loss(outputs, jerseys, mask, frame_aux_weight):
    import torch
    import torch.nn.functional as functional

    lengths, tens, units = number_targets(jerseys, jerseys.device)
    primary = (
        functional.cross_entropy(outputs["length_logits"], lengths)
        + functional.cross_entropy(outputs["units_logits"], units)
        + optional_tens_loss(outputs["tens_logits"], tens)
    )
    valid = mask.reshape(-1)
    repeated_lengths = lengths[:, None].expand_as(mask).reshape(-1)[valid]
    repeated_tens = tens[:, None].expand_as(mask).reshape(-1)[valid]
    repeated_units = units[:, None].expand_as(mask).reshape(-1)[valid]
    auxiliary = (
        functional.cross_entropy(outputs["frame_length_logits"].reshape(-1, 2)[valid], repeated_lengths)
        + functional.cross_entropy(outputs["frame_units_logits"].reshape(-1, 10)[valid], repeated_units)
        + optional_tens_loss(outputs["frame_tens_logits"].reshape(-1, 10)[valid], repeated_tens)
    )
    return primary + frame_aux_weight * auxiliary, primary, auxiliary


def optional_tens_loss(logits, targets):
    import torch.nn.functional as functional

    valid = targets != -100
    if not valid.any():
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[valid], targets[valid])


def run_epoch(model, loader, optimizer, args):
    import torch

    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "primary_loss": 0.0, "auxiliary_loss": 0.0, "tracklets": 0,
              "number_correct": 0, "length_correct": 0, "tens_correct": 0, "tens_total": 0,
              "units_correct": 0, "five_nine_confusions": 0}
    for images, mask, jerseys, _ in loader:
        images, mask, jerseys = images.to(args.device), mask.to(args.device), jerseys.to(args.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images, mask)
            loss, primary, auxiliary = recognition_loss(outputs, jerseys, mask, args.frame_aux_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch = jerseys.shape[0]
        predictions = number_log_probabilities(outputs).argmax(dim=1)
        lengths, tens, units = number_targets(jerseys, jerseys.device)
        predicted_lengths = outputs["length_logits"].argmax(dim=1)
        predicted_tens = outputs["tens_logits"].argmax(dim=1)
        predicted_units = outputs["units_logits"].argmax(dim=1)
        two = lengths == 1
        totals["loss"] += float(loss) * batch
        totals["primary_loss"] += float(primary) * batch
        totals["auxiliary_loss"] += float(auxiliary) * batch
        totals["tracklets"] += batch
        totals["number_correct"] += int((predictions == jerseys).sum())
        totals["length_correct"] += int((predicted_lengths == lengths).sum())
        totals["units_correct"] += int((predicted_units == units).sum())
        totals["tens_correct"] += int(((predicted_tens == tens) & two).sum())
        totals["tens_total"] += int(two.sum())
        tens_confusions = (
            two
            & (((tens == 5) & (predicted_tens == 9)) | ((tens == 9) & (predicted_tens == 5)))
        )
        units_confusions = (
            ((units == 5) & (predicted_units == 9))
            | ((units == 9) & (predicted_units == 5))
        )
        totals["five_nine_confusions"] += int(tens_confusions.sum() + units_confusions.sum())
    count = max(1, totals["tracklets"])
    return {
        **{key: totals[key] / count for key in ("loss", "primary_loss", "auxiliary_loss")},
        "tracklets": totals["tracklets"],
        "number_accuracy": totals["number_correct"] / count,
        "length_accuracy": totals["length_correct"] / count,
        "tens_accuracy": totals["tens_correct"] / max(1, totals["tens_total"]),
        "units_accuracy": totals["units_correct"] / count,
        "five_nine_confusions": totals["five_nine_confusions"],
    }


def validate_manifest(manifest):
    if manifest.get("format") != "jersey_multiframe_digits_v1":
        raise ValueError("unexpected multi-frame dataset format")
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("multi-frame dataset observes frozen sequences")
    if set(manifest.get("train_sequences", [])) & set(manifest.get("validation_sequences", [])):
        raise ValueError("train and validation sequences overlap")


def validate_args(args):
    positive = (args.epochs, args.early_stopping, args.batch_size, args.frames_per_track,
                args.image_height, args.image_width, args.learning_rate, args.backbone_learning_rate)
    if min(positive) <= 0 or args.freeze_backbone_epochs < 0 or args.num_workers < 0:
        raise ValueError("invalid numeric training argument")
    if args.frame_aux_weight < 0:
        raise ValueError("--frame-aux-weight must be non-negative")


def seed_everything(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    import torch
    random.seed(torch.initial_seed() % (2**32))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
