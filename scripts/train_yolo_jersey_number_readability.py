#!/usr/bin/env python3
"""Train YOLO classification for manual jersey-number readability."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11s-cls.pt")
    parser.add_argument("--project", default="runs/jersey_number_readability")
    parser.add_argument("--name", default="yolo11s_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    data = Path(args.data).resolve()
    manifest_path = data / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target") != "jersey_number_readability":
        raise ValueError("dataset target is not jersey_number_readability")
    if manifest.get("classes") != ["number_readable", "number_unreadable"]:
        raise ValueError("unexpected readability dataset classes")
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("readability dataset observes frozen sequences")
    if min(args.epochs, args.patience, args.imgsz, args.batch, args.workers + 1) <= 0:
        raise ValueError("training numeric arguments must be positive")

    try:
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required in the training environment") from exc

    project = Path(args.project).resolve()
    model = YOLO(args.model)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        project=str(project),
        name=args.name,
        exist_ok=False,
    )
    save_dir = Path(results.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Ultralytics best checkpoint missing: {best}")
    metadata = {
        "task": "classification",
        "target": "jersey_number_readability",
        "ultralytics_version": ultralytics.__version__,
        "base_model": args.model,
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256(best),
        "dataset": str(data),
        "dataset_manifest_sha256": sha256(manifest_path),
        "seed": args.seed,
        "training_parameters": vars(args),
    }
    (save_dir / "ft_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
