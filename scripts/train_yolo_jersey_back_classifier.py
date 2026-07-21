#!/usr/bin/env python3
"""Train an Ultralytics YOLO crop classifier for clean player backs."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--project", default="runs/jersey_back_classifier")
    parser.add_argument("--name", default="yolo_nano_v1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    data = Path(args.data).resolve()
    metadata_path = data / "manifest.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"dataset manifest missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("classes") != ["clean_back", "not_clean"]:
        raise ValueError("unexpected jersey-back dataset classes")
    if set(metadata.get("validation_sequences") or []) & set(metadata.get("frozen_validation_sequences") or []):
        raise ValueError("dataset validation split overlaps frozen GSR validation")
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
        data=str(data), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        device=args.device, seed=args.seed, deterministic=True,
        project=str(project), name=args.name, exist_ok=False,
    )
    save_dir = Path(results.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    if not best.is_file(): raise RuntimeError(f"Ultralytics best checkpoint missing: {best}")
    run_metadata = {
        "task": "classification",
        "target": "clean_back_vs_not_clean",
        "ultralytics_version": ultralytics.__version__,
        "base_model": args.model,
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256(best),
        "dataset": str(data),
        "dataset_manifest_sha256": sha256(metadata_path),
        "seed": args.seed,
        "training_parameters": vars(args),
        "resolved_project": str(project),
    }
    (save_dir / "ft_run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_metadata, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__": main()
