#!/usr/bin/env python3
"""Train a YOLO small detector for the jersey number region."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--manifest",
        help="Dataset provenance manifest; defaults to DATA/manifest.json",
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--project", default="runs/jersey_number_region")
    parser.add_argument("--name", default="yolo26s_v1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    data = Path(args.data).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else data / "manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    dataset_format = validate_dataset_manifest(manifest, data)
    data_yaml = data / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"YOLO data file not found: {data_yaml}")
    from ultralytics import YOLO
    import ultralytics

    result = YOLO(args.model).train(
        data=str(data_yaml), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
        seed=args.seed, deterministic=True, project=str(Path(args.project).resolve()),
        name=args.name, exist_ok=False,
    )
    save_dir = Path(result.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("best detector checkpoint missing")
    metadata = {
        "task": "detection", "target": "jersey_number_region",
        "dataset_format": dataset_format,
        "dataset": str(data),
        "dataset_manifest": str(manifest_path),
        "ultralytics_version": ultralytics.__version__, "base_model": args.model,
        "best_checkpoint": str(best), "best_checkpoint_sha256": sha256(best),
        "dataset_manifest_sha256": sha256(manifest_path), "training_parameters": vars(args),
    }
    (save_dir / "ft_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def validate_dataset_manifest(manifest, data):
    """Validate either the original GSR manifest or the SJN-210k derivation."""
    if manifest.get("target") == "jersey_number_region":
        if manifest.get("frozen_sequences_observed"):
            raise ValueError("invalid or leaked number-region dataset")
        return manifest.get("format", "gsr_jersey_number_region")

    if manifest.get("format") != "sjn210k_ft_v1":
        raise ValueError("invalid number-region dataset manifest")

    if manifest.get("official_split_preserved") is not True:
        raise ValueError("SJN training split is not the official train split")
    if manifest.get("test_used_for_gradient_updates") is not False:
        raise ValueError("SJN manifest permits gradients from the official test split")

    declared_yaml = manifest.get("number_region_yolo")
    if not declared_yaml:
        raise ValueError("SJN manifest does not declare number_region_yolo")
    if Path(declared_yaml).resolve() != (Path(data).resolve() / "data.yaml"):
        raise ValueError("SJN manifest and --data refer to different YOLO datasets")
    return "sjn210k_ft_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
