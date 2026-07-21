#!/usr/bin/env python3
"""Create a deterministic, label-blind SoccerNet-GSR benchmark manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsr-dir", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unlock the frozen test split after model selection.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "The GSR test split is frozen. Pass --allow-test only for the final locked evaluation."
        )

    root = Path(args.gsr_dir).resolve()
    split_root = root / args.split
    available = sorted(
        path.name
        for path in split_root.iterdir()
        if path.is_dir() and (path / "img1").is_dir()
    )
    if args.sequences:
        missing = sorted(set(args.sequences) - set(available))
        if missing:
            raise FileNotFoundError(f"Sequences not found in {split_root}: {missing}")
        selected = list(dict.fromkeys(args.sequences))
        selection_method = "explicit"
    else:
        if args.count <= 0 or args.count > len(available):
            raise ValueError(f"--count must be in [1, {len(available)}]")
        selected = sorted(random.Random(args.seed).sample(available, args.count))
        selection_method = "seeded_random_without_replacement"

    entries = []
    for sequence in selected:
        sequence_dir = split_root / sequence
        frames_dir = sequence_dir / "img1"
        labels = sequence_dir / "Labels-GameState.json"
        if not labels.is_file():
            raise FileNotFoundError(f"Missing labels: {labels}")
        frames = sorted(
            path for path in frames_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not frames:
            raise FileNotFoundError(f"No image frames: {frames_dir}")
        entries.append({
            "sequence": sequence,
            "split": args.split,
            "frames_dir": str(frames_dir),
            "labels": str(labels),
            "frame_count": len(frames),
            "labels_sha256": sha256_file(labels),
        })

    manifest = {
        "benchmark": "gsr_detection_tracking_v1",
        "dataset_root": str(root),
        "split": args.split,
        "selection": {
            "method": selection_method,
            "seed": args.seed if selection_method.startswith("seeded") else None,
            "population_size": len(available),
            "selected_count": len(entries),
            "labels_not_used_for_selection": True,
        },
        "protocol": {
            "operational_gt_access": False,
            "evaluation_gt_access": True,
            "primary_iou": 0.50,
            "detection_iou_thresholds": [0.50, 0.75],
            "tracking_surfaces": ["display_track_id", "raw_track_id"],
            "test_split_frozen": args.split == "test",
        },
        "sequences": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
