#!/usr/bin/env python3
"""Split number-region hard-misses into pose-driven vs detector-driven failures.

The contact-sheet qualitative review of the 12 blind tracklets (frozen GSR
surface) showed a mix: many crops are simply front/side-facing (no number in
frame, a frame-selection problem), but a few are clearly back-facing and
legible yet still missed by the region detector (a real detector gap), plus
some are back-facing with a low-contrast number on the shirt.

Rather than label this by hand, reuse the already-trained, never-promoted
`jersey_back_yolo11s_cls_v1.pt` clean-back/not-clean classifier purely as a
diagnostic here (no runtime integration, no training, no new labels) to bucket
every hard-miss crop by predicted pose. This tells us how much of the 25%
hard-miss rate is explained by pose (not attackable by improving the region
detector) versus how much remains unexplained by pose (the real detector
recall gap worth addressing).

Input: `region_detection_coverage.csv` from
`audit_jersey_number_region_detector_coverage.py`. Output: per-crop pose
prediction merged in, plus a summary contrasting hard-miss vs detected crops
and, for hard-miss crops only, clean_back vs not_clean split.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="region_detection_coverage.csv path")
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--classifier-checkpoint-sha256", default=None)
    parser.add_argument("--clean-back-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.classifier_checkpoint_sha256:
        digest = sha256_file(args.classifier_checkpoint)
        if digest.lower() != args.classifier_checkpoint_sha256.lower():
            raise ValueError(f"classifier checkpoint SHA-256 mismatch: {digest}")

    rows = read_csv(args.csv)
    for row in rows:
        row["is_hard_miss"] = row.get("detector_confidence") in (None, "")

    from ultralytics import YOLO

    classifier = YOLO(args.classifier_checkpoint)
    clean_back_index = resolve_clean_back_index(classifier.names)

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        results = classifier.predict(
            [row["crop_path"] for row in batch],
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            row["clean_back_probability"] = float(result.probs.data[clean_back_index])
            row["predicted_clean_back"] = row["clean_back_probability"] >= args.clean_back_threshold

    hard_miss = [row for row in rows if row["is_hard_miss"]]
    detected = [row for row in rows if not row["is_hard_miss"]]
    hard_miss_clean_back = [row for row in hard_miss if row["predicted_clean_back"]]
    hard_miss_not_clean = [row for row in hard_miss if not row["predicted_clean_back"]]

    per_track = tracklet_pose_breakdown(hard_miss)

    summary = {
        "classifier_checkpoint": str(Path(args.classifier_checkpoint).resolve()),
        "clean_back_threshold": args.clean_back_threshold,
        "total_crops": len(rows),
        "hard_miss_crops": len(hard_miss),
        "detected_crops": len(detected),
        "mean_clean_back_probability": {
            "hard_miss": mean([row["clean_back_probability"] for row in hard_miss]),
            "detected": mean([row["clean_back_probability"] for row in detected]),
        },
        "hard_miss_split": {
            "pose_explained_not_clean_back": len(hard_miss_not_clean),
            "pose_explained_share": ratio(len(hard_miss_not_clean), len(hard_miss)),
            "unexplained_clean_back_still_missed": len(hard_miss_clean_back),
            "unexplained_share": ratio(len(hard_miss_clean_back), len(hard_miss)),
        },
        "tracklets_all_hard_miss_not_clean_back": [
            key for key, stats in per_track.items() if stats["clean_back"] == 0
        ],
        "tracklets_with_unexplained_clean_back_miss": [
            key for key, stats in per_track.items() if stats["clean_back"] > 0
        ],
        "interpretation": (
            "pose_explained_not_clean_back: hard-miss crops the pose classifier itself "
            "does not consider a clean back view -> frame-selection/pose limit, not a "
            "detector defect. unexplained_clean_back_still_missed: hard-miss crops the "
            "pose classifier considers a clean back view -> genuine region-detector "
            "recall gap worth addressing with more training data/augmentation."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "hard_miss_pose_breakdown.csv", rows)
    (output / "hard_miss_pose_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\ncsv={output / 'hard_miss_pose_breakdown.csv'}")
    print(f"summary={output / 'hard_miss_pose_summary.json'}")


def resolve_clean_back_index(names):
    for index, name in names.items():
        if str(name).strip().lower() == "clean_back":
            return index
    raise ValueError(f"'clean_back' class not found in classifier names: {names}")


def tracklet_pose_breakdown(hard_miss_rows):
    grouped = defaultdict(list)
    for row in hard_miss_rows:
        grouped[f"{row.get('sequence', '')}::{row.get('gt_track_id', '')}"].append(row)
    return {
        key: {
            "hard_miss": len(items),
            "clean_back": sum(1 for item in items if item["predicted_clean_back"]),
        }
        for key, items in grouped.items()
    }


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


def ratio(a, b):
    return a / b if b else 0.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
