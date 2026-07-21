#!/usr/bin/env python3
"""Build a YOLO classification dataset from manual jersey readability reviews."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "readable": "number_readable",
    "unreadable": "number_unreadable",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--recognition-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--link-mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()

    split_manifest_path = Path(args.recognition_manifest).resolve()
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_sequences = set(split_manifest.get("train_sequences") or [])
    validation_sequences = set(split_manifest.get("validation_sequences") or [])
    frozen_sequences = set(split_manifest.get("frozen_sequences") or [])
    if not train_sequences or not validation_sequences:
        raise ValueError("recognition manifest must contain frozen train/validation sequences")
    if train_sequences & validation_sequences:
        raise ValueError("recognition manifest train/validation sequences overlap")

    rows, ignored = load_reviews([Path(path) for path in args.reviews])
    observed = {row["sequence"] for row in rows}
    allowed = train_sequences | validation_sequences
    if observed & frozen_sequences:
        raise ValueError(f"reviews contain frozen sequences: {sorted(observed & frozen_sequences)}")
    if observed - allowed:
        raise ValueError(f"review sequences are outside the recognition split: {sorted(observed-allowed)}")

    train_rows = [row for row in rows if row["sequence"] in train_sequences]
    validation_rows = [row for row in rows if row["sequence"] in validation_sequences]
    require_both_classes(train_rows, "train")
    require_both_classes(validation_rows, "validation")

    output = Path(args.output_dir).resolve()
    if (output / "train").exists() or (output / "val").exists():
        raise FileExistsError(f"dataset output already contains train/val: {output}")
    materialize(train_rows, output / "train", args.link_mode)
    materialize(validation_rows, output / "val", args.link_mode)

    dataset_rows = []
    for part, part_rows in (("train", train_rows), ("val", validation_rows)):
        for row in part_rows:
            dataset_rows.append({
                **row,
                "part": part,
                "dataset_path": str(
                    output / part / row["class_name"] / unique_name(row)
                ),
            })
    write_csv(output / "dataset.csv", dataset_rows)

    metadata = {
        "format": "ultralytics_classification",
        "target": "jersey_number_readability",
        "classes": ["number_readable", "number_unreadable"],
        "label_source": "manual_number_crop_readability_transferred_to_full_player_crop",
        "input_crop": "full_player_crop",
        "split_source": str(split_manifest_path),
        "split_source_sha256": sha256(split_manifest_path),
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "frozen_sequences": sorted(frozen_sequences),
        "frozen_sequences_observed": sorted(observed & frozen_sequences),
        "review_sha256": {
            str(path.resolve()): sha256(path) for path in map(Path, args.reviews)
        },
        "ignored_review_labels": dict(ignored),
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def load_reviews(paths):
    rows = {}
    ignored = Counter()
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                review = str(raw.get("review_label") or "").strip().lower()
                class_name = LABEL_MAP.get(review)
                if class_name is None:
                    ignored[review or "unreviewed"] += 1
                    continue
                crop_path = str(raw.get("crop_path") or "").strip()
                if not crop_path:
                    raise ValueError(f"review row has no full-player crop_path: {raw.get('audit_id')}")
                row = {
                    "audit_id": str(raw.get("audit_id") or ""),
                    "sequence": str(raw.get("sequence") or "").strip(),
                    "gt_track_id": str(raw.get("gt_track_id") or "").strip(),
                    "frame": integer(raw.get("frame")),
                    "crop_path": crop_path,
                    "review_label": review,
                    "class_name": class_name,
                }
                previous = rows.get(crop_path)
                if previous and previous["class_name"] != class_name:
                    raise ValueError(f"conflicting readability reviews: {crop_path}")
                rows[crop_path] = row
    return sorted(
        rows.values(),
        key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"], row["crop_path"]),
    ), ignored


def require_both_classes(rows, part):
    observed = {row["class_name"] for row in rows}
    if observed != set(LABEL_MAP.values()):
        raise ValueError(f"{part} split does not contain both readability classes: {sorted(observed)}")


def materialize(rows, root, mode):
    for row in rows:
        source = Path(row["crop_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"reviewed full-player crop missing: {source}")
        destination = root / row["class_name"] / unique_name(row)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)


def unique_name(row):
    source = Path(row["crop_path"])
    digest = hashlib.sha256(row["crop_path"].encode("utf-8")).hexdigest()[:12]
    suffix = source.suffix.lower() or ".jpg"
    return f"{row['sequence']}_track_{row['gt_track_id']}_f{row['frame']:06d}_{digest}{suffix}"


def summarize(rows):
    return {
        "images": len(rows),
        "sequences": len({row["sequence"] for row in rows}),
        "classes": dict(Counter(row["class_name"] for row in rows)),
        "tracklets": len({(row["sequence"], row["gt_track_id"]) for row in rows}),
    }


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
