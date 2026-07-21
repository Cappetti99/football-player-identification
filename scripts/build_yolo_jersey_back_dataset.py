#!/usr/bin/env python3
"""Build a sequence-disjoint YOLO classification dataset from crop reviews."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "clean_back": "clean_back",
    "usable_not_back": "clean_back",
    "not_clean": "not_clean",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-sequence-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--link-mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()
    if not 0.0 < args.train_sequence_fraction < 1.0:
        raise ValueError("--train-sequence-fraction must be between zero and one")

    manifest_path = Path(args.sequence_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, ignored = load_reviews([Path(path) for path in args.reviews])
    train_rows, validation_rows, split = split_review_rows(
        rows, manifest, args.train_sequence_fraction, args.seed
    )
    output = Path(args.output_dir).resolve()
    if (output / "train").exists() or (output / "val").exists():
        raise FileExistsError(f"dataset output already contains train/val: {output}")
    materialize(train_rows, output / "train", args.link_mode)
    materialize(validation_rows, output / "val", args.link_mode)
    dataset_rows = []
    for part, part_rows in (("train", train_rows), ("val", validation_rows)):
        for row in part_rows:
            dataset_rows.append({**row, "part": part, "dataset_path": dataset_path(output, part, row)})
    write_csv(output / "dataset.csv", dataset_rows)
    review_hashes = {str(path.resolve()): sha256(path) for path in map(Path, args.reviews)}
    metadata = {
        "format": "ultralytics_classification",
        "classes": ["clean_back", "not_clean"],
        "seed": args.seed,
        "train_sequence_fraction": args.train_sequence_fraction,
        **split,
        "frozen_validation_sequences": manifest.get("validation_sequences") or [],
        "review_sha256": review_hashes,
        "ignored_review_labels": dict(ignored),
        "train": summarize(train_rows),
        "validation": summarize(validation_rows),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def load_reviews(paths):
    by_crop = {}
    ignored = Counter()
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                review = str(raw.get("review_label") or "").strip()
                label = LABEL_MAP.get(review)
                if label is None:
                    ignored[review or "unreviewed"] += 1
                    continue
                crop = str(raw.get("crop_path") or "").strip()
                if not crop:
                    raise ValueError(f"review row has no crop_path in {path}")
                row = {
                    "sequence": str(raw.get("sequence") or "").strip(),
                    "gt_track_id": str(raw.get("gt_track_id") or "").strip(),
                    "frame": integer(raw.get("frame")),
                    "crop_path": crop,
                    "review_label": review,
                    "class_name": label,
                }
                previous = by_crop.get(crop)
                if previous and previous["class_name"] != label:
                    raise ValueError(f"conflicting reviews for crop: {crop}")
                by_crop[crop] = row
    return sorted(by_crop.values(), key=lambda r: (r["sequence"], r["gt_track_id"], r["frame"], r["crop_path"])), ignored


def split_review_rows(rows, manifest, fraction, seed):
    if manifest.get("split") != "train":
        raise ValueError("YOLO back dataset requires a GSR train manifest")
    allowed = set(manifest.get("train_sequences") or [])
    frozen = set(manifest.get("validation_sequences") or [])
    observed = {row["sequence"] for row in rows}
    if observed & frozen:
        raise ValueError(f"reviews contain frozen validation sequences: {sorted(observed & frozen)}")
    if observed - allowed:
        raise ValueError(f"review sequences are not in manifest train: {sorted(observed - allowed)}")
    sequences = sorted(observed)
    if len(sequences) < 2:
        raise ValueError("at least two reviewed train sequences are required")
    rng = random.Random(seed); rng.shuffle(sequences)
    count = min(len(sequences) - 1, max(1, round(len(sequences) * fraction)))
    chosen = None
    for offset in range(len(sequences)):
        rotated = sequences[offset:] + sequences[:offset]
        candidate = set(rotated[:count])
        train_labels = {r["class_name"] for r in rows if r["sequence"] in candidate}
        val_labels = {r["class_name"] for r in rows if r["sequence"] not in candidate}
        if train_labels == set(LABEL_MAP.values()) and val_labels == set(LABEL_MAP.values()):
            chosen = candidate; break
    if chosen is None:
        raise ValueError("could not create sequence-disjoint splits containing both classes")
    return (
        [r for r in rows if r["sequence"] in chosen],
        [r for r in rows if r["sequence"] not in chosen],
        {"train_sequences": sorted(chosen), "validation_sequences": sorted(observed - chosen)},
    )


def materialize(rows, root, mode):
    for row in rows:
        source = Path(row["crop_path"]).resolve()
        if not source.is_file(): raise FileNotFoundError(f"missing reviewed crop: {source}")
        destination = root / row["class_name"] / unique_name(row, source.suffix or ".jpg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): continue
        if mode == "symlink": destination.symlink_to(source)
        else: shutil.copy2(source, destination)


def unique_name(row, suffix):
    digest = hashlib.sha256(row["crop_path"].encode("utf-8")).hexdigest()[:12]
    return f"{row['sequence']}_track_{row['gt_track_id']}_frame_{row['frame']}_{digest}{suffix.lower()}"


def dataset_path(root, part, row):
    return str(root / part / row["class_name"] / unique_name(row, Path(row["crop_path"]).suffix or ".jpg"))


def summarize(rows):
    return {"images": len(rows), "sequences": len({r["sequence"] for r in rows}),
            "classes": dict(Counter(r["class_name"] for r in rows))}


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields: writer.writeheader(); writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def integer(value):
    try: return int(float(value))
    except (TypeError, ValueError): return 0


if __name__ == "__main__": main()
