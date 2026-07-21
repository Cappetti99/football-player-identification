#!/usr/bin/env python3
"""Build sequence-disjoint numeric CTC manifests from reviewed jersey crops."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-crops-per-track", type=int, default=12)
    args = parser.parse_args()
    if args.max_crops_per_track < 1:
        raise ValueError("--max-crops-per-track must be positive")

    split = json.loads(Path(args.sequence_manifest).read_text(encoding="utf-8"))
    train_sequences = set(split.get("train_sequences") or [])
    validation_sequences = set(split.get("validation_sequences") or [])
    frozen_sequences = set(
        split.get("frozen_validation_sequences")
        or split.get("frozen_sequences")
        or []
    )
    if not train_sequences or not validation_sequences or train_sequences & validation_sequences:
        raise ValueError("manifest must contain disjoint train_sequences and validation_sequences")
    rows, ignored = load_reviews(args.reviews)
    observed = {row["sequence"] for row in rows}
    leakage = observed & frozen_sequences
    unknown = observed - train_sequences - validation_sequences
    if leakage or unknown:
        raise ValueError(f"sequence leakage: frozen={sorted(leakage)} unknown={sorted(unknown)}")
    rows = limit_tracks(rows, args.max_crops_per_track)
    train = [row for row in rows if row["sequence"] in train_sequences]
    validation = [row for row in rows if row["sequence"] in validation_sequences]
    if not train or not validation:
        raise RuntimeError("filtering produced an empty train or validation partition")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    manifest = {
        "format": "jersey_numeric_ctc_v1",
        "alphabet": "0123456789",
        "blank_index": 10,
        "max_text_length": 2,
        "max_crops_per_track": args.max_crops_per_track,
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "frozen_sequences": sorted(frozen_sequences),
        "frozen_sequences_observed": sorted(leakage),
        "review_sha256": {str(Path(path).resolve()): sha256(path) for path in args.reviews},
        "ignored_labels": dict(ignored),
        "train": summarize(train),
        "validation": summarize(validation),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_reviews(paths):
    rows = {}
    ignored = Counter()
    for path in map(Path, paths):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                label = str(raw.get("review_label") or "").strip().lower()
                if label != "readable":
                    ignored[label or "unreviewed"] += 1
                    continue
                text = normalized_text(raw)
                image = Path(str(raw.get("number_crop_path") or ""))
                if not image.is_file():
                    raise FileNotFoundError(f"number crop missing: {image}")
                row = {
                    "audit_id": str(raw.get("audit_id") or ""),
                    "sequence": str(raw.get("sequence") or ""),
                    "gt_track_id": str(raw.get("gt_track_id") or ""),
                    "frame": int(float(raw.get("frame") or 0)),
                    "image": str(image.resolve()),
                    "text": text,
                }
                rows[str(image.resolve())] = row
    return sorted(rows.values(), key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"])), ignored


def normalized_text(row):
    gt = str(row.get("gt_jersey") or "").strip()
    transcription = str(row.get("transcription") or "").strip()
    text = transcription or gt
    if not text or not text.isdigit() or not 0 <= int(text) <= 99 or len(text) > 2:
        raise ValueError(f"invalid numeric transcription: {text!r}")
    if transcription and gt and int(transcription) != int(float(gt)):
        raise ValueError(
            f"manual transcription disagrees with GT: {row.get('audit_id')} "
            f"text={transcription} gt={gt}"
        )
    return str(int(text))


def limit_tracks(rows, maximum):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["sequence"], row["gt_track_id"]), []).append(row)
    output = []
    for values in grouped.values():
        values.sort(key=lambda row: row["frame"])
        if len(values) <= maximum:
            output.extend(values)
            continue
        indices = [round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)] if maximum > 1 else [len(values) // 2]
        output.extend(values[index] for index in indices)
    return sorted(output, key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"]))


def summarize(rows):
    return {
        "images": len(rows),
        "sequences": len({row["sequence"] for row in rows}),
        "tracklets": len({(row["sequence"], row["gt_track_id"]) for row in rows}),
        "one_digit": sum(len(row["text"]) == 1 for row in rows),
        "two_digit": sum(len(row["text"]) == 2 for row in rows),
        "digit_distribution": dict(sorted(Counter(char for row in rows for char in row["text"]).items())),
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
