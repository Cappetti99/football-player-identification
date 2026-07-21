#!/usr/bin/env python3
"""Build sequence-disjoint MMOCR text-recognition JSON from manual reviews."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", nargs="+", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-sequence-fraction", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    manifest = json.loads(Path(args.sequence_manifest).read_text())
    allowed = set(manifest.get("train_sequences") or [])
    frozen = set(manifest.get("validation_sequences") or [])
    rows, ignored = load_reviews(args.reviews)
    observed = {row["sequence"] for row in rows}
    if observed & frozen or observed - allowed:
        raise ValueError(f"review sequence leakage: frozen={sorted(observed & frozen)} unknown={sorted(observed-allowed)}")
    train, validation, split = split_rows(rows, args.train_sequence_fraction, args.seed)
    output = Path(args.output_dir).resolve()
    if (output / "images").exists():
        raise FileExistsError(f"dataset already exists: {output}")
    materialize(train + validation, output / "images")
    write_mmocr(output / "train.json", train, output)
    write_mmocr(output / "validation.json", validation, output)
    metadata = {
        "format": "mmocr_textrecog_v1",
        "alphabet": "0123456789",
        "max_text_length": 2,
        "supervision": "manual_crop_readability_plus_gsr_train_track_jersey",
        "seed": args.seed,
        **split,
        "frozen_sequences": sorted(frozen),
        "frozen_sequences_observed": sorted(observed & frozen),
        "ignored_reviews": dict(ignored),
        "train": summarize(train),
        "validation": summarize(validation),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


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
                gt = str(raw.get("gt_jersey") or "").strip()
                if not gt or not 1 <= int(float(gt)) <= 99:
                    raise ValueError(f"readable crop has invalid GSR GT: {raw.get('audit_id')} gt={gt}")
                manual_text = str(raw.get("transcription") or "").strip()
                if manual_text and (not manual_text.isdigit() or not 1 <= len(manual_text) <= 2):
                    raise ValueError(f"manual transcription must contain 1-2 digits: {raw.get('audit_id')}")
                if manual_text and str(int(manual_text)) != str(int(float(gt))):
                    raise ValueError(
                        f"manual transcription disagrees with GSR GT: "
                        f"{raw.get('audit_id')} text={manual_text} gt={gt}"
                    )
                text = str(int(manual_text)) if manual_text else str(int(float(gt)))
                image = str(raw.get("number_crop_path") or "").strip()
                if not Path(image).is_file():
                    raise FileNotFoundError(f"reviewed number crop missing: {image}")
                row = {
                    "audit_id": raw.get("audit_id"), "sequence": raw.get("sequence"),
                    "gt_track_id": raw.get("gt_track_id"), "frame": int(float(raw.get("frame") or 0)),
                    "image": image,
                    "text": text,
                    "label_source": (
                        "manual_transcription_verified_against_track_gt"
                        if manual_text else "manual_readability_plus_track_gt"
                    ),
                }
                rows[image] = row
    return sorted(rows.values(), key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"])), ignored


def split_rows(rows, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("train fraction must be between zero and one")
    sequences = sorted({row["sequence"] for row in rows})
    if len(sequences) < 2:
        raise ValueError("at least two readable sequences are required")
    random.Random(seed).shuffle(sequences)
    count = min(len(sequences)-1, max(1, round(len(sequences) * fraction)))
    train_sequences = set(sequences[:count])
    return (
        [row for row in rows if row["sequence"] in train_sequences],
        [row for row in rows if row["sequence"] not in train_sequences],
        {"train_sequences": sorted(train_sequences), "validation_sequences": sorted(set(sequences)-train_sequences)},
    )


def materialize(rows, root):
    root.mkdir(parents=True)
    for row in rows:
        destination = root / image_name(row)
        shutil.copy2(row["image"], destination)


def image_name(row):
    digest = hashlib.sha256(row["image"].encode()).hexdigest()[:10]
    return f"{row['sequence']}_track_{row['gt_track_id']}_f{row['frame']:06d}_{digest}.jpg"


def write_mmocr(path, rows, root):
    payload = {
        "metainfo": {"dataset_type": "TextRecogDataset", "task_name": "textrecog"},
        "data_list": [{
            "img_path": str((root / "images" / image_name(row)).relative_to(root)),
            "instances": [{"text": row["text"]}],
        } for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows):
    return {
        "images": len(rows), "sequences": len({row["sequence"] for row in rows}),
        "one_digit": sum(len(row["text"]) == 1 for row in rows),
        "two_digit": sum(len(row["text"]) == 2 for row in rows),
        "digits": dict(sorted(Counter(char for row in rows for char in row["text"]).items())),
    }


if __name__ == "__main__":
    main()
