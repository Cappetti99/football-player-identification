#!/usr/bin/env python3
"""Build numeric CTC data from manually annotated tight jersey-number regions."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-padding", type=float, default=0.10)
    parser.add_argument("--max-crops-per-track", type=int, default=12)
    args = parser.parse_args()
    if not 0 <= args.box_padding <= 0.5 or args.max_crops_per_track < 1:
        raise ValueError("invalid padding or track limit")
    split = json.loads(Path(args.sequence_manifest).read_text())
    train_sequences = set(split.get("train_sequences") or [])
    validation_sequences = set(split.get("validation_sequences") or [])
    frozen = set(split.get("frozen_validation_sequences") or split.get("frozen_sequences") or [])
    rows, ignored = load_annotations(args.annotations)
    observed = {row["sequence"] for row in rows}
    if observed & frozen or observed - train_sequences - validation_sequences:
        raise ValueError("region CTC annotations leak outside declared train/development split")
    rows = limit_tracks(rows, args.max_crops_per_track)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    partitions = {}
    for name, sequences in (("train", train_sequences), ("validation", validation_sequences)):
        selected = [row for row in rows if row["sequence"] in sequences]
        if not selected:
            raise RuntimeError(f"empty {name} partition")
        materialized = materialize(selected, output / "images" / name, args.box_padding)
        write_jsonl(output / f"{name}.jsonl", materialized)
        partitions[name] = summarize(materialized)
    manifest = {
        "format": "jersey_numeric_ctc_v1",
        "crop_source": "manual_number_region",
        "alphabet": "0123456789", "blank_index": 10, "max_text_length": 2,
        "box_padding": args.box_padding, "max_crops_per_track": args.max_crops_per_track,
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "frozen_sequences": sorted(frozen), "frozen_sequences_observed": sorted(observed & frozen),
        "annotation_sha256": sha256(args.annotations), "ignored_labels": dict(ignored),
        **partitions,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_annotations(path):
    rows, ignored = [], Counter()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            label = str(raw.get("region_label") or "").strip().lower()
            if label != "present":
                ignored[label or "unreviewed"] += 1
                continue
            box = tuple(float(raw[name]) for name in ("xmin", "ymin", "xmax", "ymax"))
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError(f"invalid box for {raw.get('audit_id')}: {box}")
            text = str(int(float(raw.get("gt_jersey") or -1)))
            if not text.isdigit() or not 0 <= int(text) <= 99:
                raise ValueError(f"invalid jersey GT for {raw.get('audit_id')}: {text}")
            source = Path(str(raw.get("crop_path") or ""))
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append({
                "audit_id": str(raw.get("audit_id") or ""), "sequence": str(raw["sequence"]),
                "gt_track_id": str(raw["gt_track_id"]), "frame": int(float(raw["frame"])),
                "source": source, "box": box, "text": text,
            })
    return rows, ignored


def materialize(rows, directory, padding):
    from PIL import Image

    directory.mkdir(parents=True)
    output = []
    for row in rows:
        digest = hashlib.sha256(str(row["source"]).encode()).hexdigest()[:10]
        destination = directory / f"{row['sequence']}_track_{row['gt_track_id']}_f{row['frame']:06d}_{digest}.jpg"
        with Image.open(row["source"]) as image:
            width, height = image.size
            xmin, ymin, xmax, ymax = padded_box(row["box"], padding)
            image.convert("RGB").crop((
                round(xmin * width), round(ymin * height),
                round(xmax * width), round(ymax * height),
            )).save(destination, quality=95)
        output.append({
            "audit_id": row["audit_id"], "sequence": row["sequence"],
            "gt_track_id": row["gt_track_id"], "frame": row["frame"],
            "image": str(destination.resolve()), "text": row["text"],
        })
    return output


def padded_box(box, fraction):
    xmin, ymin, xmax, ymax = box
    width, height = xmax - xmin, ymax - ymin
    return max(0, xmin-width*fraction), max(0, ymin-height*fraction), min(1, xmax+width*fraction), min(1, ymax+height*fraction)


def limit_tracks(rows, maximum):
    groups = {}
    for row in rows:
        groups.setdefault((row["sequence"], row["gt_track_id"]), []).append(row)
    output = []
    for values in groups.values():
        values.sort(key=lambda row: row["frame"])
        if len(values) <= maximum:
            output.extend(values); continue
        indices = [round(index*(len(values)-1)/(maximum-1)) for index in range(maximum)] if maximum > 1 else [len(values)//2]
        output.extend(values[index] for index in indices)
    return sorted(output, key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"]))


def summarize(rows):
    return {
        "images": len(rows), "sequences": len({row["sequence"] for row in rows}),
        "tracklets": len({(row["sequence"], row["gt_track_id"]) for row in rows}),
        "one_digit": sum(len(row["text"]) == 1 for row in rows),
        "two_digit": sum(len(row["text"]) == 2 for row in rows),
        "digit_distribution": dict(sorted(Counter(char for row in rows for char in row["text"]).items())),
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
