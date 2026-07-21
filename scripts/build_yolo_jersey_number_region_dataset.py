#!/usr/bin/env python3
"""Build a sequence-disjoint Ultralytics number-region detection dataset."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.sequence_manifest).read_text())
    train_sequences = set(split.get("train_sequences") or [])
    validation_sequences = set(split.get("validation_sequences") or [])
    frozen = set(split.get("frozen_validation_sequences") or split.get("frozen_sequences") or [])
    rows, ignored = load_annotations(args.annotations)
    observed = {row["sequence"] for row in rows}
    if observed & frozen or observed - train_sequences - validation_sequences:
        raise ValueError("number-region annotations leak outside the declared train/development split")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    counts = {}
    for part, sequences in (("train", train_sequences), ("val", validation_sequences)):
        selected = [row for row in rows if row["sequence"] in sequences]
        if not selected:
            raise RuntimeError(f"empty {part} partition")
        counts[part] = materialize(selected, output, part)
    yaml = (
        f"path: {output}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: number_region\n"
    )
    (output / "data.yaml").write_text(yaml)
    manifest = {
        "format": "ultralytics_detection",
        "target": "jersey_number_region",
        "classes": ["number_region"],
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "frozen_sequences": sorted(frozen),
        "frozen_sequences_observed": sorted(observed & frozen),
        "annotation_sha256": sha256(args.annotations),
        "ignored_labels": dict(ignored),
        **counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_annotations(path):
    rows, ignored = [], Counter()
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            label = str(raw.get("region_label") or "").strip().lower()
            if label not in {"present", "absent"}:
                ignored[label or "unreviewed"] += 1
                continue
            box = None
            if label == "present":
                box = tuple(float(raw[name]) for name in ("xmin", "ymin", "xmax", "ymax"))
                if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                    raise ValueError(f"invalid normalized box: {raw.get('audit_id')} {box}")
            source = Path(str(raw.get("crop_path") or ""))
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append({
                "audit_id": raw.get("audit_id"), "sequence": raw.get("sequence"),
                "frame": int(float(raw.get("frame") or 0)), "source": source, "label": label, "box": box,
            })
    return rows, ignored


def materialize(rows, output, part):
    images = output / "images" / part
    labels = output / "labels" / part
    images.mkdir(parents=True); labels.mkdir(parents=True)
    positives = 0
    for row in rows:
        digest = hashlib.sha256(str(row["source"]).encode()).hexdigest()[:10]
        stem = f"{row['sequence']}_f{row['frame']:06d}_{digest}"
        shutil.copy2(row["source"], images / f"{stem}{row['source'].suffix.lower()}")
        content = ""
        if row["box"]:
            xmin, ymin, xmax, ymax = row["box"]
            content = f"0 {(xmin+xmax)/2:.8f} {(ymin+ymax)/2:.8f} {xmax-xmin:.8f} {ymax-ymin:.8f}\n"
            positives += 1
        (labels / f"{stem}.txt").write_text(content)
    return {"images": len(rows), "positive": positives, "negative": len(rows) - positives, "sequences": len({row['sequence'] for row in rows})}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
