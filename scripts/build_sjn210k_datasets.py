#!/usr/bin/env python3
"""Build reproducible recognition and YOLO region datasets from SJN-210k."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sjn-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-coordinate-excess", type=float, default=0.05)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()
    if not 0 <= args.max_coordinate_excess <= 0.5:
        parser.error("--max-coordinate-excess must be between 0 and 0.5")
    return args


def parse_label_line(line, split, line_number):
    fields = line.split()
    if len(fields) not in (3, 11):
        raise ValueError(f"{split} line {line_number}: expected 3 or 11 fields")
    filename = fields[0]
    first, second = int(fields[1]), int(fields[2])
    if first not in range(10) or second not in range(11):
        raise ValueError(f"{split} line {line_number}: invalid digits {first}, {second}")
    text = str(first) if second == 10 else f"{first}{second}"
    quad = [float(value) for value in fields[3:]] if len(fields) == 11 else None
    if quad is not None and len(quad) != 8:
        raise ValueError(f"{split} line {line_number}: invalid quadrilateral")
    return {
        "filename": filename,
        "first_digit": first,
        "second_digit": None if second == 10 else second,
        "text": text,
        "quad": quad,
    }


def localization_policy(quad, maximum_excess):
    if quad is None:
        return "missing", None, 0.0
    excess = max(max(0.0, -min(quad)), max(0.0, max(quad) - 1.0))
    if excess > maximum_excess:
        return "excluded_excess", None, excess
    clipped = [min(1.0, max(0.0, value)) for value in quad]
    status = "clipped" if excess > 0 else "valid"
    return status, clipped, excess


def quadrilateral_bbox(quad):
    xs, ys = quad[0::2], quad[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate quadrilateral: {quad}")
    return (
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        width,
        height,
    )


def load_split(root, split, maximum_excess):
    label_path = root / split / f"{split}_pos_label.txt"
    image_dir = root / split / "images"
    if not label_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"missing extracted {split} inputs under {root}")
    rows = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = parse_label_line(line, split, line_number)
        image = (image_dir / row["filename"]).resolve()
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(image)
        status, clipped, excess = localization_policy(row["quad"], maximum_excess)
        rows.append({
            **row,
            "split": split,
            "image": str(image),
            "localization_status": status,
            "clipped_quad": clipped,
            "coordinate_excess": excess,
        })
    return rows, label_path


def materialize_yolo(rows, output, part, link_mode):
    image_dir = output / "images" / part
    label_dir = output / "labels" / part
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    written = 0
    for row in rows:
        if row["clipped_quad"] is None:
            continue
        source = Path(row["image"])
        destination = image_dir / source.name
        if link_mode == "symlink":
            os.symlink(source, destination)
        else:
            shutil.copy2(source, destination)
        x, y, width, height = quadrilateral_bbox(row["clipped_quad"])
        (label_dir / f"{source.stem}.txt").write_text(
            f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n"
        )
        written += 1
    return written


def recognition_row(row):
    return {
        "image": row["image"],
        "text": row["text"],
        "first_digit": row["first_digit"],
        "second_digit": row["second_digit"],
        "length": len(row["text"]),
        "has_localization": row["quad"] is not None,
        "localization_status": row["localization_status"],
        "quad": row["quad"],
        "clipped_quad": row["clipped_quad"],
        "coordinate_excess": row["coordinate_excess"],
    }


def summarize(rows):
    statuses = Counter(row["localization_status"] for row in rows)
    numbers = Counter(row["text"] for row in rows)
    digits = Counter()
    for row in rows:
        digits.update(row["text"])
    return {
        "images": len(rows),
        "one_digit": sum(len(row["text"]) == 1 for row in rows),
        "two_digit": sum(len(row["text"]) == 2 for row in rows),
        "number_classes": len(numbers),
        "digit_distribution": dict(sorted(digits.items())),
        "localization_status": dict(sorted(statuses.items())),
        "yolo_eligible": sum(row["clipped_quad"] is not None for row in rows),
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    root = Path(args.sjn_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    recognition = output / "recognition"
    yolo = output / "number_region_yolo"
    recognition.mkdir()
    yolo.mkdir()

    partitions = {}
    label_hashes = {}
    for split, yolo_part in (("train", "train"), ("test", "val")):
        rows, label_path = load_split(root, split, args.max_coordinate_excess)
        write_jsonl(recognition / f"{split}.jsonl", map(recognition_row, rows))
        yolo_images = materialize_yolo(rows, yolo, yolo_part, args.link_mode)
        stats = summarize(rows)
        if yolo_images != stats["yolo_eligible"]:
            raise RuntimeError(f"YOLO materialization mismatch for {split}")
        partitions[split] = stats
        label_hashes[split] = sha256(label_path)

    dataset_yaml = (
        f"path: {yolo}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: number_region\n"
    )
    (yolo / "data.yaml").write_text(dataset_yaml)
    manifest = {
        "format": "sjn210k_ft_v1",
        "source": str(root),
        "official_split_preserved": True,
        "official_test_role": "validation_and_checkpoint_selection_only",
        "test_used_for_gradient_updates": False,
        "license": "CC-BY-SA-4.0",
        "max_coordinate_excess": args.max_coordinate_excess,
        "localization_policy": {
            "valid": "preserve",
            "excess_lte_threshold": "clip_to_unit_interval",
            "excess_gt_threshold": "exclude_from_detector_only",
            "recognition": "retain_all_valid_numeric_labels",
        },
        "link_mode": args.link_mode,
        "label_sha256": label_hashes,
        "recognition": {
            "train": str(recognition / "train.jsonl"),
            "test": str(recognition / "test.jsonl"),
        },
        "number_region_yolo": str(yolo / "data.yaml"),
        **partitions,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
