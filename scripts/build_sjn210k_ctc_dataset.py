#!/usr/bin/env python3
"""Build a storage-light numeric CTC dataset from localized SJN-210k rows."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sjn-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--box-padding", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.box_padding <= 0.5:
        raise ValueError("--box-padding must be between 0 and 0.5")

    source_manifest_path = Path(args.sjn_manifest).resolve()
    source_manifest = json.loads(source_manifest_path.read_text())
    validate_source_manifest(source_manifest)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    partitions = {}
    for source_part, target_part in (("train", "train"), ("test", "validation")):
        source = Path(source_manifest["recognition"][source_part]).resolve()
        rows = build_rows(read_jsonl(source), source_part, args.box_padding)
        if not rows:
            raise RuntimeError(f"empty localized SJN partition: {source_part}")
        write_jsonl(output / f"{target_part}.jsonl", rows)
        partitions[target_part] = summarize(rows)

    manifest = {
        "format": "jersey_numeric_ctc_sjn210k_v1",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "official_split_preserved": True,
        "official_test_role": "validation_and_checkpoint_selection_only",
        "test_used_for_gradient_updates": False,
        "alphabet": "0123456789",
        "blank_index": 10,
        "max_text_length": 2,
        "crop_source": "sjn210k_localized_number_region",
        "box_padding": args.box_padding,
        **partitions,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def validate_source_manifest(manifest):
    if manifest.get("format") != "sjn210k_ft_v1":
        raise ValueError("unexpected SJN source manifest format")
    if manifest.get("official_split_preserved") is not True:
        raise ValueError("SJN source does not preserve its official split")
    if manifest.get("test_used_for_gradient_updates") is not False:
        raise ValueError("SJN source permits test gradient updates")
    if not all((manifest.get("recognition") or {}).get(part) for part in ("train", "test")):
        raise ValueError("SJN source manifest has incomplete recognition paths")


def build_rows(rows, part, padding):
    output = []
    for row in rows:
        quad = row.get("clipped_quad")
        if quad is None:
            continue
        image = Path(row["image"]).resolve()
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(image)
        text = str(row["text"])
        if not text.isdigit() or not 0 <= int(text) <= 99:
            raise ValueError(f"invalid SJN numeric label: {text!r}")
        output.append({
            "sequence": f"SJN-{part}",
            "gt_track_id": image.stem,
            "frame": 0,
            "image": str(image),
            "crop_box": padded_quad_box(quad, padding),
            "text": text,
        })
    return output


def padded_quad_box(quad, fraction):
    xs, ys = quad[0::2], quad[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("degenerate SJN quadrilateral")
    return [
        max(0.0, xmin - width * fraction),
        max(0.0, ymin - height * fraction),
        min(1.0, xmax + width * fraction),
        min(1.0, ymax + height * fraction),
    ]


def summarize(rows):
    return {
        "images": len(rows),
        "tracklets": len(rows),
        "one_digit": sum(len(row["text"]) == 1 for row in rows),
        "two_digit": sum(len(row["text"]) == 2 for row in rows),
        "digit_distribution": dict(sorted(Counter(
            character for row in rows for character in row["text"]
        ).items())),
    }


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
