#!/usr/bin/env python3
"""Group an existing sequence-disjoint GSR CTC dataset into track bags."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames-per-track", type=int, default=16)
    args = parser.parse_args()
    if args.max_frames_per_track < 1:
        raise ValueError("--max-frames-per-track must be positive")

    source = Path(args.source_dataset).resolve()
    source_manifest_path = source / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_source_manifest(manifest)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    summaries = {}
    observed = {}
    for part in ("train", "validation"):
        rows = read_jsonl(source / f"{part}.jsonl")
        bags = group_track_bags(rows, args.max_frames_per_track)
        write_jsonl(output / f"{part}.jsonl", bags)
        summaries[part] = summarize(bags)
        observed[part] = sorted({row["sequence"] for row in bags})

    if set(observed["train"]) & set(observed["validation"]):
        raise ValueError("train and validation sequences overlap")

    result = {
        "format": "jersey_multiframe_digits_v1",
        "source_dataset": str(source),
        "source_manifest_sha256": sha256(source_manifest_path),
        "max_frames_per_track": args.max_frames_per_track,
        "train_sequences": observed["train"],
        "validation_sequences": observed["validation"],
        "frozen_sequences_observed": [],
        "train": summaries["train"],
        "validation": summaries["validation"],
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def validate_source_manifest(manifest):
    if manifest.get("format") != "jersey_numeric_ctc_v1":
        raise ValueError("source must be a GSR jersey_numeric_ctc_v1 dataset")
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("source dataset observes frozen sequences")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def group_track_bags(rows, maximum):
    groups = defaultdict(list)
    for row in rows:
        number = int(row["text"])
        if not 0 <= number <= 99:
            raise ValueError(f"invalid jersey number: {number}")
        groups[(row["sequence"], str(row["gt_track_id"]))].append(row)
    bags = []
    for (sequence, track), values in sorted(groups.items()):
        truths = {int(row["text"]) for row in values}
        if len(truths) != 1:
            raise ValueError(f"inconsistent track GT: {sequence}/{track}: {truths}")
        values.sort(key=lambda row: int(row["frame"]))
        selected = temporal_sample(values, maximum)
        bags.append({
            "sequence": sequence,
            "gt_track_id": track,
            "jersey": truths.pop(),
            "frames": [
                {
                    "frame": int(row["frame"]),
                    "image": row["image"],
                    **({"crop_box": row["crop_box"]} if row.get("crop_box") is not None else {}),
                }
                for row in selected
            ],
        })
    return bags


def temporal_sample(rows, maximum):
    if len(rows) <= maximum:
        return list(rows)
    if maximum == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum)]
    return [rows[index] for index in indices]


def summarize(rows):
    return {
        "tracklets": len(rows),
        "frames": sum(len(row["frames"]) for row in rows),
        "sequences": len({row["sequence"] for row in rows}),
        "one_digit": sum(row["jersey"] < 10 for row in rows),
        "two_digit": sum(row["jersey"] >= 10 for row in rows),
        "jersey_distribution": dict(sorted(Counter(row["jersey"] for row in rows).items())),
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
