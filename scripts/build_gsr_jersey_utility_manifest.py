#!/usr/bin/env python3
"""Build a sequence-disjoint GSR manifest for weak jersey-frame supervision."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsr-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--train-sequence-fraction", type=float, default=0.80)
    parser.add_argument("--min-track-frames", type=int, default=8)
    parser.add_argument("--max-frames-per-track", type=int, default=64)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--roles", nargs="+", default=["player"])
    parser.add_argument("--allow-valid", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    split = normalize_split(args.split)
    if split == "valid" and not args.allow_valid:
        raise ValueError("GSR valid is frozen for model selection; pass --allow-valid explicitly")
    if split == "test" and not args.allow_test:
        raise ValueError("GSR test is frozen; pass --allow-test only for final evaluation")
    if not 0.0 < args.train_sequence_fraction < 1.0:
        raise ValueError("--train-sequence-fraction must be between 0 and 1")
    if args.min_track_frames <= 0 or args.max_frames_per_track <= 0:
        raise ValueError("frame limits must be positive")

    root = Path(args.gsr_dir).resolve()
    label_files = discover_labels(root, split)
    if args.max_sequences is not None:
        label_files = label_files[:max(0, int(args.max_sequences))]
    if len(label_files) < 2:
        raise RuntimeError("at least two GSR sequences are required for a disjoint split")

    roles = {normalize_role(value) for value in args.roles}
    tracks, stats, label_hashes = load_tracks(
        root, label_files, roles, args.min_track_frames, args.max_frames_per_track
    )
    sequences = sorted({row["sequence"] for row in tracks})
    train_sequences, validation_sequences = split_sequences(
        sequences, args.train_sequence_fraction, args.seed
    )
    train_sequences, validation_sequences, split_repairs = repair_class_coverage(
        tracks, train_sequences, validation_sequences, args.seed
    )
    parts = {
        "train": [row for row in tracks if row["sequence"] in train_sequences],
        "validation": [row for row in tracks if row["sequence"] in validation_sequences],
    }
    if args.max_tracklets is not None:
        parts = stratified_track_limit(parts, args.max_tracklets, args.seed)
    if not all(parts.values()):
        raise RuntimeError("track filtering produced an empty train or validation partition")
    tracks = sorted(
        [row for rows in parts.values() for row in rows],
        key=lambda row: (row["sequence"], row["track_id"]),
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    manifest = {
        "dataset": "SoccerNet-GSR",
        "dataset_version_required": "1.3",
        "dataset_root": str(root),
        "split": split,
        "seed": args.seed,
        "train_sequence_fraction": args.train_sequence_fraction,
        "roles": sorted(roles),
        "min_track_frames": args.min_track_frames,
        "max_frames_per_track": args.max_frames_per_track,
        "train_sequences": sorted(train_sequences),
        "validation_sequences": sorted(validation_sequences),
        "split_repairs": split_repairs,
        "parts": {name: f"{name}.jsonl" for name in parts},
        "label_sha256": label_hashes,
        "ground_truth_usage": "training_only_track_level_jersey",
        "frame_orientation_labels": False,
    }
    summary = {
        "sequences": len(sequences),
        "train_sequences": len(train_sequences),
        "validation_sequences": len(validation_sequences),
        "train_tracklets": len(parts["train"]),
        "validation_tracklets": len(parts["validation"]),
        "train_frames": sum(len(row["frames"]) for row in parts["train"]),
        "validation_frames": sum(len(row["frames"]) for row in parts["validation"]),
        "jersey_distribution": dict(sorted(Counter(row["jersey"] for row in tracks).items())),
        "validation_only_classes": sorted(
            {row["jersey"] for row in parts["validation"]}
            - {row["jersey"] for row in parts["train"]}
        ),
        "split_repairs": split_repairs,
        "load_stats": dict(stats),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def discover_labels(root, split):
    aliases = [split]
    if split == "valid":
        aliases.append("val")
    paths = []
    for alias in aliases:
        split_root = root / alias
        if split_root.is_dir():
            paths.extend(sorted(split_root.glob("*/Labels-GameState.json")))
    return sorted(set(paths))


def load_tracks(root, label_files, roles, min_frames, max_frames):
    stats = Counter()
    grouped = defaultdict(list)
    label_hashes = {}
    for label_path in label_files:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        version = str((payload.get("info") or {}).get("version") or "")
        if version and version_tuple(version) < (1, 3):
            raise ValueError(f"GSR labels older than v1.3: {label_path} version={version}")
        label_hashes[label_path.parent.name] = sha256_file(label_path)
        images = {
            str(row.get("image_id", row.get("id"))): row
            for row in payload.get("images", [])
            if row.get("image_id", row.get("id")) is not None
        }
        categories = {row.get("id"): normalize_role(row.get("name"))
                      for row in payload.get("categories", [])}
        sequence = label_path.parent.name
        split = normalize_split(label_path.parent.parent.name)
        for annotation in payload.get("annotations", []):
            stats["annotations"] += 1
            image = images.get(str(annotation.get("image_id")))
            if not image or not image.get("file_name"):
                stats["skip_missing_image"] += 1
                continue
            attrs = annotation.get("attributes") or {}
            role = normalize_role(attrs.get("role") or categories.get(annotation.get("category_id")))
            if role not in roles:
                stats["skip_role"] += 1
                continue
            jersey = normalize_jersey(attrs)
            if jersey is None:
                stats["skip_jersey"] += 1
                continue
            track_id = annotation.get("track_id")
            bbox = normalize_bbox(annotation.get("bbox_image") or annotation.get("bbox") or {})
            if track_id in (None, "") or bbox is None:
                stats["skip_track_or_bbox"] += 1
                continue
            relative_image = Path(split) / sequence / "img1" / str(image["file_name"])
            if not (root / relative_image).is_file():
                stats["skip_missing_file"] += 1
                continue
            grouped[(sequence, str(track_id))].append({
                "frame": frame_index(image, image["file_name"]),
                "image": relative_image.as_posix(),
                "bbox_xywh": bbox,
                "jersey": jersey,
                "role": role,
            })
            stats["accepted_frames"] += 1

    tracks = []
    for (sequence, track_id), frames in sorted(grouped.items()):
        jerseys = {row["jersey"] for row in frames}
        if len(jerseys) != 1:
            stats["skip_inconsistent_jersey_track"] += 1
            continue
        unique_frames = {row["frame"]: row for row in frames}
        ordered = [unique_frames[key] for key in sorted(unique_frames)]
        if len(ordered) < min_frames:
            stats["skip_short_track"] += 1
            continue
        sampled = spread_sample(ordered, max_frames)
        tracks.append({
            "sequence": sequence,
            "track_id": track_id,
            "jersey": next(iter(jerseys)),
            "role": sampled[0]["role"],
            "source_frames": len(ordered),
            "frames": [{key: value for key, value in row.items() if key not in {"jersey", "role"}}
                       for row in sampled],
        })
    stats["accepted_tracks"] = len(tracks)
    return tracks, stats, label_hashes


def split_sequences(sequences, train_fraction, seed):
    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    train_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * train_fraction)))
    return set(shuffled[:train_count]), set(shuffled[train_count:])


def repair_class_coverage(tracks, train_sequences, validation_sequences, seed):
    """Ensure every validation jersey class occurs in sequence-disjoint train."""
    train_sequences = set(train_sequences)
    validation_sequences = set(validation_sequences)
    target_validation_count = len(validation_sequences)
    sequence_rows = defaultdict(list)
    for row in tracks:
        sequence_rows[row["sequence"]].append(row)
    repairs = []

    while True:
        train_classes = {
            row["jersey"] for sequence in train_sequences for row in sequence_rows[sequence]
        }
        validation_classes = {
            row["jersey"] for sequence in validation_sequences for row in sequence_rows[sequence]
        }
        missing = validation_classes - train_classes
        if not missing:
            break
        candidates = [
            sequence for sequence in validation_sequences
            if missing & {row["jersey"] for row in sequence_rows[sequence]}
        ]
        sequence = min(
            candidates,
            key=lambda value: (
                -len(missing & {row["jersey"] for row in sequence_rows[value]}),
                value,
            ),
        )
        repaired_classes = sorted(
            missing & {row["jersey"] for row in sequence_rows[sequence]}
        )
        validation_sequences.remove(sequence)
        train_sequences.add(sequence)
        repairs.append({
            "action": "validation_to_train_for_class_coverage",
            "sequence": sequence,
            "classes": repaired_classes,
        })

    # Restore the requested validation size when a train sequence can be moved
    # without removing the last train example of any jersey class.
    priority = sorted(train_sequences)
    random.Random(seed + 1).shuffle(priority)
    while len(validation_sequences) < target_validation_count:
        train_counts = Counter(
            row["jersey"] for sequence in train_sequences for row in sequence_rows[sequence]
        )
        eligible = []
        for sequence in priority:
            if sequence not in train_sequences:
                continue
            sequence_counts = Counter(row["jersey"] for row in sequence_rows[sequence])
            if all(train_counts[label] - count > 0 for label, count in sequence_counts.items()):
                eligible.append(sequence)
        if not eligible:
            repairs.append({
                "action": "validation_size_not_restored",
                "requested": target_validation_count,
                "actual": len(validation_sequences),
            })
            break
        sequence = eligible[0]
        train_sequences.remove(sequence)
        validation_sequences.add(sequence)
        repairs.append({
            "action": "train_to_validation_size_restore",
            "sequence": sequence,
            "classes": sorted({row["jersey"] for row in sequence_rows[sequence]}),
        })

    final_train_classes = {
        row["jersey"] for sequence in train_sequences for row in sequence_rows[sequence]
    }
    final_validation_classes = {
        row["jersey"] for sequence in validation_sequences for row in sequence_rows[sequence]
    }
    missing = sorted(final_validation_classes - final_train_classes)
    if missing:
        raise RuntimeError(f"validation contains classes absent from train after repair: {missing}")
    return train_sequences, validation_sequences, repairs


def stratified_track_limit(parts, limit, seed):
    """Cap total tracklets without dropping a sequence partition by chance."""
    limit = int(limit)
    if limit < len(parts):
        raise ValueError(f"--max-tracklets must be at least {len(parts)}")
    total = sum(len(rows) for rows in parts.values())
    if limit >= total:
        return {
            name: sorted(rows, key=lambda row: (row["sequence"], row["track_id"]))
            for name, rows in parts.items()
        }
    names = sorted(parts)
    allocations = {
        name: min(len(parts[name]), max(1, round(limit * len(parts[name]) / total)))
        for name in names
    }
    while sum(allocations.values()) > limit:
        candidates = [name for name in names if allocations[name] > 1]
        allocations[max(candidates, key=lambda name: allocations[name])] -= 1
    while sum(allocations.values()) < limit:
        candidates = [name for name in names if allocations[name] < len(parts[name])]
        if not candidates:
            break
        name = max(candidates, key=lambda item: len(parts[item]) - allocations[item])
        allocations[name] += 1
    output = {}
    for index, name in enumerate(names):
        rows = list(parts[name])
        random.Random(seed + index).shuffle(rows)
        output[name] = sorted(
            rows[:allocations[name]], key=lambda row: (row["sequence"], row["track_id"])
        )
    return output


def spread_sample(rows, limit):
    if len(rows) <= limit:
        return list(rows)
    indexes = sorted({round(index * (len(rows) - 1) / max(1, limit - 1)) for index in range(limit)})
    return [rows[index] for index in indexes]


def normalize_bbox(value):
    if not isinstance(value, dict):
        return None
    x = value.get("x", value.get("left"))
    y = value.get("y", value.get("top"))
    w = value.get("w", value.get("width"))
    h = value.get("h", value.get("height"))
    try:
        values = [float(x), float(y), float(w), float(h)]
    except (TypeError, ValueError):
        return None
    return values if values[2] > 0 and values[3] > 0 else None


def normalize_jersey(attributes):
    for key in ("jersey", "jersey_number", "shirt_number"):
        value = attributes.get(key)
        if value in (None, "", -1, "-1", "null"):
            continue
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            continue
        return number if 0 <= number <= 99 else None
    return None


def normalize_role(value):
    role = str(value or "").lower().replace("-", "_").replace(" ", "_")
    return {"field_player": "player", "outfield_player": "player"}.get(role, role)


def normalize_split(value):
    value = str(value).lower()
    return "valid" if value == "val" else value


def frame_index(image, filename):
    for key in ("frame", "frame_id", "id"):
        try:
            return int(image.get(key))
        except (TypeError, ValueError):
            pass
    try:
        return int(Path(filename).stem)
    except ValueError:
        return 0


def version_tuple(value):
    output = []
    for token in str(value).lower().lstrip("v").split("."):
        try:
            output.append(int(token))
        except ValueError:
            break
    return tuple(output)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
