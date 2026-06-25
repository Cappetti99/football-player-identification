#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export ground-truth SoccerNet-GSR crops for FT-native PRTReID training.")
    parser.add_argument("--gsr-dir", required=True)
    parser.add_argument("--output-dir", default="datasets/prtreid_ft_v1")
    parser.add_argument("--train-split", action="append", default=["train"])
    parser.add_argument("--valid-split", action="append", default=["valid", "val"])
    parser.add_argument("--min-samples-per-id", type=int, default=4)
    parser.add_argument("--max-samples-per-id", type=int, default=15)
    parser.add_argument("--query-ratio", type=float, default=0.20)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()

    gsr_dir = Path(args.gsr_dir)
    label_paths = sorted(gsr_dir.rglob("Labels-GameState.json"))
    if not label_paths:
        raise SystemExit(f"No Labels-GameState.json found under {gsr_dir}")

    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "images"
    samples = []
    stats = Counter()
    train_splits = {value.lower() for value in args.train_split}
    valid_splits = {value.lower() for value in args.valid_split}
    selected_per_target = Counter()
    for label_path in label_paths:
        source_split = infer_split(label_path)
        if source_split not in train_splits | valid_splits:
            continue
        target_split = "train" if source_split in train_splits else "valid"
        # A limited smoke export still needs both train and validation data.
        # Apply the limit independently to each target split instead of
        # truncating the globally sorted path list (which normally starts with
        # every train sequence).
        if args.max_sequences is not None and selected_per_target[target_split] >= args.max_sequences:
            continue
        sequence_rows, sequence_stats = export_sequence(
            label_path,
            target_split,
            crops_dir,
            max_samples_per_id=args.max_samples_per_id,
        )
        samples.extend(sequence_rows)
        stats.update(sequence_stats)
        selected_per_target[target_split] += 1

    manifest = finalize_splits(
        samples,
        min_samples=args.min_samples_per_id,
        max_samples=args.max_samples_per_id,
        query_ratio=args.query_ratio,
    )
    validate_manifest(manifest)
    write_csv(manifest, output_dir / "manifest.csv")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "samples": len(manifest),
        "train": sum(row["split"] == "train" for row in manifest),
        "query": sum(row["split"] == "query" for row in manifest),
        "gallery": sum(row["split"] == "gallery" for row in manifest),
        "identities": len({row["identity_key"] for row in manifest}),
        "sequences": len({row["video_id"] for row in manifest}),
        "source": "SoccerNet-GSR ground truth",
        "stats": dict(stats),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def export_sequence(label_path, target_split, crops_dir, max_samples_per_id=0):
    import cv2

    payload = json.loads(label_path.read_text(encoding="utf-8"))
    images = index_images(payload.get("images") or [])
    categories = {
        category.get("id"): str(category.get("name") or "").lower()
        for category in payload.get("categories") or []
        if isinstance(category, dict)
    }
    sequence = label_path.parent.name
    image_dir = label_path.parent / "img1"
    rows = []
    stats = Counter()
    eligible = defaultdict(list)
    for annotation in payload.get("annotations") or []:
        if not isinstance(annotation, dict):
            continue
        image = images.get(str(annotation.get("image_id")))
        if image is None or not image.get("has_labeled_person", True):
            stats["missing_image_metadata"] += 1
            continue
        role = normalize_role((annotation.get("attributes") or {}).get("role") or categories.get(annotation.get("category_id")))
        if role != "player":
            stats[f"skip_role_{role or 'unknown'}"] += 1
            continue
        track_id = annotation.get("track_id")
        if track_id in (None, "", "None"):
            track_id = annotation.get("person_id")
        if track_id in (None, "", "None"):
            stats["missing_track_id"] += 1
            continue
        frame_name = image.get("file_name")
        bbox = normalize_bbox(annotation.get("bbox_image") or annotation.get("bbox") or {})
        if not frame_name or bbox is None:
            stats["invalid_crop"] += 1
            continue
        eligible[str(track_id)].append((frame_index(image, frame_name), annotation, image, bbox))

    selected = []
    for items in eligible.values():
        items.sort(key=lambda item: (item[0], str(item[1].get("id") or "")))
        if max_samples_per_id and len(items) > int(max_samples_per_id):
            indices = np.linspace(0, len(items) - 1, int(max_samples_per_id)).round().astype(int)
            items = [items[int(index)] for index in indices]
        selected.extend(items)

    for frame, annotation, image, bbox in selected:
        track_id = annotation.get("track_id")
        if track_id in (None, "", "None"):
            track_id = annotation.get("person_id")
        frame_name = image.get("file_name")
        source_path = image_dir / str(frame_name)
        crop = read_crop(source_path, bbox)
        if crop is None:
            stats["invalid_crop"] += 1
            continue
        identity_key = f"{sequence}:{track_id}"
        annotation_id = annotation.get("id") or f"{image.get('id')}_{track_id}"
        crop_path = crops_dir / target_split / sequence / f"{annotation_id}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not crop_path.exists() and not cv2.imwrite(str(crop_path), crop):
            stats["crop_write_failed"] += 1
            continue
        attributes = annotation.get("attributes") or {}
        rows.append(
            {
                "img_path": str(crop_path.resolve()),
                "source_split": target_split,
                "identity_key": identity_key,
                "pid": None,
                "camid": sequence,
                "video_id": sequence,
                "team": normalize_team(attributes.get("team") or attributes.get("side")),
                "role": "player",
                "jersey_number": attributes.get("jersey_number"),
                "frame": frame,
                "split": target_split,
                "label_source": "soccernet_gsr_ground_truth",
            }
        )
        stats["accepted"] += 1
    return rows, stats


def finalize_splits(rows, min_samples, max_samples, query_ratio):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source_split"], row["identity_key"])].append(row)
    selected = []
    for (source_split, _identity), items in sorted(grouped.items()):
        items.sort(key=lambda row: (int(row["frame"]), row["img_path"]))
        if len(items) < int(min_samples):
            continue
        if max_samples > 0 and len(items) > max_samples:
            indices = np.linspace(0, len(items) - 1, int(max_samples)).round().astype(int)
            items = [items[int(index)] for index in indices]
        if source_split == "train":
            for row in items:
                row["split"] = "train"
        else:
            query_count = max(1, int(round(len(items) * float(query_ratio))))
            query_indices = set(np.linspace(0, len(items) - 1, query_count).round().astype(int).tolist())
            for index, row in enumerate(items):
                row["split"] = "query" if index in query_indices else "gallery"
        selected.extend(items)
    # Torchreid builds the source classifier from the number of train IDs and
    # expects its train labels to be dense in [0, num_train_ids). Validation
    # identities are a separate namespace, shared only by query and gallery.
    for _namespace, namespace_rows in (
        ("train", [row for row in selected if row["split"] == "train"]),
        ("valid", [row for row in selected if row["split"] in {"query", "gallery"}]),
    ):
        pid_by_identity = {
            identity: index
            for index, identity in enumerate(sorted({row["identity_key"] for row in namespace_rows}))
        }
        camid_by_video = {
            video: index for index, video in enumerate(sorted({row["video_id"] for row in namespace_rows}))
        }
        team_values = sorted({(row["video_id"], row["team"]) for row in namespace_rows})
        team_by_value = {value: index for index, value in enumerate(team_values)}
        for row in namespace_rows:
            row["pid"] = pid_by_identity[row["identity_key"]]
            row["camid"] = camid_by_video[row["video_id"]]
            row["team_id"] = team_by_value[(row["video_id"], row["team"])]
            row["role_id"] = 3
            row["masks_path"] = ""
    return selected


def validate_manifest(rows):
    if not rows:
        raise RuntimeError("The PRTReID manifest is empty")
    paths = [row["img_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate crop paths found in PRTReID manifest")
    train_videos = {row["video_id"] for row in rows if row["split"] == "train"}
    valid_videos = {row["video_id"] for row in rows if row["split"] in {"query", "gallery"}}
    overlap = train_videos.intersection(valid_videos)
    if overlap:
        raise RuntimeError(f"Sequence leakage between train and validation: {sorted(overlap)}")
    query_ids = {row["pid"] for row in rows if row["split"] == "query"}
    gallery_ids = {row["pid"] for row in rows if row["split"] == "gallery"}
    if not query_ids or not query_ids.issubset(gallery_ids):
        raise RuntimeError("Every validation query identity must exist in gallery")


def index_images(images):
    output = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        for key in ("id", "image_id"):
            if key in image:
                output[str(image[key])] = image
    return output


def normalize_bbox(value):
    if isinstance(value, dict):
        return [float(value.get("x", 0)), float(value.get("y", 0)), float(value.get("w", 0)), float(value.get("h", 0))]
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return list(map(float, value[:4]))
    return None


def read_crop(path, bbox):
    import cv2

    if bbox is None or not path.exists():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    x, y, width, height = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image.shape[1], int(round(x + width)))
    y2 = min(image.shape[0], int(round(y + height)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def normalize_role(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {"field_player": "player", "outfield_player": "player", "players": "player"}.get(text, text or None)


def normalize_team(value):
    text = str(value or "unknown").strip().lower().replace("team_", "")
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return text


def frame_index(image, frame_name):
    for key in ("frame", "frame_id", "frame_index"):
        if image.get(key) is not None:
            return int(image[key])
    digits = "".join(character for character in Path(str(frame_name)).stem if character.isdigit())
    return int(digits or 0)


def infer_split(path):
    parts = [part.lower() for part in path.parts]
    for name in ("challenge", "test", "valid", "val", "train"):
        if name in parts:
            return name
    return "unknown"


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
