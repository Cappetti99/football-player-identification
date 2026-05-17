"""Prepare and train YOLO variants from SoccerNet-GSR.

The thesis pipeline benefits more from stable person crops than from forcing
all semantic roles into the detector. This script can create and train:

- person_ball: person = player + goalkeeper + referee, plus ball
- person_only: person = player + goalkeeper + referee
- four_class: ball, goalkeeper, player, referee

It also exports a manifest with SoccerNet-GSR attributes so role/team/jersey
metadata can be reused later for OCR and identity evaluation.
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = {
    "person_ball": ["person", "ball"],
    "person_only": ["person"],
    "four_class": ["ball", "goalkeeper", "player", "referee"],
}

PERSON_ROLES = {"player", "goalkeeper", "referee"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SoccerNet-GSR and train YOLO.")
    parser.add_argument("--gsr-dir", default="/media/data-lie/cappetti/dataset/SoccerNet-GSR")
    parser.add_argument("--output-dir", default="/media/data-lie/cappetti/dataset/soccernet_gsr_yolo_person_ball")
    parser.add_argument("--mode", choices=sorted(CLASS_NAMES), default="person_ball")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--include-other-as-person", action="store_true")
    parser.add_argument("--min-box-area", type=float, default=4.0)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--base-model", default="yolo26x.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", default="runs/ft_yolo_gsr")
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    stats = prepare_dataset(args, output_dir)
    print_summary(args, stats, output_dir)
    if args.prepare_only:
        return
    train_yolo(args, output_dir / "data.yaml")


def prepare_dataset(args, output_dir):
    gsr_dir = Path(args.gsr_dir)
    if not gsr_dir.exists():
        raise FileNotFoundError(gsr_dir)

    label_paths = discover_label_files(
        gsr_dir,
        sequences=set(args.sequence),
        splits={split.lower() for split in args.split},
    )
    if not label_paths:
        raise FileNotFoundError(f"No Labels-GameState.json files found under {gsr_dir}")

    class_names = CLASS_NAMES[args.mode]
    # The same SoccerNet-GSR annotations can be collapsed into different class
    # spaces. For the thesis runs, person_ball keeps detection robust and leaves
    # role recovery to the semantic pipeline.
    stats = {
        "mode": args.mode,
        "class_names": class_names,
        "frames": Counter(),
        "labels": Counter(),
        "roles": Counter(),
        "skipped_roles": Counter(),
        "sequences": 0,
    }
    manifest_rows = []
    for label_path in label_paths:
        exported, rows = export_sequence(args, label_path, output_dir, class_names)
        if exported:
            stats["sequences"] += 1
            stats["frames"][infer_split(label_path)] += exported
            manifest_rows.extend(rows)
            for row in rows:
                if row.get("skipped_role"):
                    stats["skipped_roles"][row["skipped_role"]] += 1
                    continue
                stats["labels"][row["class_name"]] += 1
                stats["roles"][row.get("role") or "unknown"] += 1

    write_dataset_yaml(output_dir, class_names)
    write_jsonl(output_dir / "gsr_manifest.jsonl", manifest_rows)
    write_json(output_dir / "conversion_summary.json", serializable_stats(stats))
    return stats


def discover_label_files(gsr_dir, sequences, splits):
    paths = []
    for label_path in sorted(gsr_dir.rglob("Labels-GameState.json")):
        sequence = label_path.parent.name
        split = infer_split(label_path)
        if sequences and sequence not in sequences:
            continue
        if splits and split.lower() not in splits:
            continue
        paths.append(label_path)
    return paths


def export_sequence(args, label_path, output_dir, class_names):
    payload = read_json(label_path)
    images_by_id = index_images(payload.get("images") or [])
    categories_by_id = {
        category.get("id"): (category.get("name") or "").lower()
        for category in payload.get("categories") or []
        if isinstance(category, dict)
    }
    labels_by_frame = defaultdict(list)
    manifest_by_frame = defaultdict(list)
    skipped_rows = []

    for annotation in payload.get("annotations") or []:
        if not isinstance(annotation, dict):
            continue
        image = images_by_id.get(str(annotation.get("image_id")))
        if image is None or not image.get("has_labeled_person", True):
            continue
        role = resolve_role(annotation, categories_by_id)
        class_id = map_role_to_class_id(role, args.mode, include_other=args.include_other_as_person)
        if class_id is None:
            # Keep skipped-role counts in the summary so dataset conversions are
            # auditable when changing class mappings.
            skipped_rows.append({"skipped_role": role or "unknown"})
            continue

        bbox = annotation.get("bbox_image") or {}
        yolo_bbox = bbox_to_yolo(bbox, int(image.get("width", 0)), int(image.get("height", 0)), args.min_box_area)
        if yolo_bbox is None:
            continue
        frame_name = image.get("file_name")
        if not frame_name:
            continue

        class_name = class_names[class_id]
        labels_by_frame[frame_name].append((class_id, *yolo_bbox))
        manifest_by_frame[frame_name].append(
            manifest_row(label_path, frame_name, image, annotation, role, class_id, class_name)
        )

    frame_names = sorted(labels_by_frame)
    if args.max_frames_per_sequence is not None:
        frame_names = frame_names[: args.max_frames_per_sequence]

    split = infer_split(label_path)
    image_dir = label_path.parent / "img1"
    if not image_dir.exists():
        raise FileNotFoundError(f"img1 folder not found next to {label_path}")

    exported = 0
    manifest_rows = []
    for frame_name in frame_names:
        source_image = image_dir / frame_name
        if not source_image.exists():
            continue
        stem = f"{label_path.parent.name}_{Path(frame_name).stem}"
        image_out = output_dir / split / "images" / f"{stem}{source_image.suffix.lower()}"
        label_out = output_dir / split / "labels" / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(source_image, image_out, copy_images=args.copy_images)
        write_labels(label_out, labels_by_frame[frame_name])
        manifest_rows.extend(manifest_by_frame[frame_name])
        exported += 1

    manifest_rows.extend(skipped_rows)
    return exported, manifest_rows


def index_images(images):
    by_id = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        for key in ("id", "image_id"):
            if key in image:
                by_id[str(image[key])] = image
    return by_id


def resolve_role(annotation, categories_by_id):
    attributes = annotation.get("attributes") or {}
    role = attributes.get("role")
    if role:
        return str(role).lower()
    category_id = annotation.get("category_id")
    if category_id in categories_by_id:
        return categories_by_id[category_id]
    return None


def map_role_to_class_id(role, mode, include_other=False):
    if role is None:
        return None
    role = str(role).lower()
    if mode == "person_ball":
        if role in PERSON_ROLES or (include_other and role == "other"):
            return 0
        if role == "ball":
            return 1
        return None
    if mode == "person_only":
        if role in PERSON_ROLES or (include_other and role == "other"):
            return 0
        return None
    if mode == "four_class":
        names = CLASS_NAMES[mode]
        return names.index(role) if role in names else None
    return None


def bbox_to_yolo(bbox, image_width, image_height, min_box_area):
    if not bbox or image_width <= 0 or image_height <= 0:
        return None
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("w", 0.0))
    h = float(bbox.get("h", 0.0))
    if w <= 0 or h <= 0 or w * h < float(min_box_area):
        return None
    values = [
        (x + w / 2.0) / image_width,
        (y + h / 2.0) / image_height,
        w / image_width,
        h / image_height,
    ]
    values = [min(max(value, 0.0), 1.0) for value in values]
    if values[2] <= 0.0 or values[3] <= 0.0:
        return None
    return values


def manifest_row(label_path, frame_name, image, annotation, role, class_id, class_name):
    attributes = annotation.get("attributes") or {}
    # The manifest is not consumed by YOLO. It keeps semantic labels available
    # for later OCR/ReID/evaluation experiments without re-parsing GSR JSON.
    return {
        "sequence": label_path.parent.name,
        "split": infer_split(label_path),
        "frame": frame_name,
        "image_id": image.get("id", image.get("image_id")),
        "annotation_id": annotation.get("id"),
        "track_id": annotation.get("track_id") or annotation.get("person_id"),
        "class_id": int(class_id),
        "class_name": class_name,
        "role": role,
        "team": attributes.get("team") or attributes.get("side"),
        "jersey_number": attributes.get("jersey_number"),
        "bbox_image": annotation.get("bbox_image"),
        "bbox_pitch": annotation.get("bbox_pitch"),
        "attributes": attributes,
    }


def write_dataset_yaml(output_dir, class_names):
    import yaml

    payload = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": class_names,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def train_yolo(args, data_yaml):
    from ultralytics import YOLO

    name = args.name or f"{Path(args.base_model).stem}_{args.mode}"
    model = YOLO(args.base_model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=name,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
    )


def write_labels(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row[0]} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}\n")


def link_or_copy(source, destination, copy_images):
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
    else:
        # Symlinks keep the prepared dataset small; pass --copy-images only when
        # the training machine cannot see the original SoccerNet-GSR tree.
        destination.symlink_to(source.resolve())


def infer_split(label_path):
    lowered = str(label_path).lower()
    if "/train/" in lowered:
        return "train"
    if "/valid/" in lowered or "/val/" in lowered:
        return "val"
    if "/test/" in lowered:
        return "val"
    if "/challenge/" in lowered:
        return "test"
    return "train"


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.get("skipped_role"):
                continue
            f.write(json.dumps(row) + "\n")


def serializable_stats(stats):
    return {
        "mode": stats["mode"],
        "class_names": stats["class_names"],
        "sequences": stats["sequences"],
        "frames": dict(stats["frames"]),
        "labels": dict(stats["labels"]),
        "roles": dict(stats["roles"]),
        "skipped_roles": dict(stats["skipped_roles"]),
    }


def print_summary(args, stats, output_dir):
    print(f"FT GSR dataset: {output_dir}")
    print(f"mode: {args.mode}")
    print(f"classes: {CLASS_NAMES[args.mode]}")
    print(f"sequences: {stats['sequences']}")
    print(f"frames: {dict(stats['frames'])}")
    print(f"labels: {dict(stats['labels'])}")
    print(f"roles: {dict(stats['roles'])}")
    if stats["skipped_roles"]:
        print(f"skipped_roles: {dict(stats['skipped_roles'])}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
