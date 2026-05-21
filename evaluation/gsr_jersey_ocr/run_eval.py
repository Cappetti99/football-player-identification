#!/usr/bin/env python3
"""Evaluate jersey OCR on SoccerNet-GSR ground-truth tracklets.

This script is deliberately narrower than the full football-tracking pipeline:
it uses GSR ground-truth boxes and track IDs, extracts a bounded crop set per
tracklet, runs the repository OCR module, and computes tracklet-level metrics.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ft.features.jersey_ocr import JerseyOCR, ocr_crop_score, parse_number


PLAYER_ROLES = {"player", "goalkeeper"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FT jersey OCR on SoccerNet-GSR annotations.")
    parser.add_argument("--gsr-dir", default="/media/data-lie/cappetti/dataset/SoccerNet-GSR")
    parser.add_argument("--output-dir", default="evaluation_outputs/gsr_jersey_ocr/default")
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-tracklets", type=int, default=None)
    parser.add_argument("--crops-per-tracklet", type=int, default=20)
    parser.add_argument("--min-tracklet-frames", type=int, default=3)
    parser.add_argument("--min-box-area", type=float, default=80.0)
    parser.add_argument("--crop-margin", type=float, default=0.03)
    parser.add_argument("--backend", default="easyocr")
    parser.add_argument("--easyocr-gpu", action="store_true")
    parser.add_argument("--template-matching", action="store_true")
    parser.add_argument("--template-font-image", default="docs/numberFont.jpg")
    parser.add_argument("--template-min-score", type=float, default=0.62)
    parser.add_argument("--template-weight", type=float, default=0.03)
    parser.add_argument("--template-max-candidates", type=int, default=4)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.25)
    parser.add_argument("--ocr-min-votes", type=int, default=2)
    parser.add_argument("--ocr-min-raw-confidence", type=float, default=0.05)
    parser.add_argument("--temporal-passes", type=int, default=1)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-crops", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log evaluation metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="football-tracking")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-init-timeout", type=int, default=180)
    parser.add_argument("--wandb-log-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-alert-on-finish", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-alert-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args, output_dir)

    try:
        label_files = discover_label_files(
            Path(args.gsr_dir),
            sequences=set(args.sequence),
            splits={split.lower() for split in args.split},
            max_sequences=args.max_sequences,
        )
        if not label_files:
            raise FileNotFoundError(f"No Labels-GameState.json files found under {args.gsr_dir}")

        load_stats = Counter()
        gt_tracklets = load_gt_tracklets(label_files, args, load_stats)
        if args.max_tracklets is not None:
            gt_tracklets = dict(list(sorted(gt_tracklets.items()))[: args.max_tracklets])
        if not gt_tracklets:
            write_json(output_dir / "load_stats.json", dict(load_stats))
            raise RuntimeError(
                "No GSR jersey-number tracklets passed the filters. "
                f"Debug counters written to {output_dir / 'load_stats.json'}: {dict(load_stats)}"
            )

        ocr_rows, tracklet_index = build_ocr_rows(gt_tracklets, output_dir / "crops", args)
        print(
            f"FT GSR OCR eval: sequences={len({key[0] for key in gt_tracklets})}"
            f" tracklets={len(gt_tracklets)} crops={len(ocr_rows)}",
            flush=True,
        )

        ocr = JerseyOCR(
            backend=args.backend,
            min_confidence=args.ocr_min_confidence,
            max_crops_per_tracklet=args.crops_per_tracklet,
            temporal_passes=args.temporal_passes,
            augment=args.augment,
            min_crop_quality=0.0,
            min_votes=args.ocr_min_votes,
            min_raw_confidence=args.ocr_min_raw_confidence,
            easyocr_gpu=args.easyocr_gpu,
            debug_dir=output_dir / "debug" if args.debug_crops else None,
            template_matching=args.template_matching,
            template_font_image=args.template_font_image,
            template_min_score=args.template_min_score,
            template_weight=args.template_weight,
            template_max_candidates=args.template_max_candidates,
            progress_every=10,
        )
        assignments, diagnostics = ocr.recognize(ocr_rows)
        predictions = build_predictions(gt_tracklets, tracklet_index, assignments)
        metrics = compute_metrics(predictions)
        threshold_sweep = compute_threshold_sweep(predictions)
        metrics["threshold_sweep_best_tracklet"] = best_sweep_row(
            threshold_sweep, ["accuracy_tracklet", "accuracy_assigned", "coverage"]
        )
        metrics["threshold_sweep_best_assigned"] = best_sweep_row(
            threshold_sweep, ["accuracy_assigned", "assigned_tracklets", "accuracy_tracklet"]
        )
        metrics["load_stats"] = dict(load_stats)

        write_json(output_dir / "config.json", vars(args))
        write_json(output_dir / "metrics.json", metrics)
        write_json(output_dir / "ocr_diagnostics.json", diagnostics)
        write_predictions(output_dir / "predictions.csv", predictions)
        write_confusion(output_dir / "confusion.csv", predictions)
        write_threshold_sweep(output_dir / "threshold_sweep.csv", threshold_sweep)

        log_wandb_success(wandb_run, args, output_dir, metrics)
        print_metrics(metrics)
        print(f"FT GSR OCR eval outputs: {output_dir}", flush=True)
    except Exception as exc:
        log_wandb_failure(wandb_run, args, exc)
        raise
    finally:
        finish_wandb(wandb_run)


def init_wandb(args, output_dir):
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"FT GSR OCR eval wandb: disabled, import failed: {type(exc).__name__}: {exc}", flush=True)
        return None

    name = args.wandb_name or Path(output_dir).name
    tags = ["gsr_jersey_ocr", *(args.wandb_tag or [])]
    try:
        run = wandb.init(
            project=args.wandb_project or "football-tracking",
            entity=args.wandb_entity or None,
            name=name,
            tags=tags,
            config=wandb_safe_config(vars(args)),
            settings=wandb.Settings(init_timeout=int(args.wandb_init_timeout)),
        )
    except Exception as exc:
        print(f"FT GSR OCR eval wandb: disabled, init failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    print(f"FT GSR OCR eval wandb: run={run.name} url={getattr(run, 'url', None)}", flush=True)
    return run


def log_wandb_success(run, args, output_dir, metrics):
    if run is None:
        return
    try:
        run.log(wandb_metric_payload(metrics))
        if args.wandb_log_artifacts:
            log_eval_artifact(run, output_dir)
        if args.wandb_alert_on_finish:
            import wandb

            wandb.alert(
                title="FT GSR OCR eval finished",
                text=(
                    f"{run.name} finished: "
                    f"coverage={metrics.get('coverage', 0.0):.3f}, "
                    f"accuracy_assigned={metrics.get('accuracy_assigned', 0.0):.3f}, "
                    f"accuracy_tracklet={metrics.get('accuracy_tracklet', 0.0):.3f}"
                ),
            )
    except Exception as exc:
        print(f"FT GSR OCR eval wandb: success log failed: {type(exc).__name__}: {exc}", flush=True)


def log_wandb_failure(run, args, exc):
    if run is None:
        return
    try:
        message = f"{type(exc).__name__}: {exc}"
        run.log({"status_code": -1, "error": message})
        if args.wandb_alert_on_failure:
            import wandb

            wandb.alert(title="FT GSR OCR eval failed", text=f"{run.name} failed: {message}")
    except Exception as log_exc:
        print(f"FT GSR OCR eval wandb: failure log failed: {type(log_exc).__name__}: {log_exc}", flush=True)


def finish_wandb(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"FT GSR OCR eval wandb: finish failed: {type(exc).__name__}: {exc}", flush=True)


def wandb_metric_payload(metrics):
    payload = {
        "status_code": 1,
        "tracklets": metrics.get("tracklets", 0),
        "assigned_tracklets": metrics.get("assigned_tracklets", 0),
        "correct_assigned_tracklets": metrics.get("correct_assigned_tracklets", 0),
        "coverage": metrics.get("coverage", 0.0),
        "accuracy_assigned": metrics.get("accuracy_assigned", 0.0),
        "accuracy_tracklet": metrics.get("accuracy_tracklet", 0.0),
    }
    for label in ("threshold_sweep_best_tracklet", "threshold_sweep_best_assigned"):
        row = metrics.get(label) or {}
        prefix = label.replace("threshold_sweep_", "")
        for key, value in row.items():
            payload[f"{prefix}/{key}"] = value
    return payload


def log_eval_artifact(run, output_dir):
    import wandb

    output_dir = Path(output_dir)
    artifact = wandb.Artifact(f"{run.name}-gsr-jersey-ocr-eval", type="gsr-jersey-ocr-eval")
    for name in (
        "config.json",
        "metrics.json",
        "predictions.csv",
        "confusion.csv",
        "threshold_sweep.csv",
        "ocr_diagnostics.json",
        "load_stats.json",
    ):
        path = output_dir / name
        if path.exists():
            artifact.add_file(str(path), name=name)
    run.log_artifact(artifact)


def wandb_safe_config(config):
    clean = {}
    for key, value in config.items():
        if key.lower() in {"api_key", "token", "password"}:
            clean[key] = "***"
        elif isinstance(value, Path):
            clean[key] = str(value)
        else:
            clean[key] = value
    return clean


def discover_label_files(gsr_dir, sequences, splits, max_sequences=None):
    paths = []
    for label_path in sorted(gsr_dir.rglob("Labels-GameState.json")):
        sequence = label_path.parent.name
        split = infer_split(label_path)
        if sequences and sequence not in sequences:
            continue
        if splits and split.lower() not in splits:
            continue
        paths.append(label_path)
    if max_sequences is not None:
        paths = paths[: max(0, int(max_sequences))]
    return paths


def load_gt_tracklets(label_files, args, stats):
    tracklets = defaultdict(list)
    for label_path in label_files:
        stats["label_files"] += 1
        payload = read_json(label_path)
        images = index_images(payload.get("images") or [])
        categories = {
            item.get("id"): str(item.get("name") or "").lower()
            for item in payload.get("categories") or []
            if isinstance(item, dict)
        }
        split = infer_split(label_path)
        sequence = label_path.parent.name
        image_dir = label_path.parent / "img1"
        for annotation in payload.get("annotations") or []:
            stats["annotations"] += 1
            if not isinstance(annotation, dict):
                stats["skip_non_dict_annotation"] += 1
                continue
            image = images.get(str(annotation.get("image_id")))
            if image is None or not image.get("has_labeled_person", True):
                stats["skip_missing_or_unlabeled_image"] += 1
                continue
            role = resolve_role(annotation, categories)
            stats[f"role_{role or 'missing'}"] += 1
            if role not in PLAYER_ROLES:
                stats["skip_role"] += 1
                continue
            gt_jersey = jersey_number_from_attributes(annotation.get("attributes") or {})
            if gt_jersey is None:
                stats["skip_missing_jersey_number"] += 1
                continue
            bbox = normalize_bbox(annotation.get("bbox_image") or annotation.get("bbox") or {})
            if bbox_area(bbox) < float(args.min_box_area):
                stats["skip_small_bbox"] += 1
                continue
            track_id = annotation.get("track_id") or annotation.get("person_id")
            frame_name = image.get("file_name")
            if track_id in (None, "") or not frame_name:
                stats["skip_missing_track_or_frame"] += 1
                continue
            image_path = image_dir / frame_name
            if not image_path.exists():
                stats["skip_missing_image_file"] += 1
                continue
            stats["accepted_annotations"] += 1
            row = {
                "sequence": sequence,
                "split": split,
                "gt_track_id": str(track_id),
                "frame_name": frame_name,
                "frame_index": frame_sort_key(image, frame_name),
                "image_path": image_path,
                "bbox": bbox_xyxy(bbox),
                "gt_jersey_number": int(gt_jersey),
                "role": role,
                "team": (annotation.get("attributes") or {}).get("team")
                or (annotation.get("attributes") or {}).get("side"),
            }
            tracklets[(sequence, str(track_id))].append(row)

    filtered = {}
    for key, rows in tracklets.items():
        if len(rows) < int(args.min_tracklet_frames):
            stats["skip_short_tracklet"] += 1
            continue
        jersey_counts = Counter(row["gt_jersey_number"] for row in rows)
        gt_jersey, _ = jersey_counts.most_common(1)[0]
        for row in rows:
            row["gt_jersey_number"] = int(gt_jersey)
        filtered[key] = sorted(rows, key=lambda row: row["frame_index"])
    stats["accepted_tracklets"] = len(filtered)
    return filtered


def build_ocr_rows(gt_tracklets, crop_dir, args):
    import cv2

    crop_dir.mkdir(parents=True, exist_ok=True)
    ocr_rows = []
    tracklet_index = {}
    for eval_id, ((sequence, gt_track_id), rows) in enumerate(sorted(gt_tracklets.items()), start=1):
        tracklet_index[eval_id] = {
            "sequence": sequence,
            "gt_track_id": gt_track_id,
            "gt_jersey_number": rows[0]["gt_jersey_number"],
            "role": rows[0]["role"],
            "team": rows[0].get("team"),
            "num_gt_frames": len(rows),
        }
        for row in sample_tracklet_rows(rows, args.crops_per_tracklet):
            image = cv2.imread(str(row["image_path"]))
            if image is None or image.size == 0:
                continue
            crop = crop_bbox(image, row["bbox"], margin=args.crop_margin)
            if crop is None or crop.size == 0:
                continue
            out_dir = crop_dir / safe_name(sequence)
            out_dir.mkdir(parents=True, exist_ok=True)
            crop_path = out_dir / f"track_{eval_id:06d}_gt_{safe_name(gt_track_id)}_frame_{safe_name(Path(row['frame_name']).stem)}.jpg"
            cv2.imwrite(str(crop_path), crop)
            ocr_rows.append(
                {
                    "track_id": eval_id,
                    "raw_track_id": eval_id,
                    "display_track_id": eval_id,
                    "frame": int(row["frame_index"]),
                    "crop_path": str(crop_path),
                    "crop_quality": crop_quality(row["bbox"], image.shape[1], image.shape[0]),
                    "bbox": row["bbox"],
                    "role_detection": row["role"],
                    "semantic_group_id": 3 if row["role"] == "goalkeeper" else 1,
                    "sequence": sequence,
                    "gt_track_id": gt_track_id,
                    "gt_jersey_number": row["gt_jersey_number"],
                }
            )
    return ocr_rows, tracklet_index


def sample_tracklet_rows(rows, max_count):
    max_count = int(max_count)
    if max_count <= 0 or len(rows) <= max_count:
        return list(rows)
    ordered = sorted(rows, key=lambda row: row["frame_index"])
    selected_indexes = {
        min(len(ordered) - 1, round(index * (len(ordered) - 1) / max(1, max_count - 1)))
        for index in range(max_count)
    }
    sampled = [ordered[index] for index in sorted(selected_indexes)]
    if len(sampled) < max_count:
        existing = {row["frame_index"] for row in sampled}
        by_size = sorted(ordered, key=lambda row: bbox_area_xyxy(row["bbox"]), reverse=True)
        for row in by_size:
            if row["frame_index"] in existing:
                continue
            sampled.append(row)
            existing.add(row["frame_index"])
            if len(sampled) >= max_count:
                break
    return sorted(sampled[:max_count], key=lambda row: row["frame_index"])


def build_predictions(gt_tracklets, tracklet_index, assignments):
    predictions = []
    for eval_id, meta in sorted(tracklet_index.items()):
        assignment = assignments.get(eval_id)
        pred_number = assignment.get("jersey_number") if assignment else None
        gt_number = int(meta["gt_jersey_number"])
        predictions.append(
            {
                "eval_track_id": eval_id,
                "sequence": meta["sequence"],
                "gt_track_id": meta["gt_track_id"],
                "role": meta.get("role"),
                "team": meta.get("team"),
                "num_gt_frames": int(meta["num_gt_frames"]),
                "gt_jersey_number": gt_number,
                "pred_jersey_number": pred_number,
                "assigned": pred_number is not None,
                "correct": pred_number is not None and int(pred_number) == gt_number,
                "confidence": assignment.get("confidence") if assignment else None,
                "head_confidence": assignment.get("head_confidence") if assignment else None,
                "winner_margin": assignment.get("winner_margin") if assignment else None,
                "votes": assignment.get("votes") if assignment else 0,
                "total_detections": assignment.get("total_detections") if assignment else 0,
                "candidates": json.dumps(assignment.get("candidates", []), ensure_ascii=False) if assignment else "[]",
            }
        )
    return predictions


def compute_metrics(predictions):
    total = len(predictions)
    assigned = [row for row in predictions if row["assigned"]]
    correct = [row for row in assigned if row["correct"]]
    return {
        "tracklets": int(total),
        "assigned_tracklets": int(len(assigned)),
        "correct_assigned_tracklets": int(len(correct)),
        "coverage": safe_div(len(assigned), total),
        "accuracy_assigned": safe_div(len(correct), len(assigned)),
        "accuracy_tracklet": safe_div(len(correct), total),
        "by_role": metrics_by_key(predictions, "role"),
        "by_team": metrics_by_key(predictions, "team"),
    }


def metrics_by_key(predictions, key):
    groups = defaultdict(list)
    for row in predictions:
        groups[str(row.get(key) or "unknown")].append(row)
    return {name: compute_metrics_without_groups(rows) for name, rows in sorted(groups.items())}


def compute_metrics_without_groups(rows):
    total = len(rows)
    assigned = [row for row in rows if row["assigned"]]
    correct = [row for row in assigned if row["correct"]]
    return {
        "tracklets": int(total),
        "assigned_tracklets": int(len(assigned)),
        "correct_assigned_tracklets": int(len(correct)),
        "coverage": safe_div(len(assigned), total),
        "accuracy_assigned": safe_div(len(correct), len(assigned)),
        "accuracy_tracklet": safe_div(len(correct), total),
    }


def compute_threshold_sweep(predictions):
    rows = []
    for confidence in [0.20, 0.30, 0.40, 0.50, 0.60]:
        for head_confidence in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            for winner_margin in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]:
                for votes in [2, 3, 4, 5, 8, 10]:
                    selected = [
                        row
                        for row in predictions
                        if row["assigned"]
                        and safe_float(row.get("confidence")) >= confidence
                        and safe_float(row.get("head_confidence")) >= head_confidence
                        and safe_float(row.get("winner_margin")) >= winner_margin
                        and safe_int(row.get("votes")) >= votes
                    ]
                    if not selected:
                        continue
                    correct = [row for row in selected if row["correct"]]
                    rows.append(
                        {
                            "confidence": confidence,
                            "head_confidence": head_confidence,
                            "winner_margin": winner_margin,
                            "votes": votes,
                            "assigned_tracklets": int(len(selected)),
                            "coverage": safe_div(len(selected), len(predictions)),
                            "accuracy_assigned": safe_div(len(correct), len(selected)),
                            "accuracy_tracklet": safe_div(len(correct), len(predictions)),
                        }
                    )
    return rows


def best_sweep_row(rows, sort_keys):
    if not rows:
        return None
    return dict(
        max(rows, key=lambda row: tuple(float(row.get(key, 0.0) or 0.0) for key in sort_keys))
    )


def write_predictions(path, predictions):
    fields = [
        "eval_track_id",
        "sequence",
        "gt_track_id",
        "role",
        "team",
        "num_gt_frames",
        "gt_jersey_number",
        "pred_jersey_number",
        "assigned",
        "correct",
        "confidence",
        "head_confidence",
        "winner_margin",
        "votes",
        "total_detections",
        "candidates",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def write_confusion(path, predictions):
    rows = [
        row
        for row in predictions
        if row["assigned"] and not row["correct"]
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gt_jersey_number",
                "pred_jersey_number",
                "count",
                "examples",
            ],
        )
        writer.writeheader()
        counts = defaultdict(list)
        for row in rows:
            counts[(row["gt_jersey_number"], row["pred_jersey_number"])].append(
                f"{row['sequence']}:{row['gt_track_id']}"
            )
        for (gt, pred), examples in sorted(counts.items(), key=lambda item: (-len(item[1]), item[0])):
            writer.writerow(
                {
                    "gt_jersey_number": gt,
                    "pred_jersey_number": pred,
                    "count": len(examples),
                    "examples": ";".join(examples[:10]),
                }
            )


def write_threshold_sweep(path, rows):
    fields = [
        "confidence",
        "head_confidence",
        "winner_margin",
        "votes",
        "assigned_tracklets",
        "coverage",
        "accuracy_assigned",
        "accuracy_tracklet",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            row["accuracy_tracklet"],
            row["accuracy_assigned"],
            row["coverage"],
        ),
        reverse=True,
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


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
    role = normalize_role(attributes.get("role") or attributes.get("class") or attributes.get("category"))
    if role:
        return role
    category_id = annotation.get("category_id")
    if category_id in categories_by_id:
        return normalize_role(categories_by_id[category_id])
    return None


def normalize_role(value):
    if value is None:
        return None
    role = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "field_player": "player",
        "outfield_player": "player",
        "players": "player",
        "keeper": "goalkeeper",
        "goal_keeper": "goalkeeper",
        "gk": "goalkeeper",
        "ref": "referee",
        "official": "referee",
        "match_official": "referee",
    }
    return aliases.get(role, role)


def jersey_number_from_attributes(attributes):
    if not isinstance(attributes, dict):
        return None
    direct_keys = (
        "jersey_number",
        "jersey",
        "shirt_number",
        "shirt",
        "number",
        "bib_number",
    )
    for key in direct_keys:
        if key in attributes:
            parsed = parse_number(attributes.get(key))
            if parsed is not None:
                return parsed
    for key, value in attributes.items():
        key_text = str(key).lower()
        if any(token in key_text for token in ("jersey", "shirt", "bib")):
            parsed = parse_number(value)
            if parsed is not None:
                return parsed
    return None


def infer_split(path):
    parts = [part.lower() for part in Path(path).parts]
    for split in ("train", "valid", "val", "test", "challenge"):
        if split in parts:
            return "val" if split == "valid" else split
    return "unknown"


def frame_sort_key(image, frame_name):
    for key in ("frame", "frame_id", "number"):
        value = image.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    match = re.search(r"(\d+)", Path(frame_name).stem)
    return int(match.group(1)) if match else 0


def bbox_xyxy(bbox):
    bbox = normalize_bbox(bbox)
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("w", 0.0))
    h = float(bbox.get("h", 0.0))
    return [x, y, x + w, y + h]


def bbox_area(bbox):
    bbox = normalize_bbox(bbox)
    if not bbox:
        return 0.0
    return max(0.0, float(bbox.get("w", 0.0))) * max(0.0, float(bbox.get("h", 0.0)))


def normalize_bbox(bbox):
    if isinstance(bbox, dict):
        if {"x", "y", "w", "h"}.issubset(bbox):
            return bbox
        if {"x1", "y1", "x2", "y2"}.issubset(bbox):
            x1 = float(bbox["x1"])
            y1 = float(bbox["y1"])
            x2 = float(bbox["x2"])
            y2 = float(bbox["y2"])
            return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = [float(value) for value in bbox]
        return {"x": x, "y": y, "w": w, "h": h}
    return {}


def bbox_area_xyxy(bbox):
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def crop_quality(bbox, image_width, image_height):
    row = {"bbox": bbox}
    quality = ocr_crop_score(row)
    area_fraction = bbox_area_xyxy(bbox) / max(1.0, float(image_width * image_height))
    return float(max(quality, min(1.0, area_fraction * 25.0)))


def crop_bbox(image, bbox, margin=0.03):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * float(margin)
    pad_y = box_h * float(margin)
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def safe_div(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)
        f.write("\n")


def json_default(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def print_metrics(metrics):
    print(
        "FT GSR OCR eval:"
        f" tracklets={metrics['tracklets']}"
        f" assigned={metrics['assigned_tracklets']}"
        f" coverage={metrics['coverage']:.3f}"
        f" accuracy_assigned={metrics['accuracy_assigned']:.3f}"
        f" accuracy_tracklet={metrics['accuracy_tracklet']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
