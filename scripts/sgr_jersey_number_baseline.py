#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROI_REGIONS = {
    "full_body": (0.00, 1.00, 0.00, 1.00),
    "upper_back": (0.18, 0.55, 0.20, 0.80),
    "center_back": (0.20, 0.50, 0.25, 0.75),
    "torso": (0.12, 0.62, 0.15, 0.85),
    "number_band": (0.24, 0.48, 0.18, 0.82),
}


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a lightweight supervised jersey-number baseline on SoccerNet-GSR crops."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gt-jsonl", required=True, type=Path)
    parser.add_argument("--train-sequence", action="append", required=True)
    parser.add_argument("--eval-sequence", action="append")
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--roi", choices=sorted(ROI_REGIONS), default="torso")
    parser.add_argument("--size", default="32x64", help="Feature image size WxH")
    parser.add_argument("--method", choices=["centroid", "knn"], default="centroid")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--same-sequence-split", choices=["none", "even-odd", "first-second"], default="none")
    parser.add_argument(
        "--restrict-to-eval-labels",
        action="store_true",
        help="Restrict predictions to jersey numbers present in the eval records. This approximates roster-constrained inference.",
    )
    parser.add_argument(
        "--allowed-label",
        action="append",
        type=int,
        default=[],
        help="Allowed predicted jersey number. Can be passed multiple times.",
    )
    parser.add_argument("--max-train-per-class", type=int, default=1200)
    parser.add_argument("--max-eval-per-class", type=int, default=2000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    import cv2

    train_sequences = set(args.train_sequence)
    eval_sequences = set(args.eval_sequence or args.train_sequence)
    width, height = parse_size(args.size)
    train_records = load_records(args.gt_jsonl, train_sequences)
    eval_records = load_records(args.gt_jsonl, eval_sequences)
    if train_sequences == eval_sequences and args.same_sequence_split != "none":
        train_records, eval_records = split_same_sequence_records(train_records, args.same_sequence_split)

    train_features, train_labels, train_meta = build_features(
        cv2,
        args.dataset_root,
        train_records,
        args.roi,
        width,
        height,
        split_override=args.train_split,
        max_per_class=args.max_train_per_class,
    )
    eval_features, eval_labels, eval_meta = build_features(
        cv2,
        args.dataset_root,
        eval_records,
        args.roi,
        width,
        height,
        split_override=args.eval_split,
        max_per_class=args.max_eval_per_class,
    )
    allowed_labels = resolve_allowed_labels(args, eval_labels)

    if args.method == "centroid":
        centroids = train_centroids(train_features, train_labels)
        predictions, scores = predict_centroids(eval_features, centroids, allowed_labels=allowed_labels)
    else:
        predictions, scores = predict_knn(
            eval_features,
            train_features,
            train_labels,
            k=args.knn_k,
            allowed_labels=allowed_labels,
        )
    result = evaluate(predictions, eval_labels, scores=scores)
    result.update(
        {
            "method": args.method,
            "knn_k": args.knn_k if args.method == "knn" else None,
            "same_sequence_split": args.same_sequence_split,
            "allowed_labels": sorted(allowed_labels) if allowed_labels else None,
            "train_sequences": sorted(train_sequences),
            "eval_sequences": sorted(eval_sequences),
            "roi": args.roi,
            "size": {"width": width, "height": height},
            "train_samples": len(train_labels),
            "eval_samples": len(eval_labels),
            "train_class_counts": Counter(map(str, train_labels)).most_common(),
            "eval_class_counts": Counter(map(str, eval_labels)).most_common(),
            "missing_train_images": train_meta["missing_images"],
            "missing_eval_images": eval_meta["missing_images"],
        }
    )
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_records(path, sequences):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sequence") not in sequences:
                continue
            if str(row.get("role") or "").lower() not in {"player", "goalkeeper"}:
                continue
            jersey = jersey_number(row)
            if jersey is None:
                continue
            rows.append(row)
    return rows


def resolve_allowed_labels(args, eval_labels):
    labels = set(int(item) for item in args.allowed_label)
    if args.restrict_to_eval_labels:
        labels.update(int(item) for item in sorted(set(eval_labels.tolist())))
    return labels or None


def split_same_sequence_records(records, mode):
    ordered = sorted(records, key=lambda row: (str(row.get("sequence")), int(row.get("frame", 0)), str(row.get("track_id", ""))))
    if mode == "even-odd":
        train = [row for row in ordered if int(row.get("frame", 0)) % 2 == 0]
        eval_rows = [row for row in ordered if int(row.get("frame", 0)) % 2 == 1]
        return train, eval_rows
    if mode == "first-second":
        by_sequence = defaultdict(list)
        for row in ordered:
            by_sequence[str(row.get("sequence"))].append(row)
        train = []
        eval_rows = []
        for seq_rows in by_sequence.values():
            frames = sorted({int(row.get("frame", 0)) for row in seq_rows})
            if not frames:
                continue
            midpoint = frames[len(frames) // 2]
            train.extend(row for row in seq_rows if int(row.get("frame", 0)) <= midpoint)
            eval_rows.extend(row for row in seq_rows if int(row.get("frame", 0)) > midpoint)
        return train, eval_rows
    return records, records


def build_features(cv2, dataset_root, records, roi_name, width, height, split_override=None, max_per_class=1200):
    features = []
    labels = []
    counts = Counter()
    missing_images = 0
    for row in records:
        label = jersey_number(row)
        if label is None:
            continue
        if max_per_class and counts[label] >= max_per_class:
            continue
        split = split_override or row.get("split")
        if not split:
            raise ValueError("Record has no split; pass --train-split/--eval-split")
        image_path = resolve_frame_path(dataset_root, split, row["sequence"], int(row["frame"]))
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images += 1
            continue
        crop = crop_bbox(image, row.get("bbox_image") or row.get("bbox"))
        crop = crop_roi(crop, ROI_REGIONS[roi_name])
        if crop is None:
            continue
        features.append(extract_feature(cv2, crop, width, height))
        labels.append(label)
        counts[label] += 1
    if not features:
        raise RuntimeError("No features extracted")
    return np.vstack(features), np.asarray(labels, dtype=np.int32), {"missing_images": missing_images}


def resolve_frame_path(dataset_root, split, sequence, frame):
    image_dir = resolve_sequence_image_dir(dataset_root, split, sequence)
    candidates = [
        image_dir / f"{frame:06d}.jpg",
        image_dir / f"{frame:06d}.png",
        image_dir / f"{frame:08d}.jpg",
        image_dir / f"frame_{frame:06d}.jpg",
        image_dir / f"{frame}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_sequence_image_dir(dataset_root, split, sequence):
    split = str(split)
    split_candidates = [split]
    aliases = {
        "val": ["valid", "validation"],
        "valid": ["val", "validation"],
        "validation": ["valid", "val"],
    }
    split_candidates.extend(item for item in aliases.get(split.lower(), []) if item not in split_candidates)
    for candidate_split in split_candidates:
        image_dir = Path(dataset_root) / candidate_split / sequence / "img1"
        if image_dir.exists():
            return image_dir
    return Path(dataset_root) / split / sequence / "img1"


def crop_bbox(image, bbox):
    if isinstance(bbox, str):
        bbox = json.loads(bbox)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def crop_roi(crop, region):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    top_r, bottom_r, left_r, right_r = region
    top = int(h * top_r)
    bottom = max(top + 1, int(h * bottom_r))
    left = int(w * left_r)
    right = max(left + 1, int(w * right_r))
    roi = crop[top:bottom, left:right]
    return roi if roi.size else None


def extract_feature(cv2, crop, width, height):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    feature = np.concatenate([gray.astype(np.float32).reshape(-1), magnitude.reshape(-1)], axis=0)
    feature -= feature.mean()
    std = feature.std()
    if std > 1e-6:
        feature /= std
    return feature[None, :]


def train_centroids(features, labels):
    centroids = {}
    for label in sorted(set(labels.tolist())):
        class_features = features[labels == label]
        centroid = class_features.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[int(label)] = centroid / norm if norm > 1e-6 else centroid
    return centroids


def predict_centroids(features, centroids, allowed_labels=None):
    labels = np.asarray(
        [label for label in sorted(centroids) if allowed_labels is None or int(label) in allowed_labels],
        dtype=np.int32,
    )
    if labels.size == 0:
        raise ValueError("No centroid labels remain after applying allowed label filter")
    matrix = np.vstack([centroids[int(label)] for label in labels])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-6)
    scores = normalized @ matrix.T
    order = np.argsort(-scores, axis=1)
    predictions = labels[order[:, 0]]
    top1 = scores[np.arange(scores.shape[0]), order[:, 0]]
    top2 = scores[np.arange(scores.shape[0]), order[:, 1]] if scores.shape[1] > 1 else np.zeros_like(top1)
    return predictions, {"top1_score": top1, "top2_score": top2, "margin": top1 - top2}


def predict_knn(eval_features, train_features, train_labels, k=5, chunk_size=512, allowed_labels=None):
    if allowed_labels is not None:
        mask = np.asarray([int(label) in allowed_labels for label in train_labels], dtype=bool)
        train_features = train_features[mask]
        train_labels = train_labels[mask]
        if len(train_labels) == 0:
            raise ValueError("No training samples remain after applying allowed label filter")
    train_norms = np.linalg.norm(train_features, axis=1, keepdims=True)
    normalized_train = train_features / np.maximum(train_norms, 1e-6)
    eval_norms = np.linalg.norm(eval_features, axis=1, keepdims=True)
    normalized_eval = eval_features / np.maximum(eval_norms, 1e-6)
    k = max(1, min(int(k), len(train_labels)))
    predictions = []
    top1_scores = []
    top2_scores = []
    margins = []
    for start in range(0, len(normalized_eval), chunk_size):
        chunk = normalized_eval[start : start + chunk_size]
        sims = chunk @ normalized_train.T
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for row_idx, indices in enumerate(top_idx):
            row_scores = sims[row_idx, indices]
            votes = defaultdict(float)
            counts = defaultdict(int)
            for index, score in zip(indices, row_scores):
                label = int(train_labels[index])
                votes[label] += float(score)
                counts[label] += 1
            ranked = sorted(votes.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
            predictions.append(ranked[0][0])
            top1_scores.append(ranked[0][1] / max(1, counts[ranked[0][0]]))
            if len(ranked) > 1:
                top2_scores.append(ranked[1][1] / max(1, counts[ranked[1][0]]))
            else:
                top2_scores.append(0.0)
            margins.append(top1_scores[-1] - top2_scores[-1])
    return (
        np.asarray(predictions, dtype=np.int32),
        {
            "top1_score": np.asarray(top1_scores, dtype=np.float32),
            "top2_score": np.asarray(top2_scores, dtype=np.float32),
            "margin": np.asarray(margins, dtype=np.float32),
        },
    )


def evaluate(predictions, labels, scores=None):
    correct = predictions == labels
    confusion = Counter((str(gt), str(pred)) for gt, pred in zip(labels.tolist(), predictions.tolist()))
    wrong = Counter((str(gt), str(pred)) for gt, pred in zip(labels.tolist(), predictions.tolist()) if gt != pred)
    result = {
        "accuracy": float(correct.mean()) if len(labels) else 0.0,
        "correct": int(correct.sum()),
        "total": int(len(labels)),
        "top_confusions": confusion.most_common(30),
        "top_wrong": wrong.most_common(30),
    }
    if scores is not None:
        result["selective_accuracy"] = selective_accuracy(predictions, labels, scores)
    return result


def selective_accuracy(predictions, labels, scores):
    rows = []
    margins = np.asarray(scores["margin"], dtype=np.float32)
    top1_scores = np.asarray(scores["top1_score"], dtype=np.float32)
    correct = predictions == labels
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    for threshold in thresholds:
        mask = margins >= threshold
        rows.append(
            {
                "margin_threshold": float(threshold),
                "coverage": float(mask.mean()) if len(mask) else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
                "selected": int(mask.sum()),
            }
        )
    score_thresholds = [0.2, 0.4, 0.6, 0.8]
    for threshold in score_thresholds:
        mask = top1_scores >= threshold
        rows.append(
            {
                "score_threshold": float(threshold),
                "coverage": float(mask.mean()) if len(mask) else 0.0,
                "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
                "selected": int(mask.sum()),
            }
        )
    return rows


def jersey_number(row):
    value = row.get("jersey_number") or row.get("jersey") or row.get("number")
    if value in (None, "", "None", "unknown"):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= 99 else None


def parse_size(value):
    width, height = str(value).lower().split("x", 1)
    return int(width), int(height)


if __name__ == "__main__":
    main()
