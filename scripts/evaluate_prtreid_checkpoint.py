#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ft.features.prtreid import PRTReIDFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PRTReID checkpoint on same-team validation retrieval.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--hrnet-pretrained-path", default="models/reid")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    validation = [row for row in rows if row["split"] in {"query", "gallery"}]
    extractor = PRTReIDFeatureExtractor(
        enabled=True,
        weights_path=args.weights,
        hrnet_pretrained_path=args.hrnet_pretrained_path,
        batch_size=args.batch_size,
        role_enabled=False,
    )
    extract_validation_rows(extractor, validation)
    report = evaluate(validation)
    report["weights"] = args.weights
    report["extractor"] = extractor.diagnostics()
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


def extract_validation_rows(extractor, rows):
    """Extract directly from paths in bounded batches instead of retaining all crops."""
    for row in rows:
        row["crop_path"] = row["img_path"]
    extractor.add_row_features(rows)
    failed = [row["img_path"] for row in rows if row.get("visual_embedding") is None]
    if failed:
        raise RuntimeError(f"Expected {len(rows)} embeddings; {len(failed)} crops failed, first={failed[0]}")
    for row in rows:
        row["embedding"] = np.asarray(row["visual_embedding"], dtype=np.float32)


def evaluate(rows):
    query = [row for row in rows if row["split"] == "query"]
    gallery = [row for row in rows if row["split"] == "gallery"]
    rank1 = []
    average_precisions = []
    positive_scores = []
    negative_same_team_scores = []
    for left in query:
        candidates = [right for right in gallery if right["video_id"] == left["video_id"]]
        ranked = sorted(((cosine(left["embedding"], right["embedding"]), right) for right in candidates), reverse=True, key=lambda item: item[0])
        positives = [index for index, (_score, right) in enumerate(ranked) if right["pid"] == left["pid"]]
        if not positives:
            continue
        rank1.append(positives[0] == 0)
        precisions = [(rank + 1) / (index + 1) for rank, index in enumerate(positives)]
        average_precisions.append(float(np.mean(precisions)))
        for score, right in ranked:
            if right["pid"] == left["pid"]:
                positive_scores.append(score)
            elif right["team_id"] == left["team_id"]:
                negative_same_team_scores.append(score)
    threshold = max(negative_same_team_scores, default=1.0) + 1e-8
    zero_fp_recall = sum(score >= threshold for score in positive_scores) / len(positive_scores) if positive_scores else 0.0
    return {
        "queries": len(rank1),
        "rank1": float(np.mean(rank1)) if rank1 else 0.0,
        "map": float(np.mean(average_precisions)) if average_precisions else 0.0,
        "positive_pairs": len(positive_scores),
        "same_team_negative_pairs": len(negative_same_team_scores),
        "zero_false_positive_threshold": threshold,
        "zero_false_positive_recall": zero_fp_recall,
    }


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["pid"] = int(row["pid"])
        row["team_id"] = int(row["team_id"])
    return rows


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0


if __name__ == "__main__":
    main()
