#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from build_prtreid_pair_audit import write_html


def main():
    parser = argparse.ArgumentParser(description="Build visual sheets for reliable-jersey near misses.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--output-dir")
    parser.add_argument("--crops-per-tracklet", type=int, default=6)
    args = parser.parse_args()
    metadata = Path(args.artifacts_root) / args.run / "metadata"
    report = json.loads((metadata / f"{args.video_id}_identity_gate_audit.json").read_text())
    output = Path(args.output_dir or Path("evaluation_outputs/identity_nearmiss_audit") / args.run)
    sheets = output / "tracklets"
    sheets.mkdir(parents=True, exist_ok=True)
    labels = []
    for index, row in enumerate(report.get("near_misses", []), 1):
        audit_id = f"{args.video_id}_{index:04d}"
        paths = representative(row.get("crop_paths", []), args.crops_per_tracklet)
        sheet = sheets / f"{audit_id}.jpg"
        write_sheet(paths, sheet, row)
        labels.append({
            "pair_id": audit_id,
            "audit_id": audit_id,
            "video_id": args.video_id,
            "run": args.run,
            "tracklet_id": row["tracklet_id"],
            "best_player_id": row.get("best_player_id"),
            "best_player_name": row.get("best_player_name"),
            "best_player_jersey_number": row.get("best_player_jersey_number"),
            "jersey_candidate_score": row.get("jersey_candidate_score"),
            "jersey_confidence": row.get("jersey_confidence"),
            "jersey_votes": row.get("jersey_votes"),
            "jersey_winner_margin": row.get("jersey_winner_margin"),
            "raw_jersey_distribution": json.dumps(row.get("raw_jersey_distribution", [])),
            "link_type": "identity_jersey_nearmiss",
            "visual_similarity": row.get("jersey_candidate_score"),
            "similarity_margin": 0.0,
            "sheet_path": str(sheet),
            "label": "uncertain",
            "notes": "",
        })
    write_csv(labels, output / "labels.csv")
    write_html(labels, output / "index.html")
    print(f"near_misses={len(labels)} output={output}")


def representative(paths, limit):
    paths = [str(path) for path in paths if path]
    if len(paths) <= limit:
        return paths
    indices = np.linspace(0, len(paths) - 1, int(limit)).round().astype(int)
    return [paths[int(index)] for index in indices]


def write_sheet(paths, output, row):
    images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        images.append(letterbox(image, 128, 256))
    if not images:
        images = [np.zeros((256, 128, 3), dtype=np.uint8)]
    strip = np.hstack(images)
    header = np.zeros((55, strip.shape[1], 3), dtype=np.uint8)
    text = f"{row.get('best_player_id')} #{row.get('best_player_jersey_number')} score={float(row.get('jersey_candidate_score') or 0):.4f}"
    cv2.putText(header, text, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), np.vstack([header, strip]))


def letterbox(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def write_csv(rows, path):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
