#!/usr/bin/env python3
"""Simulate re-ranking candidate crops by pose-based back/front orientation.

Variant of `simulate_clean_back_reselection.py`, which re-ranked by the
trained `clean_back` classifier and recovered 0/12 blind tracklets. The
diagnostic in `audit_crop_pose_orientation.py` showed the geometric pose
signal (shoulder-keypoint left/right ordering) is far more discriminative
than that classifier: 81.0% region-detection coverage on pose-predicted
"back" crops vs 29.0% on "front" crops, vs heavily overlapping distributions
for clean_back_probability. This script tests whether re-ranking by that
stronger signal actually recovers detections for tracklets that are
currently blind because their originally-selected top-5 crops (by legacy
legibility score) contain no back-facing candidate.

Same mechanics as the clean_back variant: globs the raw candidate crops
already on disk (no re-extraction), scores them, takes the top `--top-k` by
score, runs the frozen region detector on that alternate selection, and
reports whether coverage was recovered. Pure offline simulation, no runtime
or config changes, no re-run of the official/frozen benchmark itself.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_crop_pose_orientation import annotate_orientation  # noqa: E402
from scripts.evaluate_jersey_number_region_ctc_ocr_run import read_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument(
        "--tracks",
        required=True,
        help="comma-separated sequence::gt_track_id pairs to simulate",
    )
    parser.add_argument("--pose-checkpoint", default="yolov8x-pose.pt")
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint-sha256", default=None)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.ocr_run).resolve()
    predictions = read_predictions(root / "predictions.csv")
    requested = [parse_track(item) for item in args.tracks.split(",") if item.strip()]

    candidates = []
    for sequence, gt_track_id in requested:
        reference = predictions.get((sequence, gt_track_id))
        if reference is None:
            print(f"WARNING: {sequence}::{gt_track_id} not found in predictions.csv", file=sys.stderr)
            continue
        eval_id = int(reference["eval_track_id"])
        pattern = f"track_{eval_id:06d}_*.jpg"
        paths = sorted((root / "crops" / safe_name(sequence)).glob(pattern))
        if not paths:
            print(f"WARNING: no candidate crops found on disk for {sequence}::{gt_track_id} "
                  f"(pattern {pattern})", file=sys.stderr)
        for path in paths:
            candidates.append({
                "sequence": sequence,
                "gt_track_id": gt_track_id,
                "eval_track_id": eval_id,
                "crop_path": str(path),
            })

    if not candidates:
        raise RuntimeError("no candidate crops found for any requested track")

    verify_checkpoint(args.detector_checkpoint, args.detector_checkpoint_sha256)

    from ultralytics import YOLO

    pose_model = YOLO(args.pose_checkpoint)
    for start in range(0, len(candidates), args.batch_size):
        batch = candidates[start:start + args.batch_size]
        results = pose_model.predict(
            [row["crop_path"] for row in batch],
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            annotate_orientation(row, result, args.keypoint_confidence_threshold)
            row["pose_back_score"] = pose_back_score(row)
    del pose_model

    reselected = []
    per_track_candidates = defaultdict(list)
    for row in candidates:
        per_track_candidates[(row["sequence"], row["gt_track_id"])].append(row)
    for rows in per_track_candidates.values():
        ranked = sorted(rows, key=lambda row: -row["pose_back_score"])
        reselected.extend(ranked[: args.top_k])

    detector = YOLO(args.detector_checkpoint)
    for start in range(0, len(reselected), args.batch_size):
        batch = reselected[start:start + args.batch_size]
        results = detector.predict(
            [row["crop_path"] for row in batch],
            conf=args.detector_confidence,
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            if result.boxes is None or len(result.boxes) == 0:
                row["detected"] = False
                row["detector_confidence"] = None
                continue
            index = int(result.boxes.conf.argmax().item())
            row["detected"] = True
            row["detector_confidence"] = float(result.boxes.conf[index])

    per_track_summary = {}
    for (sequence, gt_track_id), rows in per_track_candidates.items():
        label = f"{sequence}::{gt_track_id}"
        chosen = [row for row in reselected if row["sequence"] == sequence and row["gt_track_id"] == gt_track_id]
        per_track_summary[label] = {
            "raw_candidates_on_disk": len(rows),
            "reselected_top_k": len(chosen),
            "reselected_detected": sum(1 for row in chosen if row.get("detected")),
            "reselected_orientations": [row["orientation"] for row in chosen],
            "best_pose_back_score_in_full_pool": max(row["pose_back_score"] for row in rows),
            "back_facing_candidates_in_full_pool": sum(
                1 for row in rows if row["orientation"] == "back"
            ),
        }

    recovered_tracks = [
        label for label, stats in per_track_summary.items() if stats["reselected_detected"] > 0
    ]
    summary = {
        "ocr_run": str(root),
        "pose_checkpoint": args.pose_checkpoint,
        "top_k": args.top_k,
        "tracks_simulated": len(per_track_summary),
        "tracks_with_at_least_one_detection_after_reselection": len(recovered_tracks),
        "recovered_track_ids": sorted(recovered_tracks),
        "per_track": per_track_summary,
        "note": (
            "Compare 'reselected_detected' > 0 against the original run, where these "
            "tracks had zero detections at any confidence. recovered_track_ids here vs "
            "the 0/12 recovered by the clean_back-classifier reselection is the direct "
            "signal of whether the pose-based signal is actually more useful, not just "
            "more discriminative in aggregate."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "reselected_candidates.csv", reselected)
    write_csv(output / "all_candidates_scored.csv", candidates)
    (output / "reselection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\ncsv={output / 'reselected_candidates.csv'}")
    print(f"summary={output / 'reselection_summary.json'}")


def pose_back_score(row):
    """Rank key: definite back-facing beats undetermined beats front-facing;
    ties within a category broken by shoulder-keypoint confidence."""
    orientation = row.get("orientation")
    left_conf = row.get("left_shoulder_conf") or 0.0
    right_conf = row.get("right_shoulder_conf") or 0.0
    tie_break = min(left_conf, right_conf)
    if orientation == "back":
        return 2.0 + tie_break
    if orientation == "undetermined":
        return 0.5 + tie_break
    return 0.0 + tie_break


def verify_checkpoint(path, expected_sha256):
    if not expected_sha256:
        return
    digest = sha256_file(path)
    if digest.lower() != expected_sha256.lower():
        raise ValueError(f"checkpoint SHA-256 mismatch for {path}: {digest}")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def parse_track(item):
    sequence, gt_track_id = item.strip().split("::")
    return sequence, gt_track_id


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
