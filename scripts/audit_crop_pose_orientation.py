#!/usr/bin/env python3
"""Estimate back-vs-front orientation of number-region crops using a generic
COCO pose estimator, as an alternative to the trained `clean_back` classifier.

Motivation: the `clean_back` classifier (trained on 337 GSR crops) failed to
recover any of the 12 fully-blind tracklets from the frozen coverage-gap
audit, even when the full 20-crop pool per track had candidates it scored as
"clean back" with high probability. A generic pose estimator, pretrained on
COCO (a much larger and more diverse dataset, not sports-specific), might
generalize better to unseen broadcast domains than a classifier trained on
only 337 in-domain crops.

Orientation rule (no training, pure geometry):
  COCO keypoints are labeled by the person's own anatomy, not image side.
  Facing the camera: the person's left shoulder (COCO index 5) appears on the
  image's RIGHT side (mirrored). Back to the camera: it appears on the image's
  LEFT side. Comparing the x-coordinates of left_shoulder vs right_shoulder
  therefore gives a deterministic front/back signal when both are confidently
  detected. Nose-keypoint confidence is recorded as a corroborating signal
  only (low nose confidence is consistent with, but does not by itself prove,
  a back-facing pose), never as the primary decision.

Honest caveat: pose estimators are typically evaluated on person crops much
larger than ours (25-70px min side here). This script measures, it does not
assume, whether keypoints are reliable at this crop size -- see
`pose_detected` / keypoint confidences in the output before trusting
`predicted_back_facing` on a given track.

Input: `region_detection_coverage.csv` from
`audit_jersey_number_region_detector_coverage.py`. Zero training, zero new
labels, pure inference with a stock Ultralytics pose checkpoint.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# COCO keypoint indices (Ultralytics pose output order).
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="region_detection_coverage.csv path")
    parser.add_argument(
        "--pose-checkpoint",
        default="yolov8x-pose.pt",
        help="Ultralytics pose checkpoint name (auto-downloaded). If this exact "
        "name 404s, try 'yolo11x-pose.pt' or 'yolov8n-pose.pt' (smaller/faster).",
    )
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--pose-confidence", type=float, default=0.25, help="person-detection conf for the pose model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracks", default=None, help="optional comma-separated sequence::gt_track_id filter for the printed summary")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    if not rows:
        raise RuntimeError(f"no rows in {args.csv}")

    from ultralytics import YOLO

    pose_model = YOLO(args.pose_checkpoint)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        results = pose_model.predict(
            [row["crop_path"] for row in batch],
            conf=args.pose_confidence,
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            annotate_orientation(row, result, args.keypoint_confidence_threshold)

    per_track = tracklet_breakdown(rows)
    pose_detected_count = sum(1 for row in rows if row["pose_detected"])
    # "unreliable" collapses two distinct failure modes that need separate
    # rates: missing (the pose model finds no person at all in the crop) vs.
    # failure (a person is found but the shoulder keypoints are too low
    # confidence to determine orientation). Conflating them hides whether the
    # bottleneck is person detection or keypoint confidence.
    missing_count = len(rows) - pose_detected_count
    failure_count = sum(
        1 for row in rows if row["pose_detected"] and row["orientation"] == "undetermined"
    )
    reliable_count = pose_detected_count - failure_count
    summary = {
        "pose_checkpoint": args.pose_checkpoint,
        "keypoint_confidence_threshold": args.keypoint_confidence_threshold,
        "total_crops": len(rows),
        "pose_detected": pose_detected_count,
        "reliable_count": reliable_count,
        "reliable_rate": ratio(reliable_count, len(rows)),
        "missing_count": missing_count,
        "missing_rate": ratio(missing_count, len(rows)),
        "failure_count": failure_count,
        "failure_rate": ratio(failure_count, len(rows)),
        "orientation_counts": counts_by(rows, "orientation"),
        "detected_vs_orientation": cross_tab(rows, "orientation", "detected"),
        "hard_miss_vs_orientation": cross_tab(
            [row for row in rows if not truthy(row.get("detected"))], "orientation", None
        ),
    }

    requested = {item.strip() for item in (args.tracks or "").split(",") if item.strip()}
    if requested:
        summary["requested_tracks"] = {
            key: stats for key, stats in per_track.items() if key in requested
        }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "crop_pose_orientation.csv", rows)
    (output / "crop_pose_orientation_summary.json").write_text(
        json.dumps({"summary": summary, "per_track": per_track}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\ncsv={output / 'crop_pose_orientation.csv'}")
    print(f"summary={output / 'crop_pose_orientation_summary.json'}")


def annotate_orientation(row, result, keypoint_threshold):
    row["pose_detected"] = False
    row["left_shoulder_conf"] = None
    row["right_shoulder_conf"] = None
    row["nose_conf"] = None
    row["shoulder_orientation"] = "undetermined"
    row["nose_corroborates_back"] = None
    row["orientation"] = "undetermined"
    row["predicted_back_facing"] = False

    if result.keypoints is None or result.keypoints.conf is None or len(result.keypoints.conf) == 0:
        return

    # Multiple people can appear in one crop (Fig. review showed opponents in
    # frame); take the detection with the highest overall box confidence as
    # the subject, consistent with how the region detector already treats the
    # highest-confidence box as the crop's primary subject.
    box_confidences = result.boxes.conf if result.boxes is not None else None
    if box_confidences is None or len(box_confidences) == 0:
        return
    person_index = int(box_confidences.argmax().item())

    xy = result.keypoints.xy[person_index]
    conf = result.keypoints.conf[person_index]
    if len(conf) <= max(NOSE, LEFT_SHOULDER, RIGHT_SHOULDER):
        return

    row["pose_detected"] = True
    left_conf = float(conf[LEFT_SHOULDER])
    right_conf = float(conf[RIGHT_SHOULDER])
    nose_conf = float(conf[NOSE])
    row["left_shoulder_conf"] = left_conf
    row["right_shoulder_conf"] = right_conf
    row["nose_conf"] = nose_conf

    if left_conf >= keypoint_threshold and right_conf >= keypoint_threshold:
        left_x = float(xy[LEFT_SHOULDER][0])
        right_x = float(xy[RIGHT_SHOULDER][0])
        row["shoulder_orientation"] = "back" if left_x < right_x else "front"

    row["nose_corroborates_back"] = nose_conf < (1.0 - keypoint_threshold)
    row["orientation"] = row["shoulder_orientation"]
    row["predicted_back_facing"] = row["orientation"] == "back"


def tracklet_breakdown(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row.get('sequence', '')}::{row.get('gt_track_id', '')}"].append(row)
    out = {}
    for key, items in grouped.items():
        out[key] = {
            "crops": len(items),
            "pose_detected": sum(1 for row in items if row["pose_detected"]),
            "predicted_back_facing": sum(1 for row in items if row["predicted_back_facing"]),
            "detected_region": sum(1 for row in items if truthy(row.get("detected"))),
        }
    return out


def counts_by(rows, key):
    counts = defaultdict(int)
    for row in rows:
        counts[str(row.get(key))] += 1
    return dict(counts)


def cross_tab(rows, key_a, key_b):
    table = defaultdict(lambda: defaultdict(int))
    for row in rows:
        a = str(row.get(key_a))
        b = str(truthy(row.get(key_b))) if key_b else "count"
        table[a][b] += 1
    return {a: dict(b) for a, b in table.items()}


def truthy(value):
    return str(value).strip().lower() == "true"


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
