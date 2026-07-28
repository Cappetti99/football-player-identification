#!/usr/bin/env python3
"""Evaluate the region-CTC recognizer using a pose-keypoint-derived torso box
instead of the trained YOLO number-region detector.

Motivation: the whole 2026-07-22 investigation showed the YOLO region
detector's generalization gap is structural (more epochs, external SJN-210k
pretraining: both only +3pp coverage, both with a worse hard-miss trade-off).
Published pipelines (Liu & Bhanu 2019; Koshkina et al. 2024, ViTPose-guided
crop) sidestep training a region detector entirely: they derive the
torso/number-region box directly from generic pose keypoints (shoulders,
hips), pretrained on COCO -- the same generic pose signal already shown today
to correlate far more strongly with number visibility (81% vs 29% coverage
by back/front orientation) than anything trained only on 337 GSR crops.

This script re-uses the exact same selected crops, CTC checkpoint, and
accuracy computation as `evaluate_jersey_number_region_ctc_ocr_run.py`, so
its output is directly comparable to the frozen `sjn_to_gsr` result
(accuracy_assigned=77.99%, accuracy_all=71.68%, region_detection_coverage
=81.6%) -- the only variable changed is how the crop fed to the CTC is
produced: geometry from pose keypoints instead of a trained detector.

Zero new training, zero new labels: the pose checkpoint is a stock
Ultralytics release, and the torso-box geometry is a fixed rule (tunable via
CLI), not learned on our data.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.jersey_number_ctc import aggregate_frames, build_numeric_crnn, candidate_log_probabilities  # noqa: E402
from scripts.evaluate_jersey_number_region_ctc_ocr_run import read_predictions, selected_crops  # noqa: E402

# COCO keypoint indices (Ultralytics pose output order).
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument("--pose-checkpoint", default="yolov8x-pose.pt")
    parser.add_argument("--keypoint-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--top-fraction", type=float, default=0.15,
                         help="region top edge, as a fraction of shoulder-to-hip distance below the shoulder line")
    parser.add_argument("--bottom-fraction", type=float, default=0.75,
                         help="region bottom edge, as a fraction of shoulder-to-hip distance below the shoulder line")
    parser.add_argument("--side-padding-fraction", type=float, default=0.15,
                         help="extra horizontal padding as a fraction of shoulder width")
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pose-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pose-device", default="0")
    args = parser.parse_args()

    import torch
    from torchvision import transforms
    from ultralytics import YOLO

    root = Path(args.ocr_run).resolve()
    predictions = read_predictions(root / "predictions.csv")
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    selected = selected_crops(diagnostics, predictions)
    if not selected:
        raise RuntimeError("no selected crops found in ocr_diagnostics.json")

    pose_model = YOLO(args.pose_checkpoint)
    region_items = []
    for start in range(0, len(selected), args.pose_batch_size):
        batch = selected[start:start + args.pose_batch_size]
        results = pose_model.predict(
            [row["crop_path"] for row in batch],
            device=args.pose_device,
            batch=args.pose_batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            box = torso_box_from_pose_result(
                result,
                args.keypoint_confidence_threshold,
                args.top_fraction,
                args.bottom_fraction,
                args.side_padding_fraction,
            )
            if box is None:
                continue
            image = crop_box(row["crop_path"], box)
            if image is None:
                continue
            region_items.append({**row, "image": image})
    del pose_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint = torch.load(args.ctc_checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    recognizer = build_numeric_crnn(pretrained=False).to(args.device)
    recognizer.load_state_dict(checkpoint["state_dict"])
    recognizer.eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(metadata["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(metadata["normalization"]["mean"], metadata["normalization"]["std"]),
    ])

    tracks = defaultdict(lambda: {"scores": [], "weights": []})
    for start in range(0, len(region_items), args.batch_size):
        batch = region_items[start:start + args.batch_size]
        with torch.no_grad():
            logits = recognizer(torch.stack([transform(row["image"]) for row in batch]).to(args.device)).cpu()
        for index, row in enumerate(batch):
            key = (row["sequence"], row["gt_track_id"])
            tracks[key]["scores"].append(candidate_log_probabilities(logits[:, index, :]))
            tracks[key]["weights"].append(1.0)  # pose geometry has no learned confidence to weight by

    output_rows = []
    for key, reference in sorted(predictions.items()):
        result = aggregate_frames(tracks[key]["scores"], tracks[key]["weights"])
        assigned = result["prediction"] is not None
        truth = reference["gt"]
        output_rows.append({
            "eval_track_id": reference["eval_track_id"], "sequence": key[0],
            "gt_track_id": key[1], "gt_jersey_number": truth,
            "pred_jersey_number": "" if not assigned else result["prediction"],
            "assigned": assigned, "correct": assigned and result["prediction"] == truth,
            "confidence": result["confidence"], "winner_margin": result["margin"],
            "recognized_frames": len(tracks[key]["scores"]),
            "gt_in_top5": str(truth) in dict(list(result["scores"].items())[:5]),
        })

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "predictions.csv", output_rows)
    metrics = summarize(output_rows, selected, region_items)
    metrics.update({
        "ocr_run": str(root), "ctc_checkpoint": str(Path(args.ctc_checkpoint).resolve()),
        "pose_checkpoint": args.pose_checkpoint,
        "keypoint_confidence_threshold": args.keypoint_confidence_threshold,
        "top_fraction": args.top_fraction, "bottom_fraction": args.bottom_fraction,
        "side_padding_fraction": args.side_padding_fraction,
    })
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def torso_box_from_pose_result(result, keypoint_threshold, top_fraction, bottom_fraction, side_padding_fraction):
    if result.keypoints is None or result.keypoints.conf is None or len(result.keypoints.conf) == 0:
        return None
    box_confidences = result.boxes.conf if result.boxes is not None else None
    if box_confidences is None or len(box_confidences) == 0:
        return None
    person_index = int(box_confidences.argmax().item())

    xy = result.keypoints.xy[person_index]
    conf = result.keypoints.conf[person_index]
    if len(conf) <= max(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
        return None
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if any(float(conf[index]) < keypoint_threshold for index in needed):
        return None

    ls_x, ls_y = float(xy[LEFT_SHOULDER][0]), float(xy[LEFT_SHOULDER][1])
    rs_x, rs_y = float(xy[RIGHT_SHOULDER][0]), float(xy[RIGHT_SHOULDER][1])
    lh_x, lh_y = float(xy[LEFT_HIP][0]), float(xy[LEFT_HIP][1])
    rh_x, rh_y = float(xy[RIGHT_HIP][0]), float(xy[RIGHT_HIP][1])

    shoulder_y = (ls_y + rs_y) / 2.0
    hip_y = (lh_y + rh_y) / 2.0
    torso_height = hip_y - shoulder_y
    if torso_height <= 1.0:
        return None  # degenerate pose (upside down / crossed keypoints)

    top = shoulder_y + top_fraction * torso_height
    bottom = shoulder_y + bottom_fraction * torso_height
    shoulder_width = abs(rs_x - ls_x)
    pad = side_padding_fraction * max(shoulder_width, 1.0)
    left = min(ls_x, rs_x, lh_x, rh_x) - pad
    right = max(ls_x, rs_x, lh_x, rh_x) + pad
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def crop_box(path, box):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = box
        clipped = (
            max(0, int(x1)), max(0, int(y1)),
            min(width, int(x2)), min(height, int(y2)),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return image.crop(clipped).copy()


def summarize(rows, selected, regions):
    assigned = [row for row in rows if row["assigned"]]
    correct = [row for row in assigned if row["correct"]]
    return {
        "tracklets": len(rows), "selected_crops": len(selected), "detected_regions": len(regions),
        "region_detection_coverage": ratio(len(regions), len(selected)),
        "assigned_tracklets": len(assigned), "coverage": ratio(len(assigned), len(rows)),
        "correct_tracklets": len(correct), "wrong_tracklets": len(assigned) - len(correct),
        "accuracy_assigned": ratio(len(correct), len(assigned)),
        "accuracy_all": ratio(len(correct), len(rows)),
        "gt_in_top5": sum(row["gt_in_top5"] for row in rows),
        "gt_in_top5_rate": ratio(sum(row["gt_in_top5"] for row in rows), len(rows)),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return a / b if b else 0.0


if __name__ == "__main__":
    main()
