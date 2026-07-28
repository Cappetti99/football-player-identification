#!/usr/bin/env python3
"""Evaluate YOLO number-region + numeric CTC on every track of an OCR run."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import aggregate_frames, build_numeric_crnn, candidate_log_probabilities
from scripts.evaluate_jersey_number_region_ctc import crop_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--box-padding", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="0")
    parser.add_argument(
        "--dump-raw-scores",
        default=None,
        help="optional path to also write per-frame candidate log-probabilities and "
        "weights per track, for offline comparison of aggregation strategies. Does "
        "not change any existing output when omitted.",
    )
    args = parser.parse_args()
    if args.detector_batch_size < 1:
        parser.error("--detector-batch-size must be at least 1")

    import torch
    from torchvision import transforms
    from ultralytics import YOLO

    root = Path(args.ocr_run).resolve()
    predictions = read_predictions(root / "predictions.csv")
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    selected = selected_crops(diagnostics, predictions)
    if not selected:
        raise RuntimeError(
            "no selected crops found in ocr_diagnostics.json; "
            "expected tracklet records with display_track_id and selected_crops"
        )
    detector = YOLO(args.detector_checkpoint)
    region_items = []
    for start in range(0, len(selected), args.detector_batch_size):
        selected_batch = selected[start:start + args.detector_batch_size]
        detections = detector.predict(
            [row["crop_path"] for row in selected_batch],
            conf=args.detector_confidence,
            device=args.detector_device,
            batch=args.detector_batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(selected_batch, detections):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            index = int(result.boxes.conf.argmax().item())
            region_items.append({
                **row,
                "image": crop_image(
                    row["crop_path"],
                    tuple(result.boxes.xyxyn[index].tolist()),
                    args.box_padding,
                ),
                "detector_confidence": float(result.boxes.conf[index]),
            })
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint = torch.load(args.ctc_checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    recognizer = build_numeric_crnn(pretrained=False).to(args.device)
    recognizer.load_state_dict(checkpoint["state_dict"]); recognizer.eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(metadata["image_size"])), transforms.ToTensor(),
        transforms.Normalize(metadata["normalization"]["mean"], metadata["normalization"]["std"]),
    ])
    tracks = defaultdict(lambda: {"scores": [], "weights": [], "frames": []})
    for start in range(0, len(region_items), args.batch_size):
        batch = region_items[start:start + args.batch_size]
        with torch.no_grad():
            logits = recognizer(torch.stack([transform(row["image"]) for row in batch]).to(args.device)).cpu()
        for index, row in enumerate(batch):
            key = (row["sequence"], row["gt_track_id"])
            tracks[key]["scores"].append(candidate_log_probabilities(logits[:, index, :]))
            tracks[key]["weights"].append(row["detector_confidence"])
            tracks[key]["frames"].append(row["frame"])

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
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "predictions.csv", output_rows)
    if args.dump_raw_scores:
        dump_raw_scores(args.dump_raw_scores, tracks, predictions)
    metrics = summarize(output_rows, selected, region_items)
    metrics.update({
        "ocr_run": str(root), "ctc_checkpoint": str(Path(args.ctc_checkpoint).resolve()),
        "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
        "detector_confidence": args.detector_confidence, "box_padding": args.box_padding,
        "detector_batch_size": args.detector_batch_size,
    })
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def dump_raw_scores(path, tracks, predictions):
    """Persist per-frame candidate log-probabilities and weights per track.

    Purely additive diagnostic export: lets a separate offline script compare
    alternative aggregation strategies against the exact same per-frame CTC
    outputs, without re-running the detector or the recognizer.
    """
    payload = {}
    for key, data in tracks.items():
        reference = predictions.get(key, {})
        payload[f"{key[0]}::{key[1]}"] = {
            "sequence": key[0],
            "gt_track_id": key[1],
            "eval_track_id": reference.get("eval_track_id"),
            "gt_jersey_number": reference.get("gt"),
            "frames": data["frames"],
            "weights": data["weights"],
            "scores": data["scores"],
        }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_predictions(path):
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row["sequence"]), str(row["gt_track_id"])
            output[key] = {
                "eval_track_id": str(row["eval_track_id"]),
                "gt": int(float(row["gt_jersey_number"])),
            }
    return output


def selected_crops(diagnostics, predictions):
    by_eval = {row["eval_track_id"]: key for key, row in predictions.items()}
    output, seen = [], set()
    tracklets = diagnostics.get("tracklets") if isinstance(diagnostics, dict) else None
    records = tracklets.values() if isinstance(tracklets, dict) else diagnostics.values()
    for diagnostic in records:
        if not isinstance(diagnostic, dict):
            continue
        key = by_eval.get(str(diagnostic.get("display_track_id")))
        if key is None:
            continue
        for crop in diagnostic.get("selected_crops", []):
            path = str(crop.get("crop_path") or "")
            identity = key, path
            if not path or identity in seen:
                continue
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            seen.add(identity)
            output.append({
                "sequence": key[0], "gt_track_id": key[1], "crop_path": path,
                "frame": int(crop.get("frame") or 0),
            })
    return output


def summarize(rows, selected, regions):
    assigned = [row for row in rows if row["assigned"]]
    correct = [row for row in assigned if row["correct"]]
    return {
        "tracklets": len(rows), "selected_crops": len(selected), "detected_regions": len(regions),
        "region_detection_coverage": ratio(len(regions), len(selected)),
        "assigned_tracklets": len(assigned), "coverage": ratio(len(assigned), len(rows)),
        "correct_tracklets": len(correct), "wrong_tracklets": len(assigned)-len(correct),
        "accuracy_assigned": ratio(len(correct), len(assigned)),
        "accuracy_all": ratio(len(correct), len(rows)),
        "gt_in_top5": sum(row["gt_in_top5"] for row in rows),
        "gt_in_top5_rate": ratio(sum(row["gt_in_top5"] for row in rows), len(rows)),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ratio(a, b):
    return a / b if b else 0.0


if __name__ == "__main__":
    main()
