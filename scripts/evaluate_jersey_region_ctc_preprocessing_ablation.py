#!/usr/bin/env python3
"""Evaluate Region-CTC padding/preprocessing variants on labeled GSR tracks."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
    greedy_decode,
)
from scripts.audit_jersey_region_ctc_preprocessing import (
    PREPROCESSING,
    preprocess_region,
    variant_name,
)
from scripts.evaluate_jersey_number_region_ctc import crop_image
from scripts.evaluate_jersey_number_region_ctc_ocr_run import (
    read_predictions,
    selected_crops,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-manifest", required=True)
    parser.add_argument(
        "--manifest-part",
        choices=("validation", "frozen"),
        required=True,
    )
    parser.add_argument("--padding", type=float, action="append", dest="paddings")
    parser.add_argument(
        "--preprocessing",
        action="append",
        dest="preprocessing",
        choices=PREPROCESSING,
    )
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--detector-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="0")
    args = parser.parse_args()
    args.paddings = args.paddings or [0.10, 0.25, 0.40]
    args.preprocessing = args.preprocessing or list(PREPROCESSING)
    if any(value < 0 or value > 0.5 for value in args.paddings):
        parser.error("--padding must be between 0 and 0.5")
    if args.upscale < 1 or args.batch_size < 1 or args.detector_batch_size < 1:
        parser.error("batch sizes and --upscale must be positive")
    if not 0 <= args.detector_confidence <= 1:
        parser.error("--detector-confidence must be between 0 and 1")
    return args


def detect_regions(selected, checkpoint, confidence, batch_size, device):
    from ultralytics import YOLO

    detector = YOLO(checkpoint)
    output = []
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        predictions = detector.predict(
            [row["crop_path"] for row in batch],
            conf=confidence,
            device=device,
            batch=batch_size,
            verbose=False,
            stream=False,
        )
        for row, prediction in zip(batch, predictions):
            if prediction.boxes is None or len(prediction.boxes) == 0:
                continue
            index = int(prediction.boxes.conf.argmax().item())
            output.append({
                **row,
                "region_xyxyn": tuple(
                    float(value) for value in prediction.boxes.xyxyn[index].tolist()
                ),
                "detector_confidence": float(prediction.boxes.conf[index]),
            })
    return output


def evaluate_variant(
    regions,
    predictions,
    model,
    transform,
    padding,
    preprocessing,
    upscale,
    batch_size,
    device,
):
    import torch

    frame_rows = []
    tracks = defaultdict(lambda: {"scores": [], "weights": [], "frames": []})
    for start in range(0, len(regions), batch_size):
        batch = regions[start:start + batch_size]
        images = [
            preprocess_region(
                crop_image(row["crop_path"], row["region_xyxyn"], padding),
                preprocessing,
                upscale,
            )
            for row in batch
        ]
        with torch.no_grad():
            logits = model(torch.stack([transform(image) for image in images]).to(device)).cpu()
        for index, row in enumerate(batch):
            scores = candidate_log_probabilities(logits[:, index, :])
            decoded, confidence = greedy_decode(logits[:, index, :])
            key = row["sequence"], row["gt_track_id"]
            truth = predictions[key]["gt"]
            tracks[key]["scores"].append(scores)
            tracks[key]["weights"].append(row["detector_confidence"])
            tracks[key]["frames"].append(row["frame"])
            frame_rows.append({
                "sequence": key[0],
                "gt_track_id": key[1],
                "frame": row["frame"],
                "gt_jersey_number": truth,
                "prediction": decoded,
                "correct": decoded == str(truth),
                "recognition_confidence": confidence,
                "detector_confidence": row["detector_confidence"],
                "crop_path": row["crop_path"],
            })
    track_rows = []
    for key, reference in sorted(predictions.items()):
        result = aggregate_frames(tracks[key]["scores"], tracks[key]["weights"])
        assigned = result["prediction"] is not None
        truth = reference["gt"]
        top5 = [int(value) for value in list(result["scores"])[:5]]
        track_rows.append({
            "eval_track_id": reference["eval_track_id"],
            "sequence": key[0],
            "gt_track_id": key[1],
            "gt_jersey_number": truth,
            "digits": len(str(truth)),
            "prediction": "" if not assigned else result["prediction"],
            "assigned": assigned,
            "correct": assigned and result["prediction"] == truth,
            "confidence": result["confidence"],
            "winner_margin": result["margin"],
            "recognized_frames": len(tracks[key]["scores"]),
            "gt_in_top5": truth in top5,
            "top5": top5,
        })
    return frame_rows, track_rows


def summarize(crop_rows, track_rows, selected_count, detected_count):
    assigned = [row for row in track_rows if row["assigned"]]
    correct = [row for row in assigned if row["correct"]]
    high_confidence_wrong = [
        row for row in assigned if not row["correct"] and row["confidence"] >= 0.90
    ]
    return {
        "selected_crops": selected_count,
        "detected_regions": detected_count,
        "region_detection_coverage": ratio(detected_count, selected_count),
        "recognized_crops": len(crop_rows),
        "crop_correct": sum(row["correct"] for row in crop_rows),
        "crop_accuracy": ratio(sum(row["correct"] for row in crop_rows), len(crop_rows)),
        "tracklets": len(track_rows),
        "assigned_tracklets": len(assigned),
        "coverage": ratio(len(assigned), len(track_rows)),
        "correct_tracklets": len(correct),
        "wrong_tracklets": len(assigned) - len(correct),
        "accuracy_assigned": ratio(len(correct), len(assigned)),
        "accuracy_all": ratio(len(correct), len(track_rows)),
        "gt_in_top5": sum(row["gt_in_top5"] for row in track_rows),
        "gt_in_top5_rate": ratio(sum(row["gt_in_top5"] for row in track_rows), len(track_rows)),
        "high_confidence_wrong": len(high_confidence_wrong),
        "by_digits": grouped_metrics(track_rows, "digits"),
        "by_sequence": grouped_metrics(track_rows, "sequence"),
        "prediction_distribution": dict(sorted(Counter(
            str(row["prediction"]) for row in assigned
        ).items())),
    }


def grouped_metrics(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    output = {}
    for key, values in sorted(groups.items()):
        assigned = [row for row in values if row["assigned"]]
        correct = sum(row["correct"] for row in values)
        output[key] = {
            "tracklets": len(values),
            "assigned": len(assigned),
            "correct": correct,
            "coverage": ratio(len(assigned), len(values)),
            "accuracy_all": ratio(correct, len(values)),
            "gt_in_top5_rate": ratio(sum(row["gt_in_top5"] for row in values), len(values)),
        }
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def declared_sequences(manifest, part):
    if part == "validation":
        values = manifest.get("validation_sequences") or []
    else:
        values = (
            manifest.get("frozen_validation_sequences")
            or manifest.get("frozen_sequences")
            or []
        )
    if not values:
        raise ValueError(f"sequence manifest has no sequences for part {part!r}")
    return {str(value) for value in values}


def validate_observed_sequences(predictions, allowed):
    observed = {str(sequence) for sequence, _ in predictions}
    unexpected = observed - set(allowed)
    if unexpected:
        raise ValueError(
            "OCR run contains sequences outside requested manifest part: "
            + ", ".join(sorted(unexpected))
        )
    if not observed:
        raise ValueError("OCR run contains no sequences")
    return observed


def main():
    args = parse_args()
    import torch
    from torchvision import transforms

    root = Path(args.ocr_run).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    predictions = read_predictions(root / "predictions.csv")
    sequence_manifest_path = Path(args.sequence_manifest).resolve()
    sequence_manifest = json.loads(sequence_manifest_path.read_text())
    allowed_sequences = declared_sequences(sequence_manifest, args.manifest_part)
    observed_sequences = validate_observed_sequences(predictions, allowed_sequences)
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    selected = selected_crops(diagnostics, predictions)
    if not selected:
        raise RuntimeError("OCR run has no selected crops")
    regions = detect_regions(
        selected,
        args.detector_checkpoint,
        args.detector_confidence,
        args.detector_batch_size,
        args.detector_device,
    )
    checkpoint = torch.load(args.ctc_checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    model = build_numeric_crnn(pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(metadata["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            metadata["normalization"]["mean"], metadata["normalization"]["std"]
        ),
    ])
    summary = {
        "scope": (
            "development_sequence_disjoint"
            if args.manifest_part == "validation"
            else "frozen_sequence_disjoint"
        ),
        "manifest_part": args.manifest_part,
        "sequence_manifest": str(sequence_manifest_path),
        "sequence_manifest_sha256": sha256(sequence_manifest_path),
        "observed_sequences": sorted(observed_sequences),
        "allowed_sequences": sorted(allowed_sequences),
        "unexpected_sequences": [],
        "ocr_run": str(root),
        "ctc_checkpoint": str(Path(args.ctc_checkpoint).resolve()),
        "ctc_checkpoint_sha256": sha256(args.ctc_checkpoint),
        "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
        "detector_checkpoint_sha256": sha256(args.detector_checkpoint),
        "detector_confidence": args.detector_confidence,
        "upscale": args.upscale,
        "variants": {},
    }
    for padding in args.paddings:
        for preprocessing in args.preprocessing:
            name = variant_name(padding, preprocessing, args.upscale)
            crop_rows, track_rows = evaluate_variant(
                regions, predictions, model, transform, padding, preprocessing,
                args.upscale, args.batch_size, args.device,
            )
            write_csv(output / f"{name}_crop_predictions.csv", crop_rows)
            write_csv(output / f"{name}_track_predictions.csv", track_rows)
            summary["variants"][name] = summarize(
                crop_rows, track_rows, len(selected), len(regions)
            )
            print(name, json.dumps(summary["variants"][name]))
    ranked = sorted(
        summary["variants"],
        key=lambda name: (
            -summary["variants"][name]["accuracy_all"],
            -summary["variants"][name]["gt_in_top5_rate"],
            summary["variants"][name]["high_confidence_wrong"],
            name,
        ),
    )
    summary["ranking"] = ranked
    summary["best_development_variant"] = ranked[0]
    summary["frozen_sequences_observed"] = (
        sorted(observed_sequences) if args.manifest_part == "frozen" else []
    )
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"best_development_variant": ranked[0], "ranking": ranked}, indent=2))


if __name__ == "__main__":
    main()
