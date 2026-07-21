#!/usr/bin/env python3
"""Compare fixed, manual-region and YOLO-region inputs with the same CTC model."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
    greedy_decode,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--ctc-dataset", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--detector", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument("--box-padding", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-device", default="0")
    args = parser.parse_args()
    if not 0 <= args.detector_confidence <= 1 or not 0 <= args.box_padding <= 0.5:
        raise ValueError("invalid confidence or box padding")

    import torch
    from PIL import Image
    from torchvision import transforms

    checkpoint = torch.load(args.ctc_checkpoint, map_location="cpu")
    metadata = checkpoint["metadata"]
    model = build_numeric_crnn(pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(metadata["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(metadata["normalization"]["mean"], metadata["normalization"]["std"]),
    ])

    validation = [
        json.loads(line)
        for line in (Path(args.ctc_dataset) / "validation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    annotations = load_present_annotations(args.annotations)
    surface = []
    for row in validation:
        key = row_key(row)
        if key in annotations:
            surface.append({**row, **annotations[key]})
    if not surface:
        raise RuntimeError("CTC validation and manual number-region annotations do not intersect")

    variants = {"fixed": fixed_images(surface), "manual": manual_images(surface, args.box_padding)}
    for name, path in parse_detectors(args.detector).items():
        variants[name] = detector_images(
            surface, path, args.detector_confidence, args.box_padding, args.detector_device
        )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "surface_crops": len(surface),
        "surface_tracklets": len({(row["sequence"], row["gt_track_id"]) for row in surface}),
        "box_padding": args.box_padding,
        "detector_confidence": args.detector_confidence,
        "variants": {},
    }
    for name, items in variants.items():
        crop_rows, track_rows, metrics = evaluate_variant(
            model, transform, surface, items, args.batch_size, args.device
        )
        write_csv(output / f"{name}_crop_predictions.csv", crop_rows)
        write_csv(output / f"{name}_track_predictions.csv", track_rows)
        summary["variants"][name] = metrics
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def load_present_annotations(path):
    output = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("region_label") or "").strip().lower() != "present":
                continue
            key = row_key(row)
            output[key] = {
                "full_image": str(Path(row["crop_path"]).resolve()),
                "manual_box": tuple(float(row[name]) for name in ("xmin", "ymin", "xmax", "ymax")),
            }
    return output


def parse_detectors(values):
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid detector specification: {value}")
        name, path = value.split("=", 1)
        if not name or name in output or not Path(path).is_file():
            raise ValueError(f"invalid detector name or checkpoint: {value}")
        output[name] = str(Path(path).resolve())
    return output


def fixed_images(rows):
    return {row_key(row): {"image": row["image"], "detected": True, "confidence": 1.0} for row in rows}


def manual_images(rows, padding):
    output = {}
    for row in rows:
        output[row_key(row)] = {
            "image": crop_image(row["full_image"], row["manual_box"], padding),
            "detected": True, "confidence": 1.0,
        }
    return output


def detector_images(rows, checkpoint, confidence, padding, device):
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    paths = [row["full_image"] for row in rows]
    predictions = model.predict(paths, conf=confidence, device=device, verbose=False, stream=False)
    output = {}
    for row, prediction in zip(rows, predictions):
        key = row_key(row)
        if prediction.boxes is None or len(prediction.boxes) == 0:
            output[key] = {"image": None, "detected": False, "confidence": 0.0}
            continue
        best = int(prediction.boxes.conf.argmax().item())
        xyxy = prediction.boxes.xyxyn[best].tolist()
        score = float(prediction.boxes.conf[best])
        output[key] = {
            "image": crop_image(row["full_image"], tuple(xyxy), padding),
            "detected": True, "confidence": score,
        }
    return output


def crop_image(path, box, padding):
    from PIL import Image

    with Image.open(path) as source:
        width, height = source.size
        xmin, ymin, xmax, ymax = padded_box(box, padding)
        crop = source.convert("RGB").crop((
            round(xmin * width), round(ymin * height),
            round(xmax * width), round(ymax * height),
        ))
        return crop.copy()


def padded_box(box, fraction):
    xmin, ymin, xmax, ymax = box
    width, height = xmax - xmin, ymax - ymin
    return (
        max(0.0, xmin - width * fraction), max(0.0, ymin - height * fraction),
        min(1.0, xmax + width * fraction), min(1.0, ymax + height * fraction),
    )


def evaluate_variant(model, transform, surface, items, batch_size, device):
    import torch
    from PIL import Image

    rows_by_key = {row_key(row): row for row in surface}
    keys = [row_key(row) for row in surface if items[row_key(row)]["detected"]]
    frame_scores = {}
    frame_decodes = {}
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        tensors = []
        for key in batch_keys:
            image = items[key]["image"]
            if isinstance(image, (str, Path)):
                with Image.open(image) as source:
                    tensors.append(transform(source.convert("RGB")))
            else:
                tensors.append(transform(image))
        with torch.no_grad():
            logits = model(torch.stack(tensors).to(device)).cpu()
        for index, key in enumerate(batch_keys):
            values = logits[:, index, :]
            frame_scores[key] = candidate_log_probabilities(values)
            frame_decodes[key] = greedy_decode(values)

    crop_rows = []
    tracks = defaultdict(lambda: {"scores": [], "weights": [], "truth": None})
    for row in surface:
        key = row_key(row)
        detected = items[key]["detected"]
        prediction, recognition_confidence = frame_decodes.get(key, (None, 0.0))
        truth = int(row["text"])
        crop_rows.append({
            "sequence": key[0], "gt_track_id": key[1], "frame": key[2],
            "gt_jersey": truth, "detected": detected,
            "detector_confidence": items[key]["confidence"],
            "prediction": prediction or "", "recognition_confidence": recognition_confidence,
            "correct": detected and prediction == str(truth),
        })
        track_key = key[:2]
        tracks[track_key]["truth"] = truth
        if detected:
            tracks[track_key]["scores"].append(frame_scores[key])
            tracks[track_key]["weights"].append(items[key]["confidence"])
    track_rows = []
    for key, track in sorted(tracks.items()):
        result = aggregate_frames(track["scores"], track["weights"])
        assigned = result["prediction"] is not None
        ranking = list(result["scores"].items())[:5]
        track_rows.append({
            "sequence": key[0], "gt_track_id": key[1], "gt_jersey": track["truth"],
            "assigned": assigned, "prediction": "" if not assigned else result["prediction"],
            "correct": assigned and result["prediction"] == track["truth"],
            "confidence": result["confidence"], "margin": result["margin"],
            "recognized_frames": len(track["scores"]),
            "gt_in_top5": str(track["truth"]) in dict(ranking),
        })
    detected_crops = sum(row["detected"] for row in crop_rows)
    assigned_tracks = sum(row["assigned"] for row in track_rows)
    return crop_rows, track_rows, {
        "crops": len(crop_rows), "detected_crops": detected_crops,
        "detection_coverage": ratio(detected_crops, len(crop_rows)),
        "crop_correct": sum(row["correct"] for row in crop_rows),
        "crop_accuracy_all": ratio(sum(row["correct"] for row in crop_rows), len(crop_rows)),
        "tracklets": len(track_rows), "assigned_tracklets": assigned_tracks,
        "track_coverage": ratio(assigned_tracks, len(track_rows)),
        "track_correct": sum(row["correct"] for row in track_rows),
        "track_accuracy_all": ratio(sum(row["correct"] for row in track_rows), len(track_rows)),
        "track_gt_in_top5": sum(row["gt_in_top5"] for row in track_rows),
        "track_gt_in_top5_rate": ratio(sum(row["gt_in_top5"] for row in track_rows), len(track_rows)),
    }


def row_key(row):
    return str(row["sequence"]), str(row["gt_track_id"]), int(float(row["frame"]))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
