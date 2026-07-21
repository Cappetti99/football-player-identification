#!/usr/bin/env python3
"""Offline padding and preprocessing audit for Region-CTC crop artifacts."""

import argparse
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
)
from ft.features.jersey_region_ctc_audit import crop_region


PREPROCESSING = ("color", "gray", "clahe")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--padding", type=float, action="append", dest="paddings")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.paddings = args.paddings or [0.10, 0.25, 0.40]
    if any(value < 0 or value > 0.5 for value in args.paddings):
        parser.error("--padding must be between 0 and 0.5")
    if args.upscale < 1 or args.batch_size < 1:
        parser.error("--upscale and --batch-size must be positive")
    return args


def preprocess_region(image, method, upscale):
    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.asarray(image.convert("RGB"))
    if method == "color":
        processed = rgb
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if method == "clahe":
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        elif method != "gray":
            raise ValueError(f"unknown preprocessing method: {method}")
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    height, width = processed.shape[:2]
    processed = cv2.resize(
        processed,
        (width * int(upscale), height * int(upscale)),
        interpolation=cv2.INTER_CUBIC,
    )
    return Image.fromarray(processed)


def variant_name(padding, method, upscale):
    return f"pad{padding:.2f}_{method}_x{int(upscale)}"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_items(audit):
    output = []
    for index, row in enumerate(audit.get("crops") or []):
        path = Path(str(row.get("crop_path") or ""))
        box = row.get("region_xyxyn")
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"crop row {index} has no valid region_xyxyn")
        output.append({
            "source_index": index,
            "display_track_id": int(row["display_track_id"]),
            "frame": int(row["frame"]),
            "crop_path": str(path.resolve()),
            "region_xyxyn": [float(value) for value in box],
            "detector_confidence": float(row.get("detector_confidence") or 0.0),
        })
    return output


def materialize(items, output, paddings, upscale):
    rows = []
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    for item in items:
        for padding in paddings:
            region = crop_region(item["crop_path"], item["region_xyxyn"], padding)
            if region is None:
                continue
            for method in PREPROCESSING:
                name = variant_name(padding, method, upscale)
                image = preprocess_region(region, method, upscale)
                filename = (
                    f"track_{item['display_track_id']:04d}_frame_{item['frame']:06d}_"
                    f"{name}.png"
                )
                path = image_dir / filename
                image.save(path)
                rows.append({
                    **item,
                    "padding": padding,
                    "preprocessing": method,
                    "upscale": upscale,
                    "variant": name,
                    "region_width": region.width,
                    "region_height": region.height,
                    "image_path": str(path.resolve()),
                    "image_relpath": str(path.relative_to(output)),
                })
    return rows


def recognize(rows, checkpoint_path, batch_size, device):
    import torch
    from PIL import Image
    from torchvision import transforms

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    model = build_numeric_crnn(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(tuple(metadata["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            metadata["normalization"]["mean"],
            metadata["normalization"]["std"],
        ),
    ])
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = []
        for row in batch:
            with Image.open(row["image_path"]) as source:
                images.append(transform(source.convert("RGB")))
        with torch.no_grad():
            logits = model(torch.stack(images).to(device)).cpu()
        for index, row in enumerate(batch):
            scores = candidate_log_probabilities(logits[:, index, :])
            ranking = sorted(scores.items(), key=lambda item: -item[1])
            row["ctc_top1"] = int(ranking[0][0])
            row["ctc_top1_probability"] = float(__import__("math").exp(ranking[0][1]))
            row["ctc_top5"] = [int(value) for value, _ in ranking[:5]]
            row["candidate_log_probabilities"] = scores
    return metadata


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["display_track_id"], row["variant"])].append(row)
    output = []
    for (track, variant), members in sorted(grouped.items()):
        result = aggregate_frames(
            [row["candidate_log_probabilities"] for row in members],
            [row["detector_confidence"] for row in members],
        )
        output.append({
            "display_track_id": track,
            "variant": variant,
            "padding": members[0]["padding"],
            "preprocessing": members[0]["preprocessing"],
            "upscale": members[0]["upscale"],
            "recognized_frames": len(members),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "winner_margin": result["margin"],
            "top5": [int(value) for value in list(result["scores"])[:5]],
        })
    return output


def write_csv(path, rows, excluded=()):
    rows = list(rows)
    fields = [key for key in rows[0] if key not in set(excluded)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_review(path, rows):
    unique = {}
    for row in rows:
        key = row["display_track_id"], row["frame"]
        unique.setdefault(key, row)
    review = []
    for row in unique.values():
        review.append({
            "display_track_id": row["display_track_id"],
            "frame": row["frame"],
            "crop_path": row["crop_path"],
            "readability": "",
            "jersey_number": "",
            "region_correct": "",
            "notes": "",
        })
    write_csv(path, review)


def write_html(path, rows, track_rows):
    predictions = {(row["display_track_id"], row["variant"]): row for row in track_rows}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["display_track_id"], row["frame"])].append(row)
    parts = ["""<!doctype html><meta charset='utf-8'><title>Region CTC preprocessing audit</title>
<style>body{font-family:sans-serif;max-width:1600px;margin:auto}.case{border-top:1px solid #aaa;padding:1rem 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:10px}figure{margin:0;border:1px solid #ddd;padding:6px}img{width:100%;height:180px;object-fit:contain;background:#222}figcaption{font-size:12px}</style>
<h1>Region CTC — padding and preprocessing audit</h1>
<p>Annotate <code>review.csv</code>: readability = readable/unreadable/uncertain; region_correct = yes/no/uncertain.</p>"""]
    for (track, frame), members in sorted(grouped.items()):
        parts.append(f"<section class='case'><h2>Track {track} — frame {frame}</h2><div class='grid'>")
        for row in sorted(members, key=lambda value: (value["padding"], value["preprocessing"])):
            track_result = predictions[(track, row["variant"])]
            parts.append(
                "<figure><img src='{src}'><figcaption>{variant}<br>"
                "region={width}x{height}; crop prediction={crop} ({prob:.3f})<br>"
                "track prediction={track_pred} ({track_conf:.3f}); top5={top5}"
                "</figcaption></figure>".format(
                    src=html.escape(row["image_relpath"]),
                    variant=html.escape(row["variant"]),
                    width=row["region_width"], height=row["region_height"],
                    crop=row["ctc_top1"], prob=row["ctc_top1_probability"],
                    track_pred=track_result["prediction"],
                    track_conf=track_result["confidence"],
                    top5=html.escape(str(track_result["top5"])),
                )
            )
        parts.append("</div></section>")
    path.write_text("".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    audit_path = Path(args.audit_json).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    items = load_items(audit)
    if not items:
        raise RuntimeError("audit contains no detected region crops")
    rows = materialize(items, output, args.paddings, args.upscale)
    metadata = recognize(rows, args.ctc_checkpoint, args.batch_size, args.device)
    track_rows = aggregate(rows)
    write_csv(
        output / "crop_predictions.csv",
        rows,
        excluded=("candidate_log_probabilities",),
    )
    write_csv(output / "track_predictions.csv", track_rows)
    write_review(output / "review.csv", rows)
    write_html(output / "index.html", rows, track_rows)
    summary = {
        "source_audit": str(audit_path),
        "source_audit_sha256": sha256(audit_path),
        "ctc_checkpoint": str(Path(args.ctc_checkpoint).resolve()),
        "ctc_checkpoint_sha256": sha256(args.ctc_checkpoint),
        "source_crops": len(items),
        "variants_per_crop": len(args.paddings) * len(PREPROCESSING),
        "variant_crops": len(rows),
        "track_variant_predictions": len(track_rows),
        "paddings": args.paddings,
        "preprocessing": list(PREPROCESSING),
        "upscale": args.upscale,
        "model_image_size": metadata["image_size"],
        "correctness_requires_manual_review": True,
        "review_csv": str(output / "review.csv"),
        "html": str(output / "index.html"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
