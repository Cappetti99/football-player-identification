#!/usr/bin/env python3
"""Diagnose the number-region detector coverage gap on a frozen/development OCR run.

`scripts/evaluate_jersey_number_region_ctc_ocr_run.py` reports an aggregate
`region_detection_coverage` but never persists which crops were missed. This
script re-runs *only* the region detector (no CTC recognizer, no accuracy
recomputation) over the same `selected_crops` extracted from an existing
`ocr_diagnostics.json`, and exports a per-crop CSV with detection outcome plus
intrinsic crop size, so the coverage gap can be attributed to crop
size/resolution rather than guessed at.

Does not touch `metrics.json` of the source run and does not re-score
accuracy. Must be pointed at the frozen or development OCR run, never at the
locked GSR test surface (`ctc_sjn_transfer_gsr_test_v1/shared_surface`).
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_jersey_number_region_ctc_ocr_run import (  # noqa: E402
    read_predictions,
    selected_crops,
)

FORBIDDEN_SURFACE_MARKERS = ("gsr_test", "shared_surface")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--detector-checkpoint-sha256", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector-confidence", type=float, default=0.25)
    parser.add_argument(
        "--sweep-thresholds",
        default="0.25,0.20,0.15,0.10,0.05",
        help="comma-separated confidence cutoffs evaluated in one inference pass "
        "(post-hoc filtering of a single low-floor prediction, not repeated inference)",
    )
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--detector-device", default="0")
    parser.add_argument("--size-bins", type=int, default=5)
    parser.add_argument(
        "--allow-test-surface",
        action="store_true",
        help="required to point this at a path containing gsr_test/shared_surface markers",
    )
    args = parser.parse_args()

    root = Path(args.ocr_run).resolve()
    if not args.allow_test_surface and any(marker in str(root) for marker in FORBIDDEN_SURFACE_MARKERS):
        parser.error(
            f"{root} looks like the locked GSR test surface. Re-running the detector "
            "over it is a re-evaluation of the frozen test, which the project rules "
            "forbid. Point --ocr-run at the frozen or development surface instead, "
            "or pass --allow-test-surface if this is intentional and already approved."
        )

    predictions = read_predictions(root / "predictions.csv")
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    selected = selected_crops(diagnostics, predictions)
    if not selected:
        raise RuntimeError("no selected crops found in ocr_diagnostics.json")

    quality_by_path = crop_quality_by_path(diagnostics)
    for row in selected:
        row["crop_quality"] = quality_by_path.get(row["crop_path"], 0.0)

    if args.detector_checkpoint_sha256:
        digest = sha256_file(args.detector_checkpoint)
        if digest.lower() != args.detector_checkpoint_sha256.lower():
            raise ValueError(f"detector checkpoint SHA-256 mismatch: {digest}")

    from PIL import Image
    from ultralytics import YOLO

    for row in selected:
        with Image.open(row["crop_path"]) as image:
            row["crop_width"], row["crop_height"] = image.size

    sweep_thresholds = sorted({float(value) for value in args.sweep_thresholds.split(",") if value.strip()})
    predict_floor = min([args.detector_confidence, *sweep_thresholds])

    detector = YOLO(args.detector_checkpoint)
    detected_by_path = {}
    for start in range(0, len(selected), args.detector_batch_size):
        batch = selected[start:start + args.detector_batch_size]
        results = detector.predict(
            [row["crop_path"] for row in batch],
            conf=predict_floor,
            device=args.detector_device,
            batch=args.detector_batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            index = int(result.boxes.conf.argmax().item())
            detected_by_path[row["crop_path"]] = {
                "detector_confidence": float(result.boxes.conf[index]),
                "region_xyxyn": [float(value) for value in result.boxes.xyxyn[index].tolist()],
            }

    for row in selected:
        hit = detected_by_path.get(row["crop_path"])
        row["detector_confidence"] = hit["detector_confidence"] if hit else None
        row["region_xyxyn"] = hit["region_xyxyn"] if hit else None
        row["detected"] = hit is not None and row["detector_confidence"] >= args.detector_confidence

    per_track = tracklet_breakdown(selected)
    size_bins = bucket_by_size(selected, args.size_bins)
    threshold_curve = {
        threshold: {
            "detected": sum(
                1 for row in selected
                if row["detector_confidence"] is not None and row["detector_confidence"] >= threshold
            ),
            "coverage": ratio(
                sum(
                    1 for row in selected
                    if row["detector_confidence"] is not None and row["detector_confidence"] >= threshold
                ),
                len(selected),
            ),
        }
        for threshold in sweep_thresholds
    }
    hard_miss = sum(1 for row in selected if row["detector_confidence"] is None)

    summary = {
        "ocr_run": str(root),
        "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
        "detector_confidence": args.detector_confidence,
        "selected_crops": len(selected),
        "detected_crops": sum(row["detected"] for row in selected),
        "region_detection_coverage": ratio(sum(row["detected"] for row in selected), len(selected)),
        "tracklets": len(per_track),
        "fully_blind_tracklets": sum(1 for t in per_track.values() if t["detected"] == 0),
        "crop_min_side_stats": {
            "detected": describe([min(r["crop_width"], r["crop_height"]) for r in selected if r["detected"]]),
            "missed": describe([min(r["crop_width"], r["crop_height"]) for r in selected if not r["detected"]]),
        },
        "by_min_side_bin": size_bins,
        "crop_quality_stats": {
            "detected": describe([r["crop_quality"] for r in selected if r["detected"]]),
            "missed": describe([r["crop_quality"] for r in selected if not r["detected"]]),
        },
        "confidence_threshold_curve": threshold_curve,
        "hard_miss_crops": hard_miss,
        "hard_miss_rate": ratio(hard_miss, len(selected)),
    }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "region_detection_coverage.csv", selected)
    (output / "region_detection_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\ncsv={output / 'region_detection_coverage.csv'}")
    print(f"summary={output / 'region_detection_coverage_summary.json'}")


def crop_quality_by_path(diagnostics):
    tracklets = diagnostics.get("tracklets") if isinstance(diagnostics, dict) else None
    records = tracklets.values() if isinstance(tracklets, dict) else (diagnostics.values() if isinstance(diagnostics, dict) else [])
    output = {}
    for diagnostic in records:
        if not isinstance(diagnostic, dict):
            continue
        for crop in diagnostic.get("selected_crops", []):
            path = str(crop.get("crop_path") or "")
            if path:
                output[path] = float(crop.get("crop_quality", 0.0) or 0.0)
    return output


def tracklet_breakdown(selected):
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["sequence"], row["gt_track_id"])].append(row)
    out = {}
    for key, rows in grouped.items():
        out[f"{key[0]}::{key[1]}"] = {
            "selected": len(rows),
            "detected": sum(row["detected"] for row in rows),
        }
    return out


def bucket_by_size(selected, num_bins):
    if not selected or num_bins <= 0:
        return {}
    sides = sorted(min(row["crop_width"], row["crop_height"]) for row in selected)
    edges = [sides[int(round(i * (len(sides) - 1) / num_bins))] for i in range(num_bins + 1)]
    grouped = defaultdict(lambda: {"selected": 0, "detected": 0})
    for row in selected:
        side = min(row["crop_width"], row["crop_height"])
        bin_index = bin_for(side, edges)
        bucket = grouped[bin_index]
        bucket["selected"] += 1
        bucket["detected"] += int(row["detected"])
    return {
        f"bin_{index}_[{edges[index]}-{edges[index + 1]}]px": {
            **stats,
            "coverage": ratio(stats["detected"], stats["selected"]),
        }
        for index, stats in sorted(grouped.items())
    }


def bin_for(value, edges):
    for index in range(len(edges) - 1):
        if value <= edges[index + 1] or index == len(edges) - 2:
            return index
    return len(edges) - 2


def describe(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": values[int(0.25 * (len(values) - 1))],
        "p75": values[int(0.75 * (len(values) - 1))],
        "min": values[0],
        "max": values[-1],
    }


def ratio(a, b):
    return a / b if b else 0.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()
