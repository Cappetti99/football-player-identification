#!/usr/bin/env python3
"""Simulate re-ranking candidate crops by clean-back score instead of legacy legibility.

`evaluation/gsr_jersey_ocr/run_eval.py` writes every sampled candidate crop to
disk (`--crops-per-tracklet`, default 20) under
`{ocr-run}/crops/{sequence}/track_{eval_id:06d}_gt_..._frame_....jpg`, but only
the top-5 chosen by the legacy ResNet34 legibility score are kept in
`ocr_diagnostics.json` (`selected_crops`) and therefore only those 5 ever reach
the number-region detector. The blind-tracklet review showed most of those
misses are front/side-facing crops the legibility score does not penalize.

This script does not change any runtime code or config. For a requested set of
tracklets it:
  1. globs all raw candidate crops already on disk (no re-extraction),
  2. scores them with the existing `jersey_back_yolo11s_cls_v1.pt` classifier
     (diagnostic use only, same as the previous hard-miss pose audit),
  3. re-ranks and takes the top `--top-k` by clean-back probability instead of
     legacy legibility,
  4. runs the same frozen region detector on this alternate top-k set,
  5. reports whether region coverage would have been recovered for tracklets
     that are currently blind (zero detections in the real run).

Pure offline simulation: does not touch `predictions.csv`, `metrics.json`, or
any production selection logic.
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
from scripts.evaluate_jersey_number_region_ctc_ocr_run import read_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument(
        "--tracks",
        required=True,
        help="comma-separated sequence::gt_track_id pairs to simulate",
    )
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--classifier-checkpoint-sha256", default=None)
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

    verify_checkpoint(args.classifier_checkpoint, args.classifier_checkpoint_sha256)
    verify_checkpoint(args.detector_checkpoint, args.detector_checkpoint_sha256)

    from ultralytics import YOLO

    classifier = YOLO(args.classifier_checkpoint)
    clean_back_index = resolve_clean_back_index(classifier.names)
    for start in range(0, len(candidates), args.batch_size):
        batch = candidates[start:start + args.batch_size]
        results = classifier.predict(
            [row["crop_path"] for row in batch],
            device=args.device,
            batch=args.batch_size,
            verbose=False,
            stream=False,
        )
        for row, result in zip(batch, results):
            row["clean_back_probability"] = float(result.probs.data[clean_back_index])
    del classifier

    reselected = []
    per_track_candidates = defaultdict(list)
    for row in candidates:
        per_track_candidates[(row["sequence"], row["gt_track_id"])].append(row)
    for rows in per_track_candidates.values():
        ranked = sorted(rows, key=lambda row: -row["clean_back_probability"])
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
            "reselected_mean_clean_back_probability": mean(
                [row["clean_back_probability"] for row in chosen]
            ),
            "best_clean_back_probability_in_full_pool": max(
                row["clean_back_probability"] for row in rows
            ),
        }

    recovered_tracks = [
        label for label, stats in per_track_summary.items() if stats["reselected_detected"] > 0
    ]
    summary = {
        "ocr_run": str(root),
        "top_k": args.top_k,
        "tracks_simulated": len(per_track_summary),
        "tracks_with_at_least_one_detection_after_reselection": len(recovered_tracks),
        "recovered_track_ids": sorted(recovered_tracks),
        "per_track": per_track_summary,
        "note": (
            "Compare 'reselected_detected' > 0 against the original run, where these "
            "tracks had zero detections at any confidence. Any track showing up here "
            "with reselected_detected > 0 would gain jersey-number evidence purely "
            "from choosing different candidate frames, with no detector retraining."
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


def resolve_clean_back_index(names):
    for index, name in names.items():
        if str(name).strip().lower() == "clean_back":
            return index
    raise ValueError(f"'clean_back' class not found in classifier names: {names}")


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


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else None


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
