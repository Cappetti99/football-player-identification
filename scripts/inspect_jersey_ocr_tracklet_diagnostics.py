#!/usr/bin/env python3
"""Inspect legacy OCR crop-selection diagnostics for specific (sequence, gt_track_id) tracklets.

Answers: did a hard-miss tracklet in the number-region detector coverage audit
have few candidate frames to begin with (scarce visibility, structural limit),
or plenty of candidates that the detector still failed on (a genuine detector
generalization gap)?

Reads only `{ocr-run}/predictions.csv` and `{ocr-run}/ocr_diagnostics.json`,
both already produced by the frozen/development OCR run. No inference, no new
labeling, no mutation of any artifact.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_jersey_number_region_ctc_ocr_run import read_predictions  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument(
        "--tracks",
        required=True,
        help="comma-separated sequence::gt_track_id pairs, e.g. "
        "'SNGS-068::10,SNGS-068::13,SNGS-098::7'",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.ocr_run).resolve()
    predictions = read_predictions(root / "predictions.csv")
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    tracklets = diagnostics.get("tracklets") if isinstance(diagnostics, dict) else None
    records = tracklets.values() if isinstance(tracklets, dict) else diagnostics.values()
    diagnostic_by_eval = {
        str(record.get("display_track_id")): record for record in records if isinstance(record, dict)
    }

    requested = [parse_track(item) for item in args.tracks.split(",") if item.strip()]
    output = {}
    for sequence, gt_track_id in requested:
        label = f"{sequence}::{gt_track_id}"
        reference = predictions.get((sequence, gt_track_id))
        if reference is None:
            output[label] = {"status": "not_found_in_predictions_csv"}
            continue
        diagnostic = diagnostic_by_eval.get(str(reference["eval_track_id"]))
        if diagnostic is None:
            output[label] = {
                "status": "not_found_in_ocr_diagnostics",
                "eval_track_id": reference["eval_track_id"],
            }
            continue
        selection = diagnostic.get("crop_selection", {})
        output[label] = {
            "status": "ok",
            "eval_track_id": reference["eval_track_id"],
            "gt_jersey_number": reference["gt"],
            "available_crops": diagnostic.get("available_crops"),
            "usable_crops": diagnostic.get("usable_crops"),
            "selected_crops": len(diagnostic.get("selected_crops", [])),
            "crop_selection": selection,
            "raw_detection_count": diagnostic.get("raw_detection_count"),
            "segment_index": diagnostic.get("segment_index"),
        }

    interpretation = classify(output)
    print(json.dumps({"tracks": output, "interpretation": interpretation}, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"tracks": output, "interpretation": interpretation}, indent=2),
            encoding="utf-8",
        )
        print(f"\njson={args.output_json}")


def classify(output, scarce_threshold=8):
    scarce, generous = [], []
    for label, stats in output.items():
        if stats.get("status") != "ok":
            continue
        available = stats.get("available_crops") or 0
        (scarce if available <= scarce_threshold else generous).append(label)
    return {
        "scarce_candidate_frames": scarce,
        "generous_candidate_frames_still_missed": generous,
        "scarce_threshold_available_crops": scarce_threshold,
        "note": (
            "scarce_candidate_frames: few frames were ever available for this track "
            "(structural visibility limit, not a detector failure). "
            "generous_candidate_frames_still_missed: plenty of candidate frames existed "
            "and the detector still found nothing usable (points at a detector "
            "generalization gap on this track's visual conditions, worth a qualitative look)."
        ),
    }


def parse_track(item):
    sequence, gt_track_id = item.strip().split("::")
    return sequence, gt_track_id


if __name__ == "__main__":
    main()
