#!/usr/bin/env python3
"""Create conservative crop-level utility pseudo-labels from a GSR OCR run."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {None, "", "None", "null", "-1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-source-confidence", type=float, default=0.20)
    parser.add_argument("--min-agreeing-sources", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 <= args.min_source_confidence <= 1.0:
        raise ValueError("--min-source-confidence must be between 0 and 1")
    if args.min_agreeing_sources < 1:
        raise ValueError("--min-agreeing-sources must be at least 1")

    run = Path(args.ocr_run)
    predictions = load_predictions(run / "predictions.csv")
    diagnostics = json.loads((run / "ocr_diagnostics.json").read_text(encoding="utf-8"))
    rows = build_pseudolabels(
        diagnostics,
        predictions,
        min_source_confidence=args.min_source_confidence,
        min_agreeing_sources=args.min_agreeing_sources,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "crops.csv", rows)
    write_jsonl(output / "crops.jsonl", rows)
    summary = summarize(rows, args)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_pseudolabels(diagnostics, predictions, min_source_confidence, min_agreeing_sources):
    output = []
    for diagnostic in (diagnostics.get("tracklets") or {}).values():
        eval_id = str(diagnostic.get("display_track_id"))
        prediction = predictions.get(eval_id)
        if prediction is None:
            continue
        detections_by_crop = defaultdict(list)
        for detection in diagnostic.get("detections", []):
            crop_path = str(detection.get("crop_path") or "")
            if crop_path:
                detections_by_crop[crop_path].append(detection)
        selected = {
            str(row.get("crop_path")): row
            for row in diagnostic.get("selected_crops", [])
            if row.get("crop_path")
        }
        for crop_path, crop in sorted(selected.items(), key=lambda item: (integer(item[1].get("frame")) or 0, item[0])):
            source_rows = source_winners(detections_by_crop.get(crop_path, []))
            reliable = [
                row for row in source_rows
                if row["confidence"] >= float(min_source_confidence)
            ]
            reliable_numbers = {row["number"] for row in reliable if row["number"] is not None}
            label = "ignore"
            reason = "insufficient_independent_agreement"
            predicted_number = None
            if len(reliable) >= int(min_agreeing_sources) and len(reliable_numbers) == 1:
                predicted_number = next(iter(reliable_numbers))
                single_source = int(min_agreeing_sources) == 1
                if predicted_number == prediction["gt_jersey"]:
                    label = "positive"
                    reason = (
                        "single_source_matches_gt"
                        if single_source
                        else "independent_sources_agree_with_gt"
                    )
                else:
                    label = "hard_negative"
                    reason = (
                        "single_source_disagrees_with_gt"
                        if single_source
                        else "independent_sources_agree_wrong"
                    )
            elif len(reliable_numbers) > 1:
                reason = "independent_sources_disagree"
            output.append({
                "sequence": prediction["sequence"],
                "eval_track_id": eval_id,
                "gt_track_id": prediction["gt_track_id"],
                "gt_jersey": prediction["gt_jersey"],
                "frame": integer(crop.get("frame")) or 0,
                "crop_path": crop_path,
                "crop_quality": floating(crop.get("crop_quality")),
                "pseudo_label": label,
                "pseudo_reason": reason,
                "pseudo_number": predicted_number,
                "reliable_sources": len(reliable),
                "source_winners": json.dumps(source_rows, ensure_ascii=False, sort_keys=True),
                "min_reliable_confidence": (
                    min(row["confidence"] for row in reliable) if reliable else None
                ),
            })
    return sorted(output, key=lambda row: (row["sequence"], row["gt_track_id"], row["frame"], row["crop_path"]))


def source_winners(detections):
    by_source_number = defaultdict(lambda: defaultdict(list))
    for row in detections:
        source = str(row.get("source") or "unknown")
        if source == "template":
            continue
        number = integer(row.get("number"))
        if number is None:
            continue
        by_source_number[source][number].append(floating(row.get("confidence")))
    output = []
    for source, by_number in sorted(by_source_number.items()):
        candidates = [
            {
                "source": source,
                "number": number,
                # Multiple preprocessing variants are correlated.  The best
                # confidence represents the source without creating extra votes.
                "confidence": max(confidences),
                "variants": len(confidences),
            }
            for number, confidences in by_number.items()
        ]
        output.append(max(candidates, key=lambda row: (row["confidence"], -row["number"])))
    return output


def summarize(rows, args):
    labels = Counter(row["pseudo_label"] for row in rows)
    reasons = Counter(row["pseudo_reason"] for row in rows)
    return {
        "crops": len(rows),
        "tracklets": len({(row["sequence"], row["gt_track_id"]) for row in rows}),
        "sequences": len({row["sequence"] for row in rows}),
        "labels": dict(labels),
        "label_rates": {key: value / len(rows) if rows else 0.0 for key, value in labels.items()},
        "reasons": dict(reasons),
        "min_source_confidence": args.min_source_confidence,
        "min_agreeing_sources": args.min_agreeing_sources,
        "supervision_mode": (
            "single_ocr_source_against_gt"
            if args.min_agreeing_sources == 1
            else "independent_ocr_source_agreement_against_gt"
        ),
        "independent_sources_required": args.min_agreeing_sources >= 2,
        "template_source_excluded": True,
        "no_ocr_output_policy": "ignore",
    }


def load_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["eval_track_id"]): {
            "sequence": str(row["sequence"]),
            "gt_track_id": str(row["gt_track_id"]),
            "gt_jersey": integer(row["gt_jersey_number"]),
        }
        for row in rows
    }


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def integer(value):
    if value in EMPTY:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def floating(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
