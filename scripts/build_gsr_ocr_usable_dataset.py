#!/usr/bin/env python3
"""Build sequence-disjoint crop labels for an OCR-usability ranker.

Ground truth is joined only in this offline builder.  The operational selector
never receives GT, OCR candidates or roster information.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV: sequence,split,scores,ocr,matches")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    entries = read_csv(manifest_path)
    rows = []
    sequence_summaries = []
    seen_sequences = set()
    for entry in entries:
        sequence = required(entry, "sequence")
        split = normalize_split(required(entry, "split"))
        if sequence in seen_sequences:
            raise ValueError(f"sequence appears more than once in manifest: {sequence}")
        seen_sequences.add(sequence)
        built = build_sequence(entry, manifest_path.parent, sequence, split)
        rows.extend(built)
        sequence_summaries.append(summarize_rows(sequence, split, built))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "crops.csv", rows)
    summary = {
        "target": "ocr_usable_exact_match",
        "unit": "selected FT display-track crop",
        "gt_usage": "offline_label_only",
        "rows": len(rows),
        "positive": sum(row["ocr_usable"] for row in rows),
        "negative": sum(not row["ocr_usable"] for row in rows),
        "splits": summarize_splits(rows),
        "sequences": sequence_summaries,
        "test_rows": sum(row["split"] == "test" for row in rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_sequence(entry, base_dir, sequence, split):
    scores_path = resolve(base_dir, required(entry, "scores"))
    ocr_path = resolve(base_dir, required(entry, "ocr"))
    matches_path = resolve(base_dir, required(entry, "matches"))
    scores = read_csv(scores_path)
    ocr = json.loads(ocr_path.read_text())
    matches = read_csv(matches_path)

    score_by_key = {}
    for row in scores:
        key = crop_key(row.get("display_track_id"), row.get("frame"), row.get("crop_path"))
        score_by_key[key] = row

    gt_by_frame_display = best_gt_matches(matches)
    selected = selected_ocr_crops(ocr.get("tracklets", {}))
    output = []
    for item in selected:
        key = crop_key(item["display_track_id"], item["frame"], item["crop_path"])
        score = score_by_key.get(key, {})
        gt = gt_by_frame_display.get((str(item["display_track_id"]), int(item["frame"])))
        gt_jersey = integer(gt.get("gt_jersey")) if gt else None
        candidates = sorted({integer(row.get("number")) for row in item["detections"]} - {None})
        exact = gt_jersey is not None and gt_jersey in candidates
        output.append({
            "sequence": sequence,
            "split": split,
            "display_track_id": int(item["display_track_id"]),
            "frame": int(item["frame"]),
            "crop_path": item["crop_path"],
            "legibility_score": floating(score.get("legibility_score")),
            "crop_quality": floating(score.get("crop_quality")),
            "pred_role": score.get("pred_role") or "unknown",
            "gt_track_id": gt.get("gt_track_id") if gt else None,
            "gt_jersey": gt_jersey,
            "match_iou": floating(gt.get("iou")) if gt else None,
            "ocr_usable": bool(exact),
            "label_reason": (
                "exact_candidate" if exact else
                "no_visible_gt_jersey" if gt_jersey is None else
                "no_ocr_candidate" if not candidates else
                "wrong_or_partial_candidate"
            ),
            "ocr_candidates": candidates,
            "ocr_candidate_count": len(candidates),
            "single_digit_prefix_error": bool(
                gt_jersey is not None and gt_jersey >= 10
                and any(candidate < 10 and str(gt_jersey).startswith(str(candidate)) for candidate in candidates)
                and not exact
            ),
            "source_scores": str(scores_path),
            "source_ocr": str(ocr_path),
            "source_matches": str(matches_path),
        })
    return output


def selected_ocr_crops(tracklets):
    output = []
    for key, track in tracklets.items():
        display = int(track.get("display_track_id", key))
        detections_by_crop = defaultdict(list)
        for row in track.get("aggregated_detections", []):
            detections_by_crop[(str(row.get("crop_path") or ""), integer(row.get("frame")) or 0)].append(row)
        for row in track.get("selected_crops", []):
            crop_path = str(row.get("crop_path") or "")
            frame = integer(row.get("frame")) or 0
            output.append({
                "display_track_id": display,
                "frame": frame,
                "crop_path": crop_path,
                "detections": detections_by_crop.get((crop_path, frame), []),
            })
    return output


def best_gt_matches(rows):
    output = {}
    for row in rows:
        display = str(row.get("pred_track_id") or "")
        frame = integer(row.get("frame"))
        if not display or frame is None:
            continue
        key = (display, frame)
        if key not in output or floating(row.get("iou")) > floating(output[key].get("iou")):
            output[key] = row
    return output


def summarize_rows(sequence, split, rows):
    return {
        "sequence": sequence, "split": split, "rows": len(rows),
        "positive": sum(row["ocr_usable"] for row in rows),
        "negative": sum(not row["ocr_usable"] for row in rows),
        "prefix_errors": sum(row["single_digit_prefix_error"] for row in rows),
    }


def summarize_splits(rows):
    output = {}
    for split in sorted({row["split"] for row in rows}):
        items = [row for row in rows if row["split"] == split]
        output[split] = {
            "rows": len(items),
            "sequences": len({row["sequence"] for row in items}),
            "positive": sum(row["ocr_usable"] for row in items),
            "negative": sum(not row["ocr_usable"] for row in items),
        }
    return output


def normalize_split(value):
    value = str(value).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(value, value)


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def crop_key(display, frame, path):
    return str(display), integer(frame) or 0, str(path or "")


def required(row, key):
    value = row.get(key)
    if value in {None, ""}:
        raise ValueError(f"manifest field is required: {key}")
    return str(value)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def integer(value):
    try: return int(float(value))
    except (TypeError, ValueError): return None


def floating(value):
    try: return float(value)
    except (TypeError, ValueError): return 0.0


if __name__ == "__main__":
    main()
