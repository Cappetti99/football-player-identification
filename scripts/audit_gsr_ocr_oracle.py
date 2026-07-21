#!/usr/bin/env python3
"""Measure whether FT OCR ever observes the correct GSR jersey per GT tracklet."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {None, "", "None", "null", "unknown", "-1", -1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", required=True)
    parser.add_argument("--jersey-ocr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    args = parser.parse_args()

    match_rows = read_csv(Path(args.matches))
    ocr = json.loads(Path(args.jersey_ocr).read_text())
    frame_gt, track_gt, current_predictions = build_match_indices(match_rows)
    crop_rows = build_crop_rows(ocr, frame_gt, args.min_confidence)
    audit_rows = build_tracklet_audit(track_gt, current_predictions, crop_rows)
    summary = summarize(audit_rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "tracklets.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    write_csv(output / "tracklets.csv", audit_rows)
    write_crop_csv(output / "crops.csv", crop_rows)
    print(json.dumps(summary, indent=2))


def build_match_indices(rows):
    frame_gt = {}
    jerseys_by_track = defaultdict(list)
    current_by_track = defaultdict(list)
    for row in rows:
        frame = integer(row.get("frame"))
        pred_track = text(row.get("pred_track_id"))
        gt_track = text(row.get("gt_track_id"))
        if frame is None or pred_track is None or gt_track is None:
            continue
        frame_gt[(frame, pred_track)] = gt_track
        jerseys_by_track[gt_track].append(jersey(row.get("gt_jersey")))
        current_by_track[gt_track].append(jersey(row.get("pred_jersey")))
    track_gt = {track: mode(values) for track, values in jerseys_by_track.items()}
    current = {
        track: mode([value for value in values if value is not None])
        for track, values in current_by_track.items()
    }
    return frame_gt, track_gt, current


def build_crop_rows(ocr, frame_gt, min_confidence):
    grouped = defaultdict(list)
    diagnostics = ocr.get("tracklets") or {}
    for diagnostic in diagnostics.values():
        display_id = text(diagnostic.get("display_track_id"))
        for detection in diagnostic.get("detections") or []:
            frame = integer(detection.get("frame"))
            confidence = number(detection.get("confidence"))
            value = jersey(detection.get("number"))
            if frame is None or display_id is None or value is None:
                continue
            if confidence is None or confidence < min_confidence:
                continue
            gt_track = frame_gt.get((frame, display_id))
            crop_key = text(detection.get("crop_path")) or f"frame:{frame}"
            grouped[(display_id, crop_key, frame)].append((gt_track, detection))

    output = []
    for (display_id, crop_path, frame), tagged_detections in grouped.items():
        gt_track = mode(gt_track for gt_track, _ in tagged_detections)
        detections = [detection for _, detection in tagged_detections]
        scores = defaultdict(float)
        counts = Counter()
        sources = defaultdict(set)
        variants = defaultdict(set)
        qualities = []
        for detection in detections:
            value = jersey(detection.get("number"))
            confidence = max(0.0, number(detection.get("confidence")) or 0.0)
            weight = max(0.0, number(detection.get("vote_weight")) or 1.0)
            scores[value] += max(0.01, confidence) * weight
            counts[value] += 1
            sources[value].add(text(detection.get("source")) or "unknown")
            variants[value].add(text(detection.get("variant")) or "unknown")
            quality = number(detection.get("crop_quality"))
            if quality is not None:
                qualities.append(quality)
        ranked = sorted(scores, key=lambda value: (scores[value], counts[value]), reverse=True)
        winner = ranked[0]
        total = sum(scores.values())
        winner_confidence = scores[winner] / total if total else 0.0
        agreement = counts[winner]
        source_agreement = len(sources[winner])
        crop_quality = sum(qualities) / len(qualities) if qualities else 0.0
        selection_score = (
            winner_confidence
            + 0.08 * math.log1p(agreement)
            + 0.08 * max(0, source_agreement - 1)
            + 0.15 * crop_quality
        )
        output.append({
            "display_track_id": display_id,
            "gt_track_id": gt_track,
            "frame": frame,
            "crop_path": crop_path,
            "winner": winner,
            "winner_confidence": winner_confidence,
            "agreement": agreement,
            "source_agreement": source_agreement,
            "sources": sorted(sources[winner]),
            "variants": sorted(variants[winner]),
            "crop_quality": crop_quality,
            "selection_score": selection_score,
            "all_candidate_numbers": ranked,
            "raw_detection_count": len(detections),
        })
    return sorted(
        output,
        key=lambda row: (
            row["display_track_id"], row["frame"], str(row["gt_track_id"] or "")
        ),
    )


def build_tracklet_audit(track_gt, current_predictions, crop_rows):
    by_track = defaultdict(list)
    for row in crop_rows:
        by_track[row["gt_track_id"]].append(row)
    output = []
    for gt_track, gt_jersey in sorted(track_gt.items(), key=lambda item: str(item[0])):
        if gt_jersey is None:
            continue
        crops = by_track.get(gt_track, [])
        ranked = sorted(crops, key=lambda row: row["selection_score"], reverse=True)
        correct_candidates = [
            row for row in crops
            if gt_jersey in row["all_candidate_numbers"]
        ]
        correct_winners = [row for row in crops if row["winner"] == gt_jersey]
        current = current_predictions.get(gt_track)
        output.append({
            "gt_track_id": gt_track,
            "gt_jersey": gt_jersey,
            "current_prediction": current,
            "current_correct": current == gt_jersey,
            "crop_count": len(crops),
            "oracle_candidate_available": bool(correct_candidates),
            "oracle_crop_winner_available": bool(correct_winners),
            "correct_candidate_crop_count": len(correct_candidates),
            "correct_winner_crop_count": len(correct_winners),
            "first_correct_frame": min((row["frame"] for row in correct_winners), default=None),
            "best_correct_score": max((row["selection_score"] for row in correct_winners), default=None),
            "best_wrong_score": max((row["selection_score"] for row in crops if row["winner"] != gt_jersey), default=None),
            "top1_correct": bool(ranked and ranked[0]["winner"] == gt_jersey),
            "top3_contains_correct": any(row["winner"] == gt_jersey for row in ranked[:3]),
            "top5_contains_correct": any(row["winner"] == gt_jersey for row in ranked[:5]),
            "top10_contains_correct": any(row["winner"] == gt_jersey for row in ranked[:10]),
            "recoverable_current_error": current != gt_jersey and bool(correct_winners),
            "best_crop_frame": ranked[0]["frame"] if ranked else None,
            "best_crop_prediction": ranked[0]["winner"] if ranked else None,
            "best_crop_path": ranked[0]["crop_path"] if ranked else None,
        })
    return output


def summarize(rows):
    total = len(rows)
    current_correct = sum(row["current_correct"] for row in rows)
    oracle_candidate = sum(row["oracle_candidate_available"] for row in rows)
    oracle_winner = sum(row["oracle_crop_winner_available"] for row in rows)
    return {
        "visible_gt_tracklets": total,
        "current_correct": current_correct,
        "current_accuracy": ratio(current_correct, total),
        "oracle_any_raw_candidate": oracle_candidate,
        "oracle_any_raw_candidate_accuracy": ratio(oracle_candidate, total),
        "oracle_any_crop_winner": oracle_winner,
        "oracle_any_crop_winner_accuracy": ratio(oracle_winner, total),
        "recoverable_current_errors": sum(row["recoverable_current_error"] for row in rows),
        "top1_selection_correct": sum(row["top1_correct"] for row in rows),
        "top1_selection_accuracy": ratio(sum(row["top1_correct"] for row in rows), total),
        "top3_selection_contains_correct": sum(row["top3_contains_correct"] for row in rows),
        "top3_selection_recall": ratio(sum(row["top3_contains_correct"] for row in rows), total),
        "top5_selection_contains_correct": sum(row["top5_contains_correct"] for row in rows),
        "top5_selection_recall": ratio(sum(row["top5_contains_correct"] for row in rows), total),
        "top10_selection_contains_correct": sum(row["top10_contains_correct"] for row in rows),
        "top10_selection_recall": ratio(sum(row["top10_contains_correct"] for row in rows), total),
        "tracklets_without_any_correct_crop_winner": [
            row["gt_track_id"] for row in rows if not row["oracle_crop_winner_available"]
        ],
    }


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields: writer.writeheader(); writer.writerows(rows)


def write_crop_csv(path, rows):
    serializable = []
    for row in rows:
        item = dict(row)
        for key in ("sources", "variants", "all_candidate_numbers"):
            item[key] = json.dumps(item[key])
        serializable.append(item)
    write_csv(path, serializable)


def text(value): return None if value in EMPTY else str(value)
def integer(value):
    try: return int(float(value))
    except (TypeError, ValueError): return None
def number(value):
    try: return float(value)
    except (TypeError, ValueError): return None
def jersey(value):
    value = integer(value)
    return value if value is not None and 1 <= value <= 99 else None
def mode(values):
    values = list(values)
    return Counter(values).most_common(1)[0][0] if values else None
def ratio(a, b): return float(a / b) if b else None


if __name__ == "__main__":
    main()
