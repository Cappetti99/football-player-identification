#!/usr/bin/env python3
"""Break down region_detection_coverage.csv by tracklet: hard-miss vs soft-miss vs detected.

Reads the CSV produced by audit_jersey_number_region_detector_coverage.py and
answers: are the 25% hard-miss crops concentrated on a few "blind" tracklets
(low practical impact once one usable crop exists per track) or spread evenly
across most tracklets (structural gap touching most tracks)?

Pure CSV read, zero inference, zero new labeling.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="region_detection_coverage.csv path")
    parser.add_argument("--detected-threshold", type=float, default=0.25)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("sequence", ""), row.get("gt_track_id", ""))
        grouped[key].append(row)

    per_track = {}
    for key, items in grouped.items():
        confidences = [to_float(row.get("detector_confidence")) for row in items]
        hard_miss = sum(1 for value in confidences if value is None)
        detected = sum(1 for value in confidences if value is not None and value >= args.detected_threshold)
        soft_only = len(items) - hard_miss - detected
        per_track[f"{key[0]}::{key[1]}"] = {
            "selected": len(items),
            "hard_miss": hard_miss,
            "soft_only": soft_only,
            "detected": detected,
            "all_hard_miss": hard_miss == len(items),
        }

    tracklets = len(per_track)
    blind_at_threshold = [key for key, stats in per_track.items() if stats["detected"] == 0]
    all_hard_miss_tracklets = [key for key, stats in per_track.items() if stats["all_hard_miss"]]
    recoverable_by_lower_threshold = [
        key for key, stats in per_track.items() if stats["detected"] == 0 and stats["soft_only"] > 0
    ]

    total_selected = sum(stats["selected"] for stats in per_track.values())
    total_hard_miss = sum(stats["hard_miss"] for stats in per_track.values())
    hard_miss_in_blind = sum(
        stats["hard_miss"] for key, stats in per_track.items() if stats["detected"] == 0
    )
    hard_miss_in_nonblind = total_hard_miss - hard_miss_in_blind

    summary = {
        "tracklets": tracklets,
        "total_selected_crops": total_selected,
        "total_hard_miss_crops": total_hard_miss,
        "blind_tracklets_at_threshold": len(blind_at_threshold),
        "tracklets_fully_hard_miss": len(all_hard_miss_tracklets),
        "tracklets_blind_but_recoverable_by_lower_threshold": len(recoverable_by_lower_threshold),
        "hard_miss_crops_inside_blind_tracklets": hard_miss_in_blind,
        "hard_miss_crops_inside_covered_tracklets": hard_miss_in_nonblind,
        "hard_miss_share_inside_blind_tracklets": ratio(hard_miss_in_blind, total_hard_miss),
        "blind_tracklet_ids": sorted(blind_at_threshold),
        "fully_hard_miss_tracklet_ids": sorted(all_hard_miss_tracklets),
        "recoverable_by_lower_threshold_tracklet_ids": sorted(recoverable_by_lower_threshold),
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"summary": summary, "per_track": per_track}, indent=2), encoding="utf-8"
        )
        print(f"\njson={args.output_json}")


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(a, b):
    return a / b if b else 0.0


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
