#!/usr/bin/env python3
"""Aggregate region_detection_coverage.csv by sequence, not by tracklet.

The clean-back reselection simulation recovered zero tracklets even where the
pose classifier found strong clean-back candidates (up to 0.88 probability).
That rules out "wrong frame selected" as the explanation for those 12 blind
tracklets and raises a different hypothesis: a per-video domain gap (camera
zoom, resolution, broadcast style) rather than a per-track pose problem. If
true, whole sequences should show systematically lower coverage than others,
independent of which tracks are blind.

Pure CSV read, zero inference.
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
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    grouped = defaultdict(lambda: {"selected": 0, "detected": 0, "tracklets": set(), "min_sides": []})
    for row in rows:
        sequence = row.get("sequence", "")
        detected = str(row.get("detected", "")).lower() == "true"
        bucket = grouped[sequence]
        bucket["selected"] += 1
        bucket["detected"] += int(detected)
        bucket["tracklets"].add(row.get("gt_track_id", ""))
        width, height = to_float(row.get("crop_width")), to_float(row.get("crop_height"))
        if width is not None and height is not None:
            bucket["min_sides"].append(min(width, height))

    per_sequence = {
        sequence: {
            "selected_crops": stats["selected"],
            "detected_crops": stats["detected"],
            "coverage": ratio(stats["detected"], stats["selected"]),
            "tracklets": len(stats["tracklets"]),
            "mean_crop_min_side": mean(stats["min_sides"]),
        }
        for sequence, stats in grouped.items()
    }
    ordered = dict(sorted(per_sequence.items(), key=lambda item: item[1]["coverage"]))

    coverages = [stats["coverage"] for stats in per_sequence.values()]
    summary = {
        "sequences": len(per_sequence),
        "coverage_min": min(coverages) if coverages else None,
        "coverage_max": max(coverages) if coverages else None,
        "coverage_spread": (max(coverages) - min(coverages)) if coverages else None,
        "per_sequence_sorted_by_coverage_asc": ordered,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\njson={args.output_json}")


def ratio(a, b):
    return a / b if b else 0.0


def mean(values):
    return sum(values) / len(values) if values else None


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
