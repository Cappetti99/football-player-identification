#!/usr/bin/env python3
"""Cross-check display_track_ids flagged as multi-cluster by
audit_tracklet_identity_splitter.py against real ground truth (Identity
Benchmark V1), to tell apart genuine identity mixing (different physical
players sharing one display_track_id, e.g. after a scene reset) from a
false-positive split caused only by visual variation (blur, occlusion,
lighting) of the *same* player.

For each flagged display_track_id, prints every ground-truth item that
references it and that item's gt_jersey_number/gt_player_id, plus the
scene_segment_id(s) involved -- so genuine mixing (>=2 distinct gt values
across segments) is visually obvious from real, independently-verified
labels, not just embedding geometry.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--display-track-ids", required=True, help="comma-separated, e.g. 4,49,92,99,120")
    parser.add_argument("--benchmark-manifest", default="evaluation_outputs/identity_benchmark_v1_full/benchmark_manifest.json")
    parser.add_argument("--ground-truth-csv", default="evaluation/identity_benchmark_v1_full/ground_truth.csv")
    args = parser.parse_args()

    targets = {int(x) for x in args.display_track_ids.split(",")}

    with open(args.ground_truth_csv, newline="", encoding="utf-8") as handle:
        ground_truth = {row["item_id"]: row for row in csv.DictReader(handle)}
    manifest = json.loads(Path(args.benchmark_manifest).read_text())
    units_by_id = {unit["item_id"]: unit for unit in manifest.get("identity_units", [])}

    by_track = defaultdict(list)
    for item_id, unit in units_by_id.items():
        if unit.get("video_id") != args.video_id:
            continue
        row = ground_truth.get(item_id)
        if row is None:
            continue
        for member in unit.get("members", []):
            scene_segment_id = member.get("scene_segment_id")
            for display_track_id in member.get("display_track_ids", []):
                if int(display_track_id) in targets:
                    by_track[int(display_track_id)].append({
                        "item_id": item_id,
                        "scene_segment_id": scene_segment_id,
                        "annotation_status": row.get("annotation_status"),
                        "gt_jersey_number": row.get("gt_jersey_number"),
                        "gt_player_id": row.get("gt_player_id"),
                    })

    for display_id in sorted(targets):
        entries = by_track.get(display_id, [])
        distinct_jersey = {e["gt_jersey_number"] for e in entries if e["annotation_status"] == "determinate" and e["gt_jersey_number"]}
        distinct_player = {e["gt_player_id"] for e in entries if e["annotation_status"] == "determinate" and e["gt_player_id"]}
        print(f"display_track_id={display_id} gt_entries={len(entries)} distinct_gt_jersey_numbers={sorted(distinct_jersey)} distinct_gt_player_ids={sorted(distinct_player)}")
        for e in entries:
            print(f"  {e}")


if __name__ == "__main__":
    main()
