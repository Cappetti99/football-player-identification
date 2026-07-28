#!/usr/bin/env python3
"""Compare roster-reranking's jersey_number predictions against real,
manually-labeled ground truth (Identity Benchmark V1), instead of the
GSR-oracle ceiling measured earlier.

Joins:
  - evaluation/<ground_truth_dir>/ground_truth.csv (item_id -> gt_jersey_number)
  - evaluation_outputs/<benchmark_dir>/benchmark_manifest.json
    (item_id -> video_id, display_track_ids)
  - artifacts/costume-video/<run>/metadata/<video_id>_jersey_region_ctc_audit.json
    (display_track_id -> roster_rerank_preview: original_number, preview_number)

Reports, for the given video's determinate identity units: how many tracks
the CTC assigned a number to, how many were correct before vs after
roster-reranking, and the same correct_to_correct / correct_to_wrong /
wrong_to_correct / wrong_to_wrong transition breakdown used for the earlier
oracle-roster measurement -- but now against real human labels.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.jersey_region_ctc_audit import tracklet_key  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-csv", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--video-id", required=True)
    args = parser.parse_args()

    import csv
    with open(args.ground_truth_csv, newline="", encoding="utf-8") as handle:
        ground_truth = {row["item_id"]: row for row in csv.DictReader(handle)}

    manifest = json.loads(Path(args.benchmark_manifest).read_text())
    units_by_id = {unit["item_id"]: unit for unit in manifest.get("identity_units", [])}

    audit = json.loads(Path(args.audit_json).read_text())
    roster_preview = audit.get("roster_rerank_preview", {})

    evaluated = []
    for item_id, unit in units_by_id.items():
        if unit.get("video_id") != args.video_id:
            continue
        row = ground_truth.get(item_id)
        if row is None:
            continue
        if row.get("annotation_status") != "determinate":
            continue
        gt_jersey = row.get("gt_jersey_number")
        if not gt_jersey:
            continue
        track_keys = sorted({
            (display_track_id, member.get("scene_segment_id"))
            for member in unit.get("members", [])
            for display_track_id in member.get("display_track_ids", [])
        })
        for display_track_id, scene_segment_id in track_keys:
            key = tracklet_key(display_track_id, scene_segment_id)
            preview = roster_preview.get(key)
            if preview is None:
                # Fall back to the plain display_track_id: audit runs without
                # scene-segment info in player_rows key this way too.
                preview = roster_preview.get(str(display_track_id))
            if preview is None:
                continue
            evaluated.append({
                "item_id": item_id,
                "track_key": key,
                "gt_jersey_number": str(gt_jersey),
                "original_number": (
                    str(preview["original_number"]) if preview.get("original_number") is not None else None
                ),
                "preview_number": (
                    str(preview["preview_number"]) if preview.get("preview_number") is not None else None
                ),
                "changed": preview.get("changed", False),
                "reason": preview.get("reason"),
            })

    # A raw display_track_id can be reused across scene resets for different
    # physical players (this run uses resetbytetrack). If the same
    # display_track_id maps to more than one distinct gt_jersey_number, the
    # audit's per-track prediction (keyed only by display_track_id, with no
    # scene-segment disambiguation) cannot be attributed to a single player --
    # counting it against either ground truth would double/multi-count one
    # prediction as if it were several independent observations. Exclude
    # those ambiguous IDs rather than silently mis-scoring them.
    gt_by_track = {}
    for row in evaluated:
        gt_by_track.setdefault(row["track_key"], set()).add(row["gt_jersey_number"])
    ambiguous_ids = {track_id for track_id, values in gt_by_track.items() if len(values) > 1}

    unambiguous = [row for row in evaluated if row["track_key"] not in ambiguous_ids]
    assigned = [row for row in unambiguous if row["original_number"] is not None]
    original_correct = [row for row in assigned if row["original_number"] == row["gt_jersey_number"]]
    preview_correct = [row for row in assigned if row["preview_number"] == row["gt_jersey_number"]]

    transitions = {"correct_to_correct": 0, "correct_to_wrong": 0, "wrong_to_correct": 0, "wrong_to_wrong": 0}
    for row in assigned:
        was_correct = row["original_number"] == row["gt_jersey_number"]
        now_correct = row["preview_number"] == row["gt_jersey_number"]
        key = f"{'correct' if was_correct else 'wrong'}_to_{'correct' if now_correct else 'wrong'}"
        transitions[key] += 1

    changed_rows = [row for row in assigned if row["changed"]]
    result = {
        "video_id": args.video_id,
        "identity_units_matched": len(evaluated),
        "ambiguous_display_track_ids_excluded": sorted(ambiguous_ids),
        "ambiguous_rows_excluded": sum(1 for row in evaluated if row["track_key"] in ambiguous_ids),
        "assigned_tracks_evaluated": len(assigned),
        "original_accuracy_assigned": ratio(len(original_correct), len(assigned)),
        "reranked_accuracy_assigned": ratio(len(preview_correct), len(assigned)),
        "transitions": transitions,
        "net_gain": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
        "changed_rows": len(changed_rows),
        "changed_details": changed_rows,
    }
    print(json.dumps(result, indent=2))


def ratio(a, b):
    return a / b if b else 0.0


if __name__ == "__main__":
    main()
