#!/usr/bin/env python3
"""Audit why the YOLO number-region detector misses crops in a CTC audit run.

Zero manual labeling: reuses the exact crop-selection logic from
`JerseyRegionCTCAuditor` (`collect_selected_crops`) to reconstruct the full set
of crops offered to the detector, diffs it against the crops that actually got
a detected region (`{video_id}_jersey_region_ctc_crops.csv`), and reports
whether misses concentrate in low crop-quality/selection-score crops
(structural: no informative frame) or spread across all quality levels
(detector/threshold issue worth a confidence sweep).

Inputs are the metadata artifacts already produced by a run with
`jersey_region_ctc_audit.enabled: true`:
  {video_id}_jersey_ocr.json           (tracklets -> selected_crops, pre-detector)
  {video_id}_jersey_region_ctc_audit.json (crops -> detected regions, configuration)
  {video_id}_jersey_frame_selection.csv   (selection_score/reason/rank by crop_path)

Does not run any model and does not mutate any artifact.
"""
import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.jersey_region_ctc_audit import collect_selected_crops  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Audit number-region detector coverage gap using existing artifacts only."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--run", required=True, help="run name under --artifacts-root")
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--quality-bins", type=int, default=5)
    parser.add_argument("--top", type=int, default=20, help="worst tracklets to print")
    args = parser.parse_args()

    meta = Path(args.artifacts_root) / args.run / "metadata"
    jersey_ocr = read_json(meta / f"{args.video_id}_jersey_ocr.json")
    region_audit = read_json(meta / f"{args.video_id}_jersey_region_ctc_audit.json")
    frame_selection_rows = read_csv(meta / f"{args.video_id}_jersey_frame_selection.csv")

    cfg = region_audit.get("configuration", {})
    max_crops_per_tracklet = int(cfg.get("max_crops_per_tracklet", 5))
    min_frame_gap = int(cfg.get("min_frame_gap", 5))

    selected = collect_selected_crops(
        jersey_ocr,
        frame_selection_rows=frame_selection_rows,
        max_crops_per_tracklet=max_crops_per_tracklet,
        min_frame_gap=min_frame_gap,
    )
    detected_rows = region_audit.get("crops", [])
    detected_paths = {str(row.get("crop_path")) for row in detected_rows}

    for row in selected:
        row["quality_score"] = (
            row["selection_score"] if row["selection_score"] is not None else row["crop_quality"]
        )
        row["detected"] = str(row["crop_path"]) in detected_paths

    missed = [row for row in selected if not row["detected"]]
    detected_selected = [row for row in selected if row["detected"]]

    reported_selected = int(region_audit.get("selected_crops", 0))
    reported_detected = int(region_audit.get("detected_regions", 0))
    if reported_selected and reported_selected != len(selected):
        print(
            f"WARNING: reconstructed selected={len(selected)} != reported selected_crops="
            f"{reported_selected}. Selection-logic drift or stale frame_selection_rows.",
            file=sys.stderr,
        )
    if reported_detected and reported_detected != len(detected_selected):
        print(
            f"WARNING: reconstructed detected-among-selected={len(detected_selected)} != "
            f"reported detected_regions={reported_detected}.",
            file=sys.stderr,
        )

    per_track = tracklet_breakdown(selected)
    reason_breakdown = breakdown_by(selected, "selection_reason")
    quality_bins = bucket_by_quality(selected, args.quality_bins)

    summary = {
        "video_id": args.video_id,
        "run": args.run,
        "max_crops_per_tracklet": max_crops_per_tracklet,
        "min_frame_gap": min_frame_gap,
        "selected_crops": len(selected),
        "detected_crops": len(detected_selected),
        "missed_crops": len(missed),
        "coverage": ratio(len(detected_selected), len(selected)),
        "fully_blind_tracklets": sum(1 for t in per_track.values() if t["detected"] == 0),
        "tracklets": len(per_track),
        "quality_score_stats": {
            "detected": describe([r["quality_score"] for r in detected_selected]),
            "missed": describe([r["quality_score"] for r in missed]),
        },
        "crop_quality_stats": {
            "detected": describe([r["crop_quality"] for r in detected_selected]),
            "missed": describe([r["crop_quality"] for r in missed]),
        },
        "by_selection_reason": reason_breakdown,
        "by_quality_bin": quality_bins,
    }

    out_json = meta / f"{args.video_id}_number_region_coverage_gap.json"
    out_csv = meta / f"{args.video_id}_number_region_coverage_gap.csv"
    write_json({"summary": summary, "missed_rows": missed, "tracklets": per_track}, out_json)
    write_csv(missed, out_csv)

    print(json.dumps(summary, indent=2))
    worst = sorted(per_track.items(), key=lambda item: (item[1]["detected"], -item[1]["selected"]))
    print("\nWorst tracklets (detected asc, selected desc):")
    for display_id, stats in worst[: max(0, args.top)]:
        print(
            f"display={display_id} selected={stats['selected']} detected={stats['detected']} "
            f"coverage={ratio(stats['detected'], stats['selected']):.2f} "
            f"mean_quality={stats['mean_quality']:.3f}"
        )
    print(f"\njson={out_json}\ncsv={out_csv}")


def tracklet_breakdown(selected):
    grouped = defaultdict(list)
    for row in selected:
        grouped[int(row["display_track_id"])].append(row)
    out = {}
    for display_id, rows in grouped.items():
        detected = sum(1 for row in rows if row["detected"])
        out[display_id] = {
            "selected": len(rows),
            "detected": detected,
            "mean_quality": statistics.fmean(row["quality_score"] for row in rows),
        }
    return out


def breakdown_by(selected, key):
    grouped = defaultdict(lambda: {"selected": 0, "detected": 0})
    for row in selected:
        bucket = grouped[str(row.get(key) or "unknown")]
        bucket["selected"] += 1
        bucket["detected"] += int(row["detected"])
    return {
        name: {**stats, "coverage": ratio(stats["detected"], stats["selected"])}
        for name, stats in sorted(grouped.items(), key=lambda item: -item[1]["selected"])
    }


def bucket_by_quality(selected, num_bins):
    if not selected or num_bins <= 0:
        return {}
    values = sorted(row["quality_score"] for row in selected)
    edges = [values[int(round(i * (len(values) - 1) / num_bins))] for i in range(num_bins + 1)]
    grouped = defaultdict(lambda: {"selected": 0, "detected": 0})
    for row in selected:
        bin_index = bin_for(row["quality_score"], edges)
        bucket = grouped[bin_index]
        bucket["selected"] += 1
        bucket["detected"] += int(row["detected"])
    return {
        f"bin_{index}_[{edges[index]:.3f}-{edges[index + 1]:.3f}]": {
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(rows, path):
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
                    key: json.dumps(value) if isinstance(value, (dict, list, set)) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()
