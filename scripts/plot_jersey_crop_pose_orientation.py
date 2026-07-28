#!/usr/bin/env python3
"""Plot region-detector coverage broken down by crop orientation (back/front).

Reads crop_pose_orientation_summary.json produced by
scripts/audit_crop_pose_orientation.py: for each orientation bucket predicted
by a generic, zero-training COCO pose estimator, the fraction of crops in
that bucket where the region detector actually found the number.

    python scripts/plot_jersey_crop_pose_orientation.py \
        --summary evaluation_outputs/jersey_number_region/frozen_region_coverage_gap_v3/pose_orientation_v1/crop_pose_orientation_summary.json \
        --output fig_pose_orientation.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    payload = json.loads(Path(args.summary).read_text())
    summary = payload.get("summary", payload)
    counts = summary.get("orientation_counts")
    cross = summary.get("detected_vs_orientation")
    if not counts or not cross:
        raise SystemExit(
            f"{args.summary} has no orientation_counts / detected_vs_orientation -- "
            "this does not look like a crop_pose_orientation_summary.json with the "
            "expected structure."
        )

    print("pose_checkpoint:", summary.get("pose_checkpoint"))
    print(f"pose_detected: {summary.get('pose_detected')}/{summary.get('total_crops')}")

    rows = []
    for label in ("back", "front"):
        total = counts[label]
        detected = cross[label].get("True", 0)
        coverage = 100 * detected / total if total else 0.0
        rows.append((label, coverage, detected, total))
        print(f"  {label:6s} coverage={coverage:.1f}%  ({detected}/{total})")

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = range(len(rows))
    colors = ["#2E86AB", "#C0392B"]
    bars = ax.barh(list(y), [r[1] for r in rows], color=colors, height=0.55)
    for bar, (label, coverage, detected, total) in zip(bars, rows):
        ax.annotate(f"{coverage:.1f}%", (coverage, bar.get_y() + bar.get_height() / 2),
                    va="center", ha="left", fontsize=10, xytext=(4, 0), textcoords="offset points")

    ax.set_yticks(list(y))
    ax.set_yticklabels(["schiena" if r[0] == "back" else "frontale" for r in rows], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("coverage del detector di regione (%)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
