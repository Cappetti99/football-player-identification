#!/usr/bin/env python3
"""Plot the DetA/AssA/HOTA bar chart for the GSR detection/tracking pilot.

Reads aggregate.json produced by
scripts/aggregate_gsr_detection_tracking_benchmark.py on the 12-sequence,
9000-frame validation pilot. Uses the `display_*` metrics (after FT's
tracklet linker), not `raw_*` (ByteTrack alone before linking) -- the two
differ slightly and the file keeps both, so which one is plotted must be
explicit rather than implied.

    python scripts/plot_jersey_gsr_tracking_bottleneck.py \
        --aggregate evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1_aggregate/aggregate.json \
        --output fig_tracking_bottleneck.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--source", default="display", choices=["display", "raw"],
                         help="display = after FT tracklet linking (default); raw = ByteTrack alone")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    payload = json.loads(Path(args.aggregate).read_text())
    macro = payload["macro"]
    micro = payload["micro"]

    deta = macro[f"{args.source}_deta"]["mean"]
    assa = macro[f"{args.source}_assa"]["mean"]
    hota = macro[f"{args.source}_hota"]["mean"]
    mota = macro[f"{args.source}_mota"]["mean"]
    recall = macro["det_recall_50"]["mean"]
    ap50 = macro["ap50"]["mean"]
    ap75 = macro["ap75"]["mean"]
    switches = micro[f"{args.source}_id_switches"]
    fragments = micro[f"{args.source}_fragmentations"]

    print(f"source: {args.source} (after FT linking)" if args.source == "display" else "source: raw ByteTrack")
    print(f"sequence_count: {payload['sequence_count']}")
    print(f"DetA={deta:.3f}  AssA={assa:.3f}  HOTA={hota:.3f}  MOTA={mota:.3f}")
    print(f"recall@0.50={recall*100:.1f}%  AP50={ap50:.3f}  AP75={ap75:.3f}")
    print(f"ID switches={switches}  fragmentations={fragments}")

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    labels = ["DetA", "AssA", "HOTA", "MOTA"]
    values = [deta, assa, hota, mota]
    colors = ["#2E86AB", "#C0392B", "#7F8C8D", "#F0A202"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=11)

    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
