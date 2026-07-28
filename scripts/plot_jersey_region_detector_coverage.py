#!/usr/bin/env python3
"""Plot the detector-coverage comparison for the three frozen-set checkpoints.

Reads region_detection_coverage_summary.json produced by
scripts/audit_jersey_number_region_detector_coverage.py for each checkpoint,
matches them by `detector_checkpoint` path (not by directory name, which is
not a reliable label), and renders a grouped bar chart:
coverage@0.25 / coverage@0.05 / fully_blind_tracklets.

coverage@0.05 replaces hard-miss on purpose: hard_miss_rate and coverage@0.05
are complementary (they sum to ~1 by construction) and plotting both wastes a
series on redundant information. fully_blind_tracklets carries independent
information -- how many tracklets have zero detected crops at all, not just
what fraction of crops are missed.

    python scripts/plot_jersey_region_detector_coverage.py \
        --run "smoke 10 ep.=evaluation_outputs/jersey_number_region/frozen_region_coverage_gap_v3" \
        --run "full GSR 43 ep.=evaluation_outputs/jersey_number_region/frozen_region_coverage_gap_v4_full_v1" \
        --run "SJN->GSR 25 ep.=evaluation_outputs/jersey_number_region/frozen_region_coverage_gap_v5_sjn_to_gsr" \
        --output fig_detector_coverage.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_runs(values):
    runs = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --run {value!r}; expected LABEL=DIR")
        label, directory = value.split("=", 1)
        runs.append((label.strip(), Path(directory)))
    return runs


def load_summary(directory):
    path = directory / "region_detection_coverage_summary.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    payload = json.loads(path.read_text())
    curve = payload.get("confidence_threshold_curve")
    if not curve:
        raise SystemExit(
            f"{path} has no confidence_threshold_curve -- this run predates the "
            "threshold sweep and cannot provide coverage@0.05. Re-run the audit "
            "script, or pick a different run directory for this checkpoint."
        )
    return {
        "checkpoint": payload["detector_checkpoint"],
        "coverage_025": 100 * curve["0.25"]["coverage"],
        "coverage_005": 100 * curve["0.05"]["coverage"],
        "fully_blind": payload["fully_blind_tracklets"],
        "tracklets": payload["tracklets"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    runs = parse_runs(args.run)
    rows = [(label, load_summary(directory)) for label, directory in runs]

    seen = set()
    for label, row in rows:
        if row["checkpoint"] in seen:
            raise SystemExit(f"two --run entries point at the same checkpoint: {row['checkpoint']}")
        seen.add(row["checkpoint"])
        print(f"{label:16s} checkpoint={row['checkpoint']}")
        print(
            f"{'':16s} cov@0.25={row['coverage_025']:.1f}%  "
            f"cov@0.05={row['coverage_005']:.1f}%  "
            f"fully_blind={row['fully_blind']}/{row['tracklets']}"
        )

    labels = [label for label, _ in rows]
    cov25 = [row["coverage_025"] for _, row in rows]
    cov05 = [row["coverage_005"] for _, row in rows]
    blind_pct = [100 * row["fully_blind"] / row["tracklets"] for _, row in rows]
    blind_counts = [f"{row['fully_blind']}/{row['tracklets']}" for _, row in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = range(len(labels))
    width = 0.26
    bars25 = ax.bar([i - width for i in x], cov25, width, label="coverage@0.25", color="#2E86AB")
    bars05 = ax.bar(x, cov05, width, label="coverage@0.05", color="#BDC3C7")
    barsbl = ax.bar([i + width for i in x], blind_pct, width, label="tracklet ciechi (%)", color="#C0392B")

    for bar, value in zip(bars25, cov25):
        ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)
    for bar, value in zip(bars05, cov05):
        ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)
    for bar, value, count in zip(barsbl, blind_pct, blind_counts):
        ax.annotate(count, (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 85)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
