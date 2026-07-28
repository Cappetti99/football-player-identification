#!/usr/bin/env python3
"""Plot correct-tracklet counts for SAR / policy A / policy B on the held-out.

Reads aggregate.csv produced by scripts/aggregate_jersey_thesis_benchmark.py
on the 7-sequence held-out (offline surface, GT tracks). Three simple bars:
baseline SAR, policy_fallback (A, conservative), policy_ctc_primary (B,
aggressive) -- correct tracklets out of 116.

    python scripts/plot_jersey_policy_offline_bars.py \
        --aggregate evaluation_outputs/jersey_policy_heldout_v1/benchmark/aggregate.csv \
        --output fig_policy_offline_bars.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "baseline": "SAR\n(attuale)",
    "policy_fallback": "A\nconservativo",
    "policy_ctc_primary": "B\nCTC primario",
}
ORDER = ["baseline", "policy_fallback", "policy_ctc_primary"]
COLORS = {"baseline": "#7F8C8D", "policy_fallback": "#8FC1DA", "policy_ctc_primary": "#2E86AB"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    with open(args.aggregate, newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}

    missing = [m for m in ORDER if m not in rows]
    if missing:
        raise SystemExit(f"{args.aggregate} is missing methods: {missing}")

    tracklets = rows[ORDER[0]]["tracklets"]
    print(f"tracklet totali: {tracklets}")
    for method in ORDER:
        row = rows[method]
        print(f"  {method:20s} corretti={row['correct']:>3s}  coverage={float(row['coverage']):.3f}  "
              f"accuracy_all={float(row['accuracy_all']):.3f}")

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    values = [int(rows[m]["correct"]) for m in ORDER]
    labels = [LABELS[m] for m in ORDER]
    colors = [COLORS[m] for m in ORDER]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=12)

    ax.set_ylim(0, 85)
    ax.set_ylabel(f"tracklet corretti su {tracklets}")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
