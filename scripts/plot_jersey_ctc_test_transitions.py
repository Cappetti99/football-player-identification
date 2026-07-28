#!/usr/bin/env python3
"""Plot the paired transitions of the CTC SJN->GSR test against the baseline.

Reads transition_summary.json and bootstrap.json produced by
scripts/aggregate_jersey_thesis_benchmark.py on the locked GSR test
(ctc_sjn_transfer_gsr_test_v1). Renders a single horizontal stacked bar:
unchanged_correct / recovered_correct / correct_to_wrong / (unchanged_wrong +
wrong_to_wrong, merged into one "errato" segment) / both_abstain.

    python scripts/plot_jersey_ctc_test_transitions.py \
        --transition-summary evaluation_outputs/jersey_thesis_benchmark_v1/ctc_sjn_transfer_gsr_test_v1/transition_summary.json \
        --bootstrap evaluation_outputs/jersey_thesis_benchmark_v1/ctc_sjn_transfer_gsr_test_v1/bootstrap.json \
        --comparison ctc_sjn_to_gsr \
        --output fig_ctc_test_transitions.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transition-summary", required=True)
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--comparison", required=True, help="key under comparisons in both JSON files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    transitions = json.loads(Path(args.transition_summary).read_text())
    bootstrap = json.loads(Path(args.bootstrap).read_text())

    counts = transitions["comparisons"][args.comparison]["counts"]
    p_value = transitions["comparisons"][args.comparison]["paired_binomial_p_value"]
    net_gain = transitions["comparisons"][args.comparison]["net_correct_gain"]

    segments = [
        ("corr. $\\to$ corr.", counts["unchanged_correct"], "#27AE60"),
        ("recuperati", counts["recovered_correct"], "#8FD9AE"),
        ("corr. $\\to$ err.", counts["correct_to_wrong"], "#C0392B"),
        ("errato", counts["unchanged_wrong"] + counts["wrong_to_wrong"], "#E8B4AC"),
        ("entrambi astenuti", counts["both_abstain"], "#BDC3C7"),
    ]
    total = sum(value for _, value, _ in segments)

    print(f"totale tracklet: {total}")
    for label, value, _ in segments:
        print(f"  {label:20s} {value}")
    print(f"guadagno netto: +{net_gain}")
    print(f"p-value binomiale esatto: {p_value:.3e}")

    delta = bootstrap["comparisons"][args.comparison]["accuracy_all"]
    print(
        f"bootstrap delta accuracy_all: {delta['mean_delta']:+.3f}  "
        f"IC95 [{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}]"
    )

    fig, ax = plt.subplots(figsize=(5.4, 2.0))
    left = 0
    for label, value, color in segments:
        ax.barh(0, value, left=left, color=color, height=0.6, edgecolor="white", linewidth=0.6)
        left += value
    ax.set_xlim(0, total)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.set_title(f"{total} tracklet del test", fontsize=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in segments]
    labels = [f"{label}: {value}" for label, value, _ in segments]
    ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0, -0.15),
        ncol=1, frameon=False, fontsize=9, handlelength=1.2, handleheight=1.2,
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
