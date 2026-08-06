#!/usr/bin/env python3
"""Plot the roster-reranking transition breakdown against manual ground truth.

Reads the JSON printed by scripts/evaluate_roster_rerank_ground_truth.py (save
it with `| tee roster_gt.json`). Four bars, one per transition, coloured by
whether the change helped or hurt -- the two middle bars are the ones the
roster constraint is responsible for.

    python scripts/evaluate_roster_rerank_ground_truth.py ... | tee /tmp/roster_gt.json
    python scripts/plot_roster_rerank_transitions.py \
        --result /tmp/roster_gt.json \
        --output fig_roster_transitions.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Order tells the story: unchanged-good, recovered, regressed, unchanged-bad.
ORDER = [
    ("correct_to_correct", "corr\n$\\to$corr", "#8FBF8F"),
    ("wrong_to_correct", "err\n$\\to$corr", "#4E9A4E"),
    ("correct_to_wrong", "corr\n$\\to$err", "#C0392B"),
    ("wrong_to_wrong", "err\n$\\to$err", "#D9A6A0"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True, help="JSON from evaluate_roster_rerank_ground_truth.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    payload = json.loads(Path(args.result).read_text())
    transitions = payload["transitions"]
    missing = [key for key, _, _ in ORDER if key not in transitions]
    if missing:
        raise SystemExit(f"{args.result} is missing transitions: {missing}")

    total = payload["assigned_tracks_evaluated"]
    if sum(transitions[key] for key, _, _ in ORDER) != total:
        raise SystemExit(
            "transitions do not sum to assigned_tracks_evaluated -- refusing to "
            "plot a breakdown that does not account for every track"
        )

    print(f"video                : {payload['video_id']}")
    print(f"track valutati       : {total}")
    print(f"ambigui esclusi      : {payload['ambiguous_rows_excluded']}")
    print(f"accuracy assegnata   : {payload['original_accuracy_assigned']:.4f} "
          f"-> {payload['reranked_accuracy_assigned']:.4f}")
    for key, _, _ in ORDER:
        print(f"  {key:20s}: {transitions[key]}")
    print(f"guadagno netto       : {payload['net_gain']}")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    values = [transitions[key] for key, _, _ in ORDER]
    labels = [label for _, label, _ in ORDER]
    colors = [color for _, _, color in ORDER]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=11)

    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel(f"track su {total}")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
