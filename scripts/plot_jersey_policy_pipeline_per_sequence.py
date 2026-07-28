#!/usr/bin/env python3
"""Plot correct-tracklet counts per sequence, SAR vs CTC-primary, on the pipeline surface.

Reads summary.json from scripts/evaluate_ft_gsr.py for each of the 7 held-out
sequences and both policy arms, produced by real detection and tracking (not
ground-truth tracks). Grouped bars, one pair per sequence.

    python scripts/plot_jersey_policy_pipeline_per_sequence.py \
        --pipeline-dir evaluation_outputs/heldout_policy_pipeline_apply \
        --sequences SNGS-089 SNGS-091 SNGS-092 SNGS-093 SNGS-094 SNGS-095 SNGS-096 \
        --output fig_policy_pipeline_per_sequence.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def correct_count(pipeline_dir, arm, sequence):
    summary = json.loads((Path(pipeline_dir) / arm / sequence / "summary.json").read_text())
    decisions = summary["jersey"]["tracklet_level"]["decisions"]
    return sum(1 for d in decisions if d["gt_jersey"] is not None and d["pred_jersey"] == d["gt_jersey"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    sar = [correct_count(args.pipeline_dir, "baseline", s) for s in args.sequences]
    ctc = [correct_count(args.pipeline_dir, "ctc_primary", s) for s in args.sequences]
    short_labels = [s.replace("SNGS-0", "") for s in args.sequences]

    wins = sum(1 for a, b in zip(sar, ctc) if b > a)
    print(f"{'sequenza':10s}{'SAR':>6s}{'CTC':>6s}")
    for label, a, b in zip(args.sequences, sar, ctc):
        print(f"{label:10s}{a:6d}{b:6d}")
    print(f"\nCTC vince in {wins} sequenze su {len(args.sequences)}")
    print(f"totale corretti: SAR={sum(sar)}  CTC={sum(ctc)}")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(args.sequences))
    width = 0.35
    bars_sar = ax.bar([i - width / 2 for i in x], sar, width, label="SAR (attuale)", color="#7F8C8D")
    bars_ctc = ax.bar([i + width / 2 for i in x], ctc, width, label="CTC primario", color="#2E86AB")
    for bar, value in zip(bars_sar, sar):
        ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)
    for bar, value in zip(bars_ctc, ctc):
        ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, max(ctc) + 3)
    ax.set_ylabel("tracklet corretti")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
