#!/usr/bin/env python3
"""Plot accuracy_all for SAR vs CTC-primary, offline surface vs pipeline surface.

Two independent measurements of the same policy question:
- offline: aggregate.csv from scripts/aggregate_jersey_thesis_benchmark.py,
  ground-truth boxes/tracks (evaluation_outputs/jersey_policy_heldout_v1/benchmark)
- pipeline: summary.json from scripts/evaluate_ft_gsr.py, real detection and
  tracking, summed over the 7 held-out sequences for each arm.

The two surfaces use different units and denominators (gt_track_id vs
display_track_id, 750 vs 300 frames) and are never meant to match numerically
-- only the direction of the effect is comparable.

    python scripts/plot_jersey_policy_two_surfaces.py \
        --offline-aggregate evaluation_outputs/jersey_policy_heldout_v1/benchmark/aggregate.csv \
        --pipeline-dir evaluation_outputs/heldout_policy_pipeline_apply \
        --sequences SNGS-089 SNGS-091 SNGS-092 SNGS-093 SNGS-094 SNGS-095 SNGS-096 \
        --output fig_policy_two_surfaces.png
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def offline_accuracy(path, method):
    with open(path, newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    if method not in rows:
        raise SystemExit(f"{path} has no method {method!r}")
    return float(rows[method]["accuracy_all"])


def pipeline_accuracy(pipeline_dir, arm, sequences):
    correct = 0
    gt_visible = 0
    for sequence in sequences:
        summary = json.loads((Path(pipeline_dir) / arm / sequence / "summary.json").read_text())
        tracklet = summary["jersey"]["tracklet_level"]
        gt_visible += tracklet["gt_visible_tracklets"]
        correct += sum(
            1 for d in tracklet["decisions"]
            if d["gt_jersey"] is not None and d["pred_jersey"] == d["gt_jersey"]
        )
    if gt_visible == 0:
        raise SystemExit(f"no gt-visible tracklets found under {pipeline_dir}/{arm}")
    return correct / gt_visible, correct, gt_visible


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline-aggregate", required=True)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    offline_sar = offline_accuracy(args.offline_aggregate, "baseline")
    offline_ctc = offline_accuracy(args.offline_aggregate, "policy_ctc_primary")
    pipeline_sar, sar_correct, sar_total = pipeline_accuracy(args.pipeline_dir, "baseline", args.sequences)
    pipeline_ctc, ctc_correct, ctc_total = pipeline_accuracy(args.pipeline_dir, "ctc_primary", args.sequences)

    print(f"offline   SAR={offline_sar:.3f}  CTC={offline_ctc:.3f}")
    print(f"pipeline  SAR={pipeline_sar:.3f} ({sar_correct}/{sar_total})  "
          f"CTC={pipeline_ctc:.3f} ({ctc_correct}/{ctc_total})")

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    groups = ["offline (GT)", "pipeline"]
    sar_values = [offline_sar, pipeline_sar]
    ctc_values = [offline_ctc, pipeline_ctc]
    x = range(len(groups))
    width = 0.3
    bars_sar = ax.bar([i - width / 2 for i in x], sar_values, width, label="SAR", color="#7F8C8D")
    bars_ctc = ax.bar([i + width / 2 for i in x], ctc_values, width, label="CTC primario", color="#2E86AB")
    for bar, value in zip(bars_sar, sar_values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    for bar, value in zip(bars_ctc, ctc_values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("accuracy complessiva")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
