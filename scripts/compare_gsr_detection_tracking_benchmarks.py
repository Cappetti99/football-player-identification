#!/usr/bin/env python3
"""Paired comparison of two aggregated GSR detection/tracking benchmark runs."""

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np


METRICS = {
    "det_precision_50": "higher",
    "det_recall_50": "higher",
    "det_f1_50": "higher",
    "ap50": "higher",
    "display_hota": "higher",
    "display_deta": "higher",
    "display_assa": "higher",
    "display_idf1": "higher",
    "display_mota": "higher",
    "display_id_switches_per_1000_gt": "lower",
    "display_fragmentations_per_1000_gt": "lower",
    "raw_hota": "higher",
    "raw_idf1": "higher",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    baseline = read_rows(Path(args.baseline_dir) / "per_sequence.csv")
    candidate = read_rows(Path(args.candidate_dir) / "per_sequence.csv")
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "Paired comparison requires identical sequences; "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired_rows = []
    summary = {}
    for metric_index, (metric, direction) in enumerate(METRICS.items()):
        metric_rows = []
        for sequence in sorted(baseline):
            base_value = number(baseline[sequence].get(metric))
            candidate_value = number(candidate[sequence].get(metric))
            if base_value is None or candidate_value is None:
                continue
            raw_delta = candidate_value - base_value
            improvement = raw_delta if direction == "higher" else -raw_delta
            row = {
                "sequence": sequence,
                "metric": metric,
                "direction": direction,
                "baseline": base_value,
                "candidate": candidate_value,
                "raw_delta": raw_delta,
                "improvement": improvement,
            }
            paired_rows.append(row)
            metric_rows.append(row)
        summary[metric] = paired_summary(
            metric_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + metric_index,
        )

    write_csv(output / "paired_deltas.csv", paired_rows)
    report = {
        "baseline_dir": str(Path(args.baseline_dir).resolve()),
        "candidate_dir": str(Path(args.candidate_dir).resolve()),
        "sequence_count": len(baseline),
        "delta_convention": {
            "raw_delta": "candidate - baseline",
            "improvement": "positive always means better after applying metric direction",
        },
        "metrics": summary,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(
        output / "comparison_table.csv",
        [{"metric": metric, **values} for metric, values in summary.items()],
    )
    if not args.no_charts:
        render_charts(paired_rows, output / "charts")
    print(json.dumps(report, indent=2))


def paired_summary(rows, bootstrap_samples, seed):
    values = [row["improvement"] for row in rows]
    if not values:
        return {
            "n": 0, "mean_improvement": None, "median_improvement": None,
            "ci95_low": None, "ci95_high": None,
            "improved_sequences": 0, "tied_sequences": 0, "regressed_sequences": 0,
        }
    rng = random.Random(seed)
    means = []
    for _ in range(max(0, bootstrap_samples)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(float(np.mean(sample)))
    epsilon = 1e-12
    return {
        "n": len(values),
        "mean_improvement": float(np.mean(values)),
        "median_improvement": float(np.median(values)),
        "ci95_low": float(np.percentile(means, 2.5)) if means else None,
        "ci95_high": float(np.percentile(means, 97.5)) if means else None,
        "improved_sequences": sum(value > epsilon for value in values),
        "tied_sequences": sum(abs(value) <= epsilon for value in values),
        "regressed_sequences": sum(value < -epsilon for value in values),
    }


def render_charts(rows, chart_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for charts; pass --no-charts to skip") from error

    chart_dir.mkdir(parents=True, exist_ok=True)
    by_metric = {
        metric: [row for row in rows if row["metric"] == metric]
        for metric in METRICS
    }
    metrics = [metric for metric, items in by_metric.items() if items]
    means = [float(np.mean([row["improvement"] for row in by_metric[metric]])) for metric in metrics]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = ["#15803d" if value >= 0 else "#b91c1c" for value in means]
    axis.bar(np.arange(len(metrics)), means, color=colors)
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set_xticks(np.arange(len(metrics)))
    axis.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Mean paired improvement")
    axis.set_title("Candidate improvement over baseline (positive is better)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, chart_dir / "mean_paired_improvement")

    main_metrics = ["det_recall_50", "display_hota", "display_idf1", "display_mota"]
    sequences = sorted({row["sequence"] for row in rows})
    matrix = np.full((len(sequences), len(main_metrics)), np.nan)
    sequence_index = {sequence: index for index, sequence in enumerate(sequences)}
    metric_index = {metric: index for index, metric in enumerate(main_metrics)}
    for row in rows:
        if row["metric"] in metric_index:
            matrix[sequence_index[row["sequence"]], metric_index[row["metric"]]] = row["improvement"]
    limit = float(np.nanmax(np.abs(matrix))) if np.any(np.isfinite(matrix)) else 1.0
    limit = max(limit, 1e-6)
    figure, axis = plt.subplots(figsize=(7.5, max(4.5, len(sequences) * 0.25)))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(main_metrics)))
    axis.set_xticklabels(main_metrics, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(sequences)), sequences, fontsize=7)
    axis.set_title("Where the candidate improves or regresses")
    figure.colorbar(image, ax=axis, label="Paired improvement")
    figure.tight_layout()
    save_figure(figure, chart_dir / "per_sequence_improvement_heatmap")


def save_figure(figure, target):
    figure.savefig(target.with_suffix(".png"), dpi=160)
    figure.savefig(target.with_suffix(".svg"))
    import matplotlib.pyplot as plt
    plt.close(figure)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sequence"]: row for row in csv.DictReader(handle)}


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["metric"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value in {None, "", "None", "null"}:
        return None
    return float(value)


if __name__ == "__main__":
    main()
