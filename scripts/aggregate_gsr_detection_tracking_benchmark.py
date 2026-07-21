#!/usr/bin/env python3
"""Aggregate per-sequence GSR metrics and render benchmark charts."""

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np


METRICS = {
    "det_precision_50": ("detection", "by_iou", "0.50", "precision"),
    "det_recall_50": ("detection", "by_iou", "0.50", "recall"),
    "det_f1_50": ("detection", "by_iou", "0.50", "f1"),
    "det_mean_iou_50": ("detection", "by_iou", "0.50", "mean_iou"),
    "det_precision_75": ("detection", "by_iou", "0.75", "precision"),
    "det_recall_75": ("detection", "by_iou", "0.75", "recall"),
    "det_f1_75": ("detection", "by_iou", "0.75", "f1"),
    "ap50": ("detection", "average_precision", "0.50", "ap"),
    "ap75": ("detection", "average_precision", "0.75", "ap"),
    "fp_per_frame": ("detection", "fp_per_frame"),
    "fn_per_frame": ("detection", "fn_per_frame"),
    "recall_small": ("detection", "by_bbox_size", "small", "recall"),
    "recall_medium": ("detection", "by_bbox_size", "medium", "recall"),
    "recall_large": ("detection", "by_bbox_size", "large", "recall"),
    "display_hota": ("tracking", "hota"),
    "display_deta": ("tracking", "deta"),
    "display_assa": ("tracking", "assa"),
    "display_mota": ("tracking", "mota"),
    "display_idf1": ("tracking", "idf1"),
    "display_id_switches_per_1000_gt": (
        "tracking", "id_switches_per_1000_gt"
    ),
    "display_fragmentations_per_1000_gt": (
        "tracking", "fragmentations_per_1000_gt"
    ),
    "display_mostly_tracked_rate": ("tracking", "mostly_tracked_rate"),
    "display_association_purity": ("tracking", "association_purity"),
    "raw_hota": ("tracking_raw", "hota"),
    "raw_deta": ("tracking_raw", "deta"),
    "raw_assa": ("tracking_raw", "assa"),
    "raw_mota": ("tracking_raw", "mota"),
    "raw_idf1": ("tracking_raw", "idf1"),
    "raw_id_switches_per_1000_gt": (
        "tracking_raw", "id_switches_per_1000_gt"
    ),
    "raw_fragmentations_per_1000_gt": (
        "tracking_raw", "fragmentations_per_1000_gt"
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    evaluation_root = Path(args.evaluation_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summaries = []
    rows = []
    for entry in manifest.get("sequences") or []:
        sequence = entry["sequence"]
        summary_path = evaluation_root / sequence / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing sequence evaluation: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append((sequence, summary))
        row = {"sequence": sequence}
        for name, path in METRICS.items():
            row[name] = nested(summary, path)
        rows.append(row)

    write_csv(output / "per_sequence.csv", rows)
    macro = {
        name: distribution_summary(
            [row[name] for row in rows if row.get(name) is not None],
            args.bootstrap_samples,
            args.seed + index,
        )
        for index, name in enumerate(METRICS)
    }
    aggregate = {
        "benchmark": manifest.get("benchmark"),
        "sequence_count": len(rows),
        "macro": macro,
        "micro": micro_summary([summary for _, summary in summaries]),
        "metric_definitions": {
            "macro": "unweighted distribution across sequences",
            "micro": "pooled count-based metrics where mathematically composable",
            "bootstrap_ci": "percentile CI from sequence-level bootstrap resampling",
            "hota": "TrackEval-compatible global-alignment HOTA averaged over IoU 0.05:0.95",
            "tracking": "FT display_track_id after linking",
            "tracking_raw": "raw tracker identity before FT display-track linking",
        },
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_metric_table(output / "metric_table.csv", macro)

    if not args.no_charts:
        render_charts(rows, summaries, evaluation_root, output / "charts")
    print(json.dumps(aggregate, indent=2))


def micro_summary(summaries):
    tp = sum(nested(item, ("detection", "true_positives")) or 0 for item in summaries)
    fp = sum(nested(item, ("detection", "false_positives")) or 0 for item in summaries)
    fn = sum(nested(item, ("detection", "false_negatives")) or 0 for item in summaries)
    gt = sum(nested(item, ("detection", "gt")) or 0 for item in summaries)
    frames = sum(nested(item, ("dataset", "frames")) or 0 for item in summaries)
    output = {
        "detection_precision_50": safe_ratio(tp, tp + fp),
        "detection_recall_50": safe_ratio(tp, tp + fn),
        "detection_f1_50": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "false_positives": fp,
        "false_negatives": fn,
        "fp_per_frame": safe_ratio(fp, frames),
        "fn_per_frame": safe_ratio(fn, frames),
    }
    for section, prefix in (("tracking", "display"), ("tracking_raw", "raw")):
        idtp = sum(nested(item, (section, "idtp")) or 0 for item in summaries)
        idfp = sum(nested(item, (section, "idfp")) or 0 for item in summaries)
        idfn = sum(nested(item, (section, "idfn")) or 0 for item in summaries)
        switches = sum(nested(item, (section, "id_switches")) or 0 for item in summaries)
        fragments = sum(nested(item, (section, "fragmentations")) or 0 for item in summaries)
        tracking_gt = sum(nested(item, (section, "gt_detections")) or 0 for item in summaries)
        tracking_fp = sum(nested(item, (section, "false_positives")) or 0 for item in summaries)
        tracking_fn = sum(nested(item, (section, "false_negatives")) or 0 for item in summaries)
        output.update({
            f"{prefix}_idf1": safe_ratio(2 * idtp, 2 * idtp + idfp + idfn),
            f"{prefix}_id_precision": safe_ratio(idtp, idtp + idfp),
            f"{prefix}_id_recall": safe_ratio(idtp, idtp + idfn),
            f"{prefix}_id_switches": switches,
            f"{prefix}_id_switches_per_1000_gt": safe_ratio(1000 * switches, tracking_gt),
            f"{prefix}_fragmentations": fragments,
            f"{prefix}_fragmentations_per_1000_gt": safe_ratio(1000 * fragments, tracking_gt),
            f"{prefix}_mota": safe_ratio(
                tracking_gt - tracking_fn - tracking_fp - switches,
                tracking_gt,
            ),
        })
    return output


def distribution_summary(values, bootstrap_samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "n": 0, "mean": None, "median": None, "q25": None, "q75": None,
            "ci95_low": None, "ci95_high": None,
        }
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(max(0, bootstrap_samples)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        bootstrap_means.append(float(np.mean(sample)))
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "ci95_low": float(np.percentile(bootstrap_means, 2.5)) if bootstrap_means else None,
        "ci95_high": float(np.percentile(bootstrap_means, 97.5)) if bootstrap_means else None,
    }


def render_charts(rows, summaries, evaluation_root, chart_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for charts; install project requirements or pass --no-charts"
        ) from error

    chart_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["sequence"] for row in rows]
    plot_series(
        plt, labels, rows,
        [
            ("det_precision_50", "Precision@0.50"),
            ("det_recall_50", "Recall@0.50"),
            ("det_f1_50", "F1@0.50"),
        ],
        "Detection per sequence", "Score", chart_dir / "detection_per_sequence",
        bounded=True,
    )
    plot_series(
        plt, labels, rows,
        [
            ("display_hota", "HOTA"),
            ("display_idf1", "IDF1"),
            ("display_mota", "MOTA"),
        ],
        "FT display-track quality per sequence", "Score",
        chart_dir / "tracking_per_sequence", bounded=False,
    )
    plot_series(
        plt, labels, rows,
        [
            ("display_id_switches_per_1000_gt", "ID switches / 1k GT"),
            ("display_fragmentations_per_1000_gt", "Fragments / 1k GT"),
        ],
        "Tracking errors per sequence", "Errors per 1,000 GT detections",
        chart_dir / "tracking_errors_per_sequence", bounded=False,
    )
    plot_raw_display(plt, labels, rows, chart_dir / "raw_vs_display_idf1")
    plot_size_recall(plt, rows, chart_dir / "recall_by_bbox_size")
    plot_recall_vs_idf1(plt, rows, chart_dir / "detection_recall_vs_idf1")
    plot_heatmap(plt, labels, rows, chart_dir / "metric_heatmap")
    plot_precision_recall(plt, labels, evaluation_root, chart_dir / "precision_recall_50")


def plot_series(plt, labels, rows, series, title, ylabel, target, bounded):
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.32), 5.2))
    x = np.arange(len(labels))
    for key, label in series:
        axis.plot(x, [to_nan(row.get(key)) for row in rows], marker="o", markersize=3, label=label)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=90 if len(labels) > 12 else 45, ha="right", fontsize=7)
    if bounded:
        axis.set_ylim(0, 1.03)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=len(series))
    figure.tight_layout()
    save_figure(figure, target)


def plot_raw_display(plt, labels, rows, target):
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    for row in rows:
        raw, display = row.get("raw_idf1"), row.get("display_idf1")
        if raw is None or display is None:
            continue
        axis.plot([0, 1], [raw, display], color="#6b7280", alpha=0.45)
        axis.scatter([0, 1], [raw, display], color=["#2563eb", "#ea580c"], s=20)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Raw ByteTrack", "FT display linking"])
    axis.set_ylabel("IDF1")
    axis.set_title("Effect of FT linking on identity tracking")
    axis.set_ylim(0, 1.03)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, target)


def plot_size_recall(plt, rows, target):
    values = [
        [row.get(f"recall_{size}") for row in rows if row.get(f"recall_{size}") is not None]
        for size in ("small", "medium", "large")
    ]
    figure, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.boxplot(values, labels=["Small (<0.5%)", "Medium (0.5–2%)", "Large (≥2%)"])
    axis.set_ylabel("Detection recall @ IoU 0.50")
    axis.set_title("Recall sensitivity to athlete bbox size")
    axis.set_ylim(0, 1.03)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, target)


def plot_recall_vs_idf1(plt, rows, target):
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    x = [row.get("det_recall_50") for row in rows]
    y = [row.get("display_idf1") for row in rows]
    axis.scatter(x, y, color="#2563eb", alpha=0.8)
    if len(rows) <= 20:
        for row, x_value, y_value in zip(rows, x, y):
            if x_value is not None and y_value is not None:
                axis.annotate(row["sequence"], (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Detection recall @ IoU 0.50")
    axis.set_ylabel("Display-track IDF1")
    axis.set_title("Does missed detection explain tracking quality?")
    axis.set_xlim(0, 1.03)
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    save_figure(figure, target)


def plot_heatmap(plt, labels, rows, target):
    keys = ["det_precision_50", "det_recall_50", "det_f1_50", "display_hota", "display_idf1", "display_mota"]
    names = ["Precision", "Recall", "F1", "HOTA", "IDF1", "MOTA"]
    matrix = np.asarray([[to_nan(row.get(key)) for key in keys] for row in rows])
    figure, axis = plt.subplots(figsize=(8.5, max(4.5, len(labels) * 0.25)))
    minimum = min(0.0, float(np.nanmin(matrix))) if np.any(np.isfinite(matrix)) else 0.0
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=minimum, vmax=1)
    axis.set_xticks(np.arange(len(keys)))
    axis.set_xticklabels(names)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_title("Sequence-level metric map")
    figure.colorbar(image, ax=axis, label="Score")
    figure.tight_layout()
    save_figure(figure, target)


def plot_precision_recall(plt, labels, evaluation_root, target):
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    plotted = 0
    for sequence in labels:
        path = evaluation_root / sequence / "precision_recall.csv"
        if not path.is_file() or not path.stat().st_size:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            points = [
                row for row in csv.DictReader(handle)
                if abs(float(row["iou_threshold"]) - 0.50) < 1e-9
            ]
        if not points:
            continue
        axis.plot(
            [float(row["recall"]) for row in points],
            [float(row["precision"]) for row in points],
            alpha=0.4,
            linewidth=1,
            label=sequence if len(labels) <= 12 else None,
        )
        plotted += 1
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Detector precision–recall curves @ IoU 0.50")
    axis.set_xlim(0, 1.03)
    axis.set_ylim(0, 1.03)
    axis.grid(alpha=0.25)
    if plotted and len(labels) <= 12:
        axis.legend(frameon=False, fontsize=7, ncol=2)
    if not plotted:
        axis.text(0.5, 0.5, "Detector confidence unavailable in these artifacts", ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    save_figure(figure, target)


def save_figure(figure, target):
    figure.savefig(target.with_suffix(".png"), dpi=160)
    figure.savefig(target.with_suffix(".svg"))
    import matplotlib.pyplot as plt
    plt.close(figure)


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["sequence"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_table(path, macro):
    rows = [{"metric": name, **values} for name, values in macro.items()]
    write_csv(path, rows)


def nested(value, path):
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def to_nan(value):
    return np.nan if value is None else float(value)


if __name__ == "__main__":
    main()
