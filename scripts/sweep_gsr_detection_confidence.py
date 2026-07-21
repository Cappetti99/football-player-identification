#!/usr/bin/env python3
"""Pool existing GSR precision/recall artifacts over confidence thresholds.

This is an offline analysis: it does not run FT, YOLO, or tracking again.
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.005)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.evaluation_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sequences = []
    for entry in manifest.get("sequences") or []:
        sequence = entry["sequence"]
        summary = json.loads(
            (root / sequence / "summary.json").read_text(encoding="utf-8")
        )
        curve = read_curve(root / sequence / "precision_recall.csv", args.iou)
        sequences.append({
            "sequence": sequence,
            "gt": int(summary["detection"]["average_precision"][key(args.iou)]["gt"]),
            "curve": curve,
        })

    thresholds = grid(args.grid_step)
    rows = [evaluate_threshold(sequences, threshold) for threshold in thresholds]
    best_f1 = max(rows, key=lambda row: (row["micro_f1"], row["threshold"]))
    feasible = [row for row in rows if row["micro_recall"] >= args.min_recall]
    best_precision_at_recall = max(
        feasible,
        key=lambda row: (row["micro_precision"], row["micro_f1"], row["threshold"]),
    ) if feasible else None

    write_csv(output / "threshold_sweep.csv", rows)
    report = {
        "source": "existing per-sequence precision_recall.csv artifacts",
        "inference_rerun": False,
        "iou_threshold": args.iou,
        "sequence_count": len(sequences),
        "minimum_recall_constraint": args.min_recall,
        "best_micro_f1": best_f1,
        "best_precision_subject_to_recall": best_precision_at_recall,
    }
    (output / "recommendation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    render(rows, best_f1, best_precision_at_recall, output / "confidence_sweep.png")
    print(json.dumps(report, indent=2))


def read_curve(path, iou):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if abs(float(row["iou_threshold"]) - iou) < 1e-9
        ]
    rows.sort(key=lambda row: (-float(row["confidence"]), int(row["rank"])))
    return rows


def evaluate_threshold(sequences, threshold):
    per_sequence = []
    for item in sequences:
        included = [row for row in item["curve"] if float(row["confidence"]) >= threshold]
        if included:
            last = included[-1]
            tp, fp = int(last["tp"]), int(last["fp"])
        else:
            tp, fp = 0, 0
        fn = item["gt"] - tp
        per_sequence.append((tp, fp, fn))
    tp = sum(row[0] for row in per_sequence)
    fp = sum(row[1] for row in per_sequence)
    fn = sum(row[2] for row in per_sequence)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    macro_precision = mean([ratio(a, a + b) for a, b, _ in per_sequence])
    macro_recall = mean([ratio(a, a + c) for a, _, c in per_sequence])
    macro_f1 = mean([ratio(2 * a, 2 * a + b + c) for a, b, c in per_sequence])
    return {
        "threshold": round(threshold, 6),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def grid(step):
    if step <= 0 or step > 1:
        raise ValueError("--grid-step must be in (0, 1]")
    count = int(round(1.0 / step))
    return [min(1.0, index * step) for index in range(count + 1)]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def key(value):
    return f"{value:.2f}"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def render(rows, best_f1, constrained, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render the sweep chart") from error
    x = [row["threshold"] for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(x, [row["micro_precision"] for row in rows], label="Precision")
    axis.plot(x, [row["micro_recall"] for row in rows], label="Recall")
    axis.plot(x, [row["micro_f1"] for row in rows], label="F1")
    axis.axvline(best_f1["threshold"], color="#111827", linestyle="--", alpha=.7,
                 label=f"Best F1: {best_f1['threshold']:.3f}")
    if constrained:
        axis.axvline(constrained["threshold"], color="#dc2626", linestyle=":", alpha=.8,
                     label=f"Best P with recall constraint: {constrained['threshold']:.3f}")
    axis.set(xlabel="YOLO confidence threshold", ylabel="Micro score",
             title="GSR detection confidence sweep (IoU 0.50)", ylim=(0, 1.03))
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
