#!/usr/bin/env python3
"""Generate thesis figures from aggregate jersey benchmark CSV files."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    root = Path(args.benchmark_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "figures"
    output.mkdir(parents=True, exist_ok=True)

    aggregate = read_csv(root / "aggregate.csv")
    per_sequence = read_csv(root / "per_sequence.csv")
    per_track = read_csv(root / "per_track.csv")
    transitions = read_csv(root / "transitions.csv")

    plt = configure_matplotlib()
    accuracy_coverage(plt, aggregate, output, args.dpi)
    coverage_scatter(plt, aggregate, output, args.dpi)
    sequence_boxplot(plt, per_sequence, output, args.dpi)
    transition_matrix(plt, transitions, output, args.dpi)
    prediction_bias(plt, per_track, output, args.dpi)
    jersey_confusion(plt, per_track, output, args.dpi)

    files = sorted(path.name for path in output.glob("*.png"))
    print({"output_dir": str(output), "figures": files})


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (8.0, 4.8),
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "savefig.bbox": "tight",
    })
    return plt


def accuracy_coverage(plt, rows, output, dpi):
    methods = [row["method"] for row in rows]
    accuracy = [float(row["accuracy_all"]) for row in rows]
    coverage = [float(row["coverage"]) for row in rows]
    x = list(range(len(methods)))
    width = 0.38
    fig, ax = plt.subplots()
    ax.bar([value - width / 2 for value in x], accuracy, width, label="Accuracy all")
    ax.bar([value + width / 2 for value in x], coverage, width, label="Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.legend(frameon=False)
    save(plt, fig, output / "ocr_accuracy_coverage.png", dpi)


def coverage_scatter(plt, rows, output, dpi):
    fig, ax = plt.subplots()
    for row in rows:
        x = float(row["coverage"])
        y = float(row["accuracy_all"])
        ax.scatter(x, y, s=70)
        ax.annotate(row["method"], (x, y), xytext=(5, 4), textcoords="offset points")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy all-track")
    save(plt, fig, output / "ocr_coverage_accuracy_scatter.png", dpi)


def sequence_boxplot(plt, rows, output, dpi):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(float(row["accuracy_all"]))
    methods = list(groups)
    fig, ax = plt.subplots()
    ax.boxplot(
        [groups[method] for method in methods],
        tick_labels=methods,
        showmeans=True,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy all-track per sequence")
    ax.tick_params(axis="x", rotation=25)
    save(plt, fig, output / "ocr_per_sequence_boxplot.png", dpi)


def transition_matrix(plt, rows, output, dpi):
    candidates = []
    for row in rows:
        if row["candidate"] not in candidates:
            candidates.append(row["candidate"])
    categories = [
        "unchanged_correct",
        "recovered_correct",
        "correct_to_wrong",
        "new_wrong_emission",
        "wrong_to_abstention",
        "wrong_to_wrong",
        "unchanged_wrong",
        "both_abstain",
    ]
    matrix = []
    for candidate in candidates:
        counts = Counter(
            row["transition"] for row in rows if row["candidate"] == candidate
        )
        matrix.append([counts[category] for category in categories])
    fig_width = max(8.0, len(categories) * 1.25)
    fig, ax = plt.subplots(figsize=(fig_width, max(3.5, len(candidates) * 0.65)))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(categories)), categories, rotation=35, ha="right")
    ax.set_yticks(range(len(candidates)), candidates)
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            ax.text(column_index, row_index, value, ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Tracklets")
    save(plt, fig, output / "ocr_transition_matrix.png", dpi)


def prediction_bias(plt, rows, output, dpi):
    groups = defaultdict(Counter)
    for row in rows:
        value = row["pred_jersey"]
        if value != "":
            groups[row["method"]][int(float(value))] += 1
    methods = list(groups)
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(9, max(3.2, 2.4 * len(methods))), squeeze=False
    )
    for ax, method in zip(axes[:, 0], methods):
        values = groups[method].most_common(15)
        ax.bar([str(number) for number, _ in values], [count for _, count in values])
        ax.set_title(method, loc="left", fontsize=10)
        ax.set_ylabel("Predictions")
    axes[-1, 0].set_xlabel("Jersey number (15 most frequent)")
    save(plt, fig, output / "ocr_prediction_bias.png", dpi)


def jersey_confusion(plt, rows, output, dpi):
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    best_method = max(
        methods,
        key=lambda name: sum(
            int(row["correct"]) for row in rows if row["method"] == name
        ),
    )
    for method in methods:
        assigned = [row for row in rows if row["method"] == method and row["pred_jersey"]]
        labels = sorted({int(float(row["gt_jersey"])) for row in assigned} | {
            int(float(row["pred_jersey"])) for row in assigned
        })
        if not labels:
            continue
        index = {label: position for position, label in enumerate(labels)}
        matrix = [[0 for _ in labels] for _ in labels]
        for row in assigned:
            matrix[index[int(float(row["gt_jersey"]))]][
                index[int(float(row["pred_jersey"]))]
            ] += 1
        size = max(6.5, min(14.0, len(labels) * 0.32))
        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(matrix, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Predicted jersey")
        ax.set_ylabel("Ground-truth jersey")
        ax.set_title(method)
        fig.colorbar(image, ax=ax, label="Tracklets")
        safe_method = "".join(character if character.isalnum() else "_" for character in method)
        save(plt, fig, output / f"ocr_confusion_{safe_method}.png", dpi)
        if method == best_method:
            fig, ax = plt.subplots(figsize=(size, size))
            image = ax.imshow(matrix, cmap="magma", aspect="auto")
            ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
            ax.set_yticks(range(len(labels)), labels, fontsize=7)
            ax.set_xlabel("Predicted jersey")
            ax.set_ylabel("Ground-truth jersey")
            ax.set_title(f"Best method: {method}")
            fig.colorbar(image, ax=ax, label="Tracklets")
            save(plt, fig, output / "jersey_confusion_matrix.png", dpi)


def save(plt, figure, path, dpi):
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
