#!/usr/bin/env python3
"""Aggregate GSR calibration quality and pitch-link counterfactuals."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.audit_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        evaluation_path = sequence_dir / "evaluation" / "summary.json"
        audit_path = sequence_dir / "link_audit" / "pitch_link_audit.json"
        if not evaluation_path.is_file() or not audit_path.is_file():
            continue
        evaluation = json.loads(evaluation_path.read_text())
        audit = json.loads(audit_path.read_text())["summary"]
        pitch = evaluation["pitch"]
        rows.append({
            "sequence": sequence_dir.name,
            "pitch_coverage": pitch.get("coverage"),
            "pitch_mean_error_m": pitch.get("mean_error"),
            "pitch_median_error_m": pitch.get("median_error"),
            "pitch_p90_error_m": pitch.get("p90_error"),
            "candidate_pairs": audit.get("candidate_pairs"),
            "pitch_usable_pairs": audit.get("pitch_usable_pairs"),
            "currently_linked_pairs": audit.get("currently_linked_pairs"),
            "would_block_current_links": audit.get("would_block_current_links"),
            "blocked_wrong_links_offline": audit.get("blocked_wrong_links_offline"),
            "blocked_correct_links_offline": audit.get("blocked_correct_links_offline"),
        })
    write_csv(output / "per_sequence.csv", rows)
    aggregate = aggregate_rows(rows)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    if rows:
        render(rows, output / "charts")
    print(json.dumps(aggregate, indent=2))


def aggregate_rows(rows):
    blocked_wrong = total(rows, "blocked_wrong_links_offline")
    blocked_correct = total(rows, "blocked_correct_links_offline")
    return {
        "mode": "audit_only",
        "sequence_count": len(rows),
        "mean_sequence_pitch_median_error_m": average(rows, "pitch_median_error_m"),
        "mean_sequence_pitch_p90_error_m": average(rows, "pitch_p90_error_m"),
        "candidate_pairs": total(rows, "candidate_pairs"),
        "pitch_usable_pairs": total(rows, "pitch_usable_pairs"),
        "currently_linked_pairs": total(rows, "currently_linked_pairs"),
        "would_block_current_links": total(rows, "would_block_current_links"),
        "blocked_wrong_links_offline": blocked_wrong,
        "blocked_correct_links_offline": blocked_correct,
        "block_precision_offline": ratio(blocked_wrong, blocked_wrong + blocked_correct),
        "promotion_rule": "do not apply unless blocked_correct_links_offline == 0 and calibration error is acceptable",
    }


def render(rows, chart_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chart_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["sequence"] for row in rows]
    figure, axis = plt.subplots(figsize=(max(9, len(rows) * .6), 5))
    axis.bar(labels, [number(row["pitch_median_error_m"]) for row in rows], label="Median")
    axis.plot(labels, [number(row["pitch_p90_error_m"]) for row in rows], color="#dc2626", marker="o", label="P90")
    axis.set(title="Pitch calibration error by sequence", ylabel="Error (metres)")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(chart_dir / "pitch_error.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(9, len(rows) * .6), 5))
    wrong = [number(row["blocked_wrong_links_offline"]) for row in rows]
    correct = [number(row["blocked_correct_links_offline"]) for row in rows]
    axis.bar(labels, wrong, label="Wrong links blocked")
    axis.bar(labels, correct, bottom=wrong, label="Correct links blocked")
    axis.set(title="Counterfactual speed gate", ylabel="Current links blocked")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(chart_dir / "blocked_links.png", dpi=180)
    plt.close(figure)


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["sequence"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def number(value):
    return 0.0 if value is None else float(value)


def total(rows, key):
    return sum(number(row.get(key)) for row in rows)


def average(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def ratio(a, b):
    return a / b if b else None


if __name__ == "__main__":
    main()
