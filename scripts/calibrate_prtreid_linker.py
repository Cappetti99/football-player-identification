#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


VALID_LABELS = {"same", "different"}


def main():
    parser = argparse.ArgumentParser(description="Calibrate conservative PRTReID linker thresholds from pair audits.")
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--output-dir", default="evaluation_outputs/prtreid_linker_calibration")
    parser.add_argument("--min-recall", type=float, default=0.20)
    parser.add_argument("--min-correct-links", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for path in args.labels:
        rows.extend(read_labels(Path(path)))
    rows = [row for row in rows if row["label"] in VALID_LABELS and row["mutual_nearest"]]
    if not rows:
        raise SystemExit("No decided mutual-nearest labels found")

    videos = sorted({row["video_id"] for row in rows})
    report = {"videos": videos, "policies": {}, "folds": []}
    final = {}
    for link_type in ("same_scene", "cross_scene"):
        policy_rows = [row for row in rows if row["link_type"] == link_type]
        if not policy_rows:
            report["policies"][link_type] = {"status": "no_labels"}
            continue
        fold_thresholds = []
        for held_out in videos:
            train = [row for row in policy_rows if row["video_id"] != held_out]
            valid = [row for row in policy_rows if row["video_id"] == held_out]
            if not train or not valid:
                continue
            threshold = select_zero_false_positive_threshold(train)
            metrics = evaluate(valid, threshold)
            fold = {"link_type": link_type, "held_out_video": held_out, "threshold": threshold, "metrics": metrics}
            report["folds"].append(fold)
            fold_thresholds.append(threshold)
        if fold_thresholds:
            threshold = {
                "min_similarity": max(row["min_similarity"] for row in fold_thresholds),
                "min_margin": max(row["min_margin"] for row in fold_thresholds),
            }
        else:
            threshold = select_zero_false_positive_threshold(policy_rows)
        metrics = evaluate(policy_rows, threshold)
        enabled = bool(metrics["false_positives"] == 0 and metrics["true_positives"] >= 1)
        final[link_type] = {**threshold, "enabled": enabled}
        report["policies"][link_type] = {
            "status": "calibrated",
            "threshold": threshold,
            "metrics": metrics,
            "enabled": enabled,
        }

    overall = aggregate_metrics(report["policies"])
    report["overall"] = overall
    report["pretrained_gate_pass"] = bool(
        overall["false_positives"] == 0
        and overall["recall"] >= float(args.min_recall)
        and overall["true_positives"] >= int(args.min_correct_links)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_overlay(final, output_dir / "prtreid_linking_calibrated.yaml")
    print(json.dumps(report, indent=2))


def read_labels(path):
    output = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label") or "uncertain").strip().lower()
            output.append(
                {
                    "video_id": str(row.get("video_id") or "unknown"),
                    "link_type": str(row.get("link_type") or "same_scene"),
                    "similarity": float(row.get("visual_similarity") or 0.0),
                    "margin": float(row.get("similarity_margin") or 0.0),
                    "mutual_nearest": parse_bool(row.get("mutual_nearest")),
                    "label": label,
                }
            )
    return output


def select_zero_false_positive_threshold(rows):
    similarities = sorted({float(row["similarity"]) for row in rows})
    margins = sorted({float(row["margin"]) for row in rows})
    best = None
    for similarity in similarities:
        for margin in margins:
            threshold = {"min_similarity": similarity, "min_margin": margin}
            metrics = evaluate(rows, threshold)
            if metrics["false_positives"] != 0:
                continue
            score = (metrics["true_positives"], metrics["recall"], similarity, margin)
            if best is None or score > best[0]:
                best = (score, threshold)
    if best is None:
        return {"min_similarity": 1.0, "min_margin": 1.0}
    return best[1]


def evaluate(rows, threshold):
    positives = sum(row["label"] == "same" for row in rows)
    negatives = sum(row["label"] == "different" for row in rows)
    accepted = [
        row
        for row in rows
        if row["similarity"] >= threshold["min_similarity"] and row["margin"] >= threshold["min_margin"]
    ]
    tp = sum(row["label"] == "same" for row in accepted)
    fp = sum(row["label"] == "different" for row in accepted)
    return {
        "labels": len(rows),
        "positives": positives,
        "negatives": negatives,
        "accepted": len(accepted),
        "true_positives": tp,
        "false_positives": fp,
        "precision": tp / len(accepted) if accepted else 1.0,
        "recall": tp / positives if positives else 0.0,
    }


def aggregate_metrics(policies):
    totals = defaultdict(int)
    for policy in policies.values():
        metrics = policy.get("metrics") or {}
        for key in ("labels", "positives", "negatives", "accepted", "true_positives", "false_positives"):
            totals[key] += int(metrics.get(key) or 0)
    totals["precision"] = totals["true_positives"] / totals["accepted"] if totals["accepted"] else 1.0
    totals["recall"] = totals["true_positives"] / totals["positives"] if totals["positives"] else 0.0
    return dict(totals)


def write_overlay(policies, path):
    lines = ["prtreid_linking:"]
    for link_type in ("same_scene", "cross_scene"):
        threshold = policies.get(link_type)
        if not threshold:
            continue
        lines.extend(
            [
                f"  {link_type}:",
                f"    enabled: {'true' if threshold['enabled'] else 'false'}",
                f"    min_similarity: {threshold['min_similarity']:.17g}",
                f"    min_margin: {threshold['min_margin']:.17g}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
