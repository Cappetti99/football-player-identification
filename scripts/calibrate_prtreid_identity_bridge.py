#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from calibrate_prtreid_linker import evaluate, select_zero_false_positive_threshold


def main():
    parser = argparse.ArgumentParser(description="Calibrate zero-FP PRTReID identity-bridge thresholds.")
    parser.add_argument("labels")
    parser.add_argument("--output-dir", default="evaluation_outputs/prtreid_identity_bridge_calibration")
    parser.add_argument("--base-config", default="configs/prtreid_identity_bridge_audit_int_ata.yaml")
    args = parser.parse_args()
    rows = read_labels(Path(args.labels))
    if any(row["label"] not in {"same", "different"} for row in rows):
        raise SystemExit("Every bridge candidate must be labeled same or different before calibration")
    decided = [row for row in rows if row["label"] in {"same", "different"} and row["mutual_nearest"]]
    if not decided:
        raise SystemExit("No decided mutual-nearest labels found")
    threshold = select_zero_false_positive_threshold(decided)
    metrics = evaluate(decided, threshold)
    promotable = metrics["false_positives"] == 0 and metrics["true_positives"] > 0
    report = {"threshold": threshold, "metrics": metrics, "promotable": promotable}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "prtreid_identity_bridge_calibrated.yaml").write_text(
        f"base_config: {Path(args.base_config).resolve()}\n\n"
        "prtreid_identity_bridge:\n"
        "  enabled: true\n"
        f"  apply: {'true' if promotable else 'false'}\n"
        f"  min_similarity: {threshold['min_similarity']:.17g}\n"
        f"  min_margin: {threshold['min_margin']:.17g}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def read_labels(path):
    output = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.append({
                "video_id": str(row.get("video_id") or "unknown"),
                "link_type": "cross_scene",
                "similarity": float(row.get("visual_similarity") or 0.0),
                "margin": float(row.get("similarity_margin") or 0.0),
                "mutual_nearest": str(row.get("mutual_nearest")).lower() in {"1", "true", "yes"},
                "label": str(row.get("label") or "uncertain").lower(),
            })
    return output


if __name__ == "__main__":
    main()
