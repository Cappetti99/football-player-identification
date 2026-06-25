#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Select a zero-FP reliable-jersey candidate threshold.")
    parser.add_argument("labels")
    parser.add_argument("--output-dir", default="evaluation_outputs/identity_nearmiss_calibration")
    parser.add_argument("--base-config", default="configs/prtreid_linking_additive_control_disabled.yaml")
    args = parser.parse_args()
    rows = read_labels(Path(args.labels))
    if any(row["label"] not in {"same", "different"} for row in rows):
        raise SystemExit("Every near-miss must be labeled same or different before calibration")
    decided = [row for row in rows if row["label"] in {"same", "different"}]
    if not decided:
        raise SystemExit("No decided labels found")
    threshold, metrics = select_threshold(decided)
    promotable = metrics["false_positives"] == 0 and metrics["true_positives"] > 0
    report = {"threshold": threshold, "metrics": metrics, "promotable": promotable}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    yaml = (
        f"base_config: {Path(args.base_config).resolve()}\n\n"
        "identity_propagation:\n  enabled: false\n"
        "jersey_identity_linking:\n  enabled: false\n"
        "prtreid_linking:\n  enabled: false\n"
        "prtreid_identity_bridge:\n  enabled: false\n"
        "identity:\n  reliable_jersey_min_candidate_score: {:.17g}\n"
    ).format(threshold if promotable else 0.20)
    (output / "identity_nearmiss_calibrated.yaml").write_text(yaml, encoding="utf-8")
    print(json.dumps(report, indent=2))


def read_labels(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {"score": float(row.get("jersey_candidate_score") or 0.0), "label": str(row.get("label") or "uncertain").lower()}
            for row in csv.DictReader(handle)
        ]


def select_threshold(rows):
    best = None
    for threshold in sorted({row["score"] for row in rows}):
        accepted = [row for row in rows if row["score"] >= threshold]
        tp = sum(row["label"] == "same" for row in accepted)
        fp = sum(row["label"] == "different" for row in accepted)
        if fp:
            continue
        score = (tp, -threshold)
        if best is None or score > best[0]:
            best = (score, threshold, accepted)
    if best is None:
        return 0.20, metrics(rows, 0.20)
    return best[1], metrics(rows, best[1])


def metrics(rows, threshold):
    accepted = [row for row in rows if row["score"] >= threshold]
    positives = sum(row["label"] == "same" for row in rows)
    tp = sum(row["label"] == "same" for row in accepted)
    fp = sum(row["label"] == "different" for row in accepted)
    return {
        "labels": len(rows), "accepted": len(accepted), "true_positives": tp, "false_positives": fp,
        "precision": tp / len(accepted) if accepted else 1.0,
        "recall": tp / positives if positives else 0.0,
    }


if __name__ == "__main__":
    main()
