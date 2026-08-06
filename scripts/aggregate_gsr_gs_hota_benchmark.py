#!/usr/bin/env python3
"""Aggregate GS-HOTA across sequences for both jersey decision-policy arms.

Reads the per-sequence summary.json files produced by two runs of
scripts/run_gsr_gs_hota_benchmark.py (one per arm, same manifest, same
sequences) and reports:

- macro (unweighted mean/median/bootstrap CI across sequences) for each arm;
- a paired per-sequence delta (arm B - arm A), bootstrapped over sequences,
  same style as the jersey held-out policy experiment's paired comparison.

    python3 scripts/aggregate_gsr_gs_hota_benchmark.py \
        --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \
        --arm-a-root evaluation_outputs/gs_hota_benchmark/arm_a \
        --arm-b-root evaluation_outputs/gs_hota_benchmark/arm_b \
        --output-dir evaluation_outputs/gs_hota_benchmark/aggregate
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np


METRICS = ("gs_hota", "gs_deta", "gs_assa", "gs_loca")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm-a-root", required=True)
    parser.add_argument("--arm-b-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    arm_a_root = Path(args.arm_a_root)
    arm_b_root = Path(args.arm_b_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in manifest.get("sequences") or []:
        sequence = entry["sequence"]
        a = load_summary(arm_a_root / sequence / "summary.json")
        b = load_summary(arm_b_root / sequence / "summary.json")
        row = {"sequence": sequence}
        for metric in METRICS:
            row[f"a_{metric}"] = a["tracking"].get(metric)
            row[f"b_{metric}"] = b["tracking"].get(metric)
            row[f"delta_{metric}"] = safe_delta(b["tracking"].get(metric), a["tracking"].get(metric))
        row["a_jersey_coverage_tracklet"] = a["jersey"]["tracklet_level"]["coverage_visible"]
        row["b_jersey_coverage_tracklet"] = b["jersey"]["tracklet_level"]["coverage_visible"]
        row["a_jersey_accuracy_tracklet"] = a["jersey"]["tracklet_level"]["accuracy_all_visible"]
        row["b_jersey_accuracy_tracklet"] = b["jersey"]["tracklet_level"]["accuracy_all_visible"]
        row["a_pitch_mean_error"] = a["pitch"]["mean_error"]
        row["b_pitch_mean_error"] = b["pitch"]["mean_error"]
        rows.append(row)

    write_csv(output / "per_sequence.csv", rows)

    macro = {}
    for metric in METRICS:
        macro[f"a_{metric}"] = distribution_summary(
            [row[f"a_{metric}"] for row in rows if row[f"a_{metric}"] is not None],
            args.bootstrap_samples, args.seed,
        )
        macro[f"b_{metric}"] = distribution_summary(
            [row[f"b_{metric}"] for row in rows if row[f"b_{metric}"] is not None],
            args.bootstrap_samples, args.seed + 1,
        )
        macro[f"delta_{metric}"] = distribution_summary(
            [row[f"delta_{metric}"] for row in rows if row[f"delta_{metric}"] is not None],
            args.bootstrap_samples, args.seed + 2,
        )

    aggregate = {
        "sequence_count": len(rows),
        "manifest": str(Path(args.manifest).resolve()),
        "macro": macro,
        "metric_definitions": {
            "macro": "unweighted distribution across sequences",
            "delta": "paired per-sequence (arm B - arm A), bootstrapped over sequences",
            "bootstrap_ci": "percentile CI from sequence-level bootstrap resampling",
            "gs_hota": "Somers et al. CVPR24 GS-HOTA, LocSim(tau=5m) x IdSim(team+role+jersey)",
        },
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


def load_summary(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing sequence evaluation: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_delta(b, a):
    if b is None or a is None:
        return None
    return float(b) - float(a)


def distribution_summary(values, bootstrap_samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    bootstrap_means = []
    for _ in range(max(0, bootstrap_samples)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        bootstrap_means.append(float(np.mean(sample)))
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(bootstrap_means, 2.5)) if bootstrap_means else None,
        "ci95_high": float(np.percentile(bootstrap_means, 97.5)) if bootstrap_means else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    import csv
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
