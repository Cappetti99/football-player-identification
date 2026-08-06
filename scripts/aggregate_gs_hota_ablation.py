#!/usr/bin/env python3
"""Aggregate the GS-HOTA oracle ablation across sequences and policy arms.

Runs scripts/ablate_gs_hota_oracles.py's ablation on every sequence of a
manifest, for each jersey decision-policy arm, and reports sequence-level
distributions with bootstrap confidence intervals. A single sequence is not
enough to attribute error to a module: GS-HOTA varies by orders of magnitude
between sequences (Somers et al., Fig. 5), so per-module costs must be read
across a block, never off one video.

Pure re-analysis of existing tracklet artifacts -- no inference, no GPU.

Emits, in --output-dir:
  per_sequence.csv      one row per (arm, sequence, configuration)
  attribution.csv       one row per (arm, module) with bootstrap CIs
  aggregate.json        everything, machine-readable
  tables.tex            booktabs tables ready to \\input into the thesis

    python3 scripts/aggregate_gs_hota_ablation.py \\
        --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \\
        --artifacts-root artifacts/gs_hota_benchmark \\
        --arms a b \\
        --output-dir evaluation_outputs/gs_hota_ablation/pilot12
"""

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


ARM_LABELS = {"a": "SAR primary", "b": "region CTC primary"}
CONFIGURATION_LABELS = {
    "baseline": "baseline (no oracle)",
    "oracle_calibration": "+ oracle calibration",
    "oracle_team": "+ oracle team",
    "oracle_role": "+ oracle role",
    "oracle_jersey": "+ oracle jersey",
    "oracle_all_but_calibration": "all attributes but calibration",
    "oracle_all_but_team": "all attributes but team",
    "oracle_all_but_role": "all attributes but role",
    "oracle_all_but_jersey": "all attributes but jersey",
    "oracle_all_attributes": "all attributes oracle",
    "full_oracle": "full oracle (sanity)",
}
# oracle_detection_tracking is deliberately excluded from the reported tables:
# it is not a clean counterfactual (see ablate_gs_hota_oracles.py).
EXCLUDED_FROM_TABLES = {"oracle_detection_tracking"}


def load_ablation():
    path = REPO_ROOT / "scripts" / "ablate_gs_hota_oracles.py"
    spec = importlib.util.spec_from_file_location("ablate_gs_hota_oracles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--arms", nargs="+", default=["a", "b"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tau-metres", type=float, default=5.0)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="required to run on the frozen GSR test split",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("split") == "test" and not args.allow_test:
        raise SystemExit(
            "The GSR test split is frozen. This ablation is a diagnostic, not a "
            "final evaluation: run it on valid. Pass --allow-test only if you "
            "have decided to spend the test split on it."
        )

    ablation = load_ablation()
    artifacts_root = Path(args.artifacts_root)
    entries = manifest.get("sequences") or []

    rows = []
    failures = []
    for arm in args.arms:
        for entry in entries:
            sequence = entry["sequence"]
            tracklets = (
                artifacts_root / f"arm_{arm}" / sequence / "metadata"
                / f"{sequence}_tracklets.csv"
            )
            if not tracklets.is_file():
                raise SystemExit(f"missing tracklets for arm {arm} / {sequence}: {tracklets}")
            print(f"ablating arm {arm.upper()} / {sequence}", flush=True)
            payload = ablation.run_ablation(
                labels=entry["labels"],
                tracklets=str(tracklets),
                tau_metres=args.tau_metres,
                max_frames=args.max_frames,
            )
            if not payload["sanity_check"]["passed"]:
                failures.append(f"arm {arm} / {sequence}: {payload['sanity_check']['value']}")
            dropped = payload["unscoreable_subjects"]["ground_truth_boxes"]
            if dropped:
                print(
                    f"  dropped {dropped} unscoreable ground-truth boxes "
                    f"(no bbox_pitch) and {payload['unscoreable_subjects']['matched_predictions']} "
                    "matched detections",
                    flush=True,
                )
            rows.append({"arm": arm, "sequence": sequence, "payload": payload})

    if failures:
        raise SystemExit(
            "full_oracle != 1.0 on:\n  " + "\n  ".join(failures)
            + "\nThe harness does not reproduce ground truth on these sequences; "
            "no aggregate would be trustworthy."
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    per_sequence = flatten_per_sequence(rows)
    write_csv(output / "per_sequence.csv", per_sequence)

    aggregate = {
        "manifest": str(Path(args.manifest).resolve()),
        "split": manifest.get("split"),
        "sequence_count": len(entries),
        "arms": {arm: ARM_LABELS.get(arm, arm) for arm in args.arms},
        "tau_metres": args.tau_metres,
        "unscoreable_subjects": {
            row["sequence"]: row["payload"]["unscoreable_subjects"]
            for row in rows
            if row["arm"] == args.arms[0]
            and row["payload"]["unscoreable_subjects"]["ground_truth_boxes"]
        },
        "configurations": aggregate_configurations(rows, args),
        "attribution": aggregate_attribution(rows, args),
        "method": (
            "evaluation-surface oracle substitution, aggregated over sequences; "
            "macro = unweighted across sequences, CI = sequence-level bootstrap"
        ),
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    write_attribution_csv(output / "attribution.csv", aggregate["attribution"])
    (output / "tables.tex").write_text(latex_tables(aggregate), encoding="utf-8")

    print_report(aggregate)
    print(f"\nwritten: {output / 'aggregate.json'}")
    print(f"written: {output / 'tables.tex'}")


def flatten_per_sequence(rows):
    output = []
    for row in rows:
        for name, configuration in row["payload"]["configurations"].items():
            output.append({
                "arm": row["arm"],
                "sequence": row["sequence"],
                "configuration": name,
                "gs_hota": configuration["gs_hota"],
                "gs_deta": configuration["gs_deta"],
                "gs_assa": configuration["gs_assa"],
                "gs_loca": configuration["gs_loca"],
                "delta_vs_baseline": configuration["delta_vs_baseline"],
            })
    return output


def aggregate_configurations(rows, args):
    output = {}
    for index, name in enumerate(CONFIGURATION_LABELS):
        output[name] = {}
        for arm in args.arms:
            values = [
                row["payload"]["configurations"][name]["gs_hota"]
                for row in rows
                if row["arm"] == arm
            ]
            output[name][arm] = distribution(values, args.bootstrap_samples, args.seed + index)
    return output


def aggregate_attribution(rows, args):
    ablation = load_ablation()
    modules = list(ablation.ATTRIBUTE_MODULES) + ["detection_tracking"]
    output = {}
    for index, module in enumerate(modules):
        output[module] = {}
        for arm in args.arms:
            arm_rows = [row for row in rows if row["arm"] == arm]
            costs = [row["payload"]["module_attribution"][module]["isolated_cost"] for row in arm_rows]
            gains = [
                row["payload"]["module_attribution"][module]["marginal_gain"]
                for row in arm_rows
                if row["payload"]["module_attribution"][module]["marginal_gain"] is not None
            ]
            output[module][arm] = {
                "isolated_cost": distribution(costs, args.bootstrap_samples, args.seed + 100 + index),
                "marginal_gain": (
                    distribution(gains, args.bootstrap_samples, args.seed + 200 + index)
                    if gains else None
                ),
            }
    return output


def distribution(values, bootstrap_samples, seed):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    means = []
    for _ in range(max(0, bootstrap_samples)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(float(np.mean(sample)))
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(means, 2.5)) if means else None,
        "ci95_high": float(np.percentile(means, 97.5)) if means else None,
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_attribution_csv(path, attribution):
    rows = []
    for module, arms in attribution.items():
        for arm, entry in arms.items():
            cost = entry["isolated_cost"]
            gain = entry["marginal_gain"]
            rows.append({
                "module": module,
                "arm": arm,
                "isolated_cost_mean": cost["mean"],
                "isolated_cost_ci95_low": cost["ci95_low"],
                "isolated_cost_ci95_high": cost["ci95_high"],
                "marginal_gain_mean": gain["mean"] if gain else None,
                "marginal_gain_ci95_low": gain["ci95_low"] if gain else None,
                "marginal_gain_ci95_high": gain["ci95_high"] if gain else None,
            })
    write_csv(path, rows)


def latex_tables(aggregate):
    arms = list(aggregate["arms"])
    lines = [
        "% Generated by scripts/aggregate_gs_hota_ablation.py -- do not edit by hand.",
        "% Requires \\usepackage{booktabs}.",
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{GS-HOTA under oracle substitution, "
        f"macro mean over {aggregate['sequence_count']} sequences "
        "(95\\% sequence-level bootstrap CI in brackets).}",
        "  \\label{tab:gs-hota-ablation}",
        "  \\begin{tabular}{l" + "r" * len(arms) + "}",
        "    \\toprule",
        "    Configuration & " + " & ".join(ARM_LABELS.get(a, a) for a in arms) + " \\\\",
        "    \\midrule",
    ]
    for name, label in CONFIGURATION_LABELS.items():
        if name in EXCLUDED_FROM_TABLES:
            continue
        cells = []
        for arm in arms:
            entry = aggregate["configurations"][name][arm]
            cells.append(f"{entry['mean']:.3f} [{entry['ci95_low']:.3f}, {entry['ci95_high']:.3f}]")
        if name == "oracle_all_attributes":
            lines.append("    \\midrule")
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Isolated cost of each module in GS-HOTA points: how much "
        "the module still costs once every other one is perfect. Costs are not "
        "additive, because GS-HOTA multiplies localisation by identity "
        "similarity and failures on the same subject overlap.}",
        "  \\label{tab:gs-hota-attribution}",
        "  \\begin{tabular}{l" + "r" * len(arms) + "}",
        "    \\toprule",
        "    Module & " + " & ".join(ARM_LABELS.get(a, a) for a in arms) + " \\\\",
        "    \\midrule",
    ]
    ordering = sorted(
        aggregate["attribution"],
        key=lambda m: -aggregate["attribution"][m][arms[0]]["isolated_cost"]["mean"],
    )
    for module in ordering:
        cells = []
        for arm in arms:
            entry = aggregate["attribution"][module][arm]["isolated_cost"]
            cells.append(f"{entry['mean']:.3f} [{entry['ci95_low']:.3f}, {entry['ci95_high']:.3f}]")
        lines.append(f"    {module.replace('_', '/')} & " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def print_report(aggregate):
    arms = list(aggregate["arms"])
    print()
    print(f"GS-HOTA over {aggregate['sequence_count']} sequences (split: {aggregate['split']})")
    header = f"{'configuration':<34}" + "".join(f"{ARM_LABELS.get(a, a):>26}" for a in arms)
    print(header)
    print("-" * len(header))
    for name in CONFIGURATION_LABELS:
        if name in EXCLUDED_FROM_TABLES:
            continue
        cells = ""
        for arm in arms:
            entry = aggregate["configurations"][name][arm]
            cells += f"{entry['mean']:>10.4f} [{entry['ci95_low']:.3f},{entry['ci95_high']:.3f}]"
        print(f"{name:<34}{cells}")
    print("-" * len(header))
    print()
    print("isolated cost per module (GS-HOTA points)")
    print(f"{'module':<24}" + "".join(f"{ARM_LABELS.get(a, a):>26}" for a in arms))
    print("-" * len(header))
    ordering = sorted(
        aggregate["attribution"],
        key=lambda m: -aggregate["attribution"][m][arms[0]]["isolated_cost"]["mean"],
    )
    for module in ordering:
        cells = ""
        for arm in arms:
            entry = aggregate["attribution"][module][arm]["isolated_cost"]
            cells += f"{entry['mean']:>10.4f} [{entry['ci95_low']:.3f},{entry['ci95_high']:.3f}]"
        print(f"{module:<24}{cells}")
    print("-" * len(header))


if __name__ == "__main__":
    main()
