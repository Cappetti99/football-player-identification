#!/usr/bin/env python3
"""Aggregate comparable GSR jersey OCR runs for thesis reporting."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {None, "", "None", "null", "-1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="Comparable OCR run; repeat for every method.",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", default="development")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--table-prefix", default="ocr")
    args = parser.parse_args()

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    run_dirs = parse_runs(args.run)
    if args.baseline not in run_dirs:
        raise ValueError(f"baseline is not one of the named runs: {args.baseline}")

    output = Path(args.output_dir).resolve()
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    predictions = {}
    availability = {}
    input_files = {}
    for method, root in run_dirs.items():
        pred_path = root / "predictions.csv"
        diag_path = root / "ocr_diagnostics.json"
        predictions[method] = load_predictions(pred_path)
        if diag_path.is_file():
            availability[method] = load_candidate_availability(
                diag_path, predictions[method]
            )
        else:
            availability[method] = {
                key: boolean(row.get("gt_in_top5"))
                for key, row in predictions[method].items()
            }
        input_files[method] = {
            "directory": str(root),
            "predictions_csv": file_record(pred_path),
            "ocr_diagnostics_json": (
                file_record(diag_path) if diag_path.is_file() else None
            ),
        }

    validate_surfaces(predictions)
    per_track = build_per_track(predictions, availability, args.split_name)
    per_sequence = build_per_sequence(per_track)
    aggregate = build_aggregate(per_track, per_sequence)
    transitions = build_transitions(predictions, args.baseline, args.split_name)
    transition_summary = build_transition_summary(transitions)
    bootstrap = build_bootstrap(
        per_track,
        baseline=args.baseline,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )

    write_csv(output / "per_track.csv", per_track)
    write_csv(output / "per_sequence.csv", per_sequence)
    write_csv(output / "aggregate.csv", aggregate)
    write_csv(output / "transitions.csv", transitions)
    qualitative = [
        row for row in transitions
        if row["transition"] in {"recovered_correct", "correct_to_wrong"}
    ]
    write_csv(output / "qualitative_cases.csv", qualitative)
    (output / "transition_summary.json").write_text(
        json.dumps(transition_summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8"
    )
    table_files = write_latex_tables(
        tables, aggregate, per_sequence, bootstrap, transition_summary,
        prefix=args.table_prefix,
    )

    manifest = {
        "format_version": 1,
        "split_name": args.split_name,
        "baseline": args.baseline,
        "methods": list(run_dirs),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "sequence",
        "input_files": input_files,
        "surface": {
            "tracklets": len(next(iter(predictions.values()))),
            "sequences": sorted({key[0] for key in next(iter(predictions.values()))}),
        },
        "outputs": [
            "per_track.csv",
            "per_sequence.csv",
            "aggregate.csv",
            "transitions.csv",
            "qualitative_cases.csv",
            "transition_summary.json",
            "bootstrap.json",
            *[f"tables/{name}" for name in table_files],
        ],
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), **manifest["surface"]}, indent=2))


def parse_runs(values):
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid --run value: {value!r}; expected NAME=DIR")
        name, directory = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"empty or duplicate run name: {name!r}")
        output[name] = Path(directory).resolve()
    return output


def load_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["gt_track_id"]))
        if key in output:
            raise ValueError(f"duplicate prediction key in {path}: {key}")
        output[key] = {
            **row,
            "eval_track_id": str(row.get("eval_track_id", "")),
            "gt": integer(row.get("gt_jersey_number")),
            "pred": integer(row.get("pred_jersey_number")),
        }
    if not output:
        raise ValueError(f"empty predictions file: {path}")
    return output


def load_candidate_availability(path, predictions):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_eval = defaultdict(list)
    for key, row in predictions.items():
        by_eval[row["eval_track_id"]].append(key)
    output = {key: False for key in predictions}
    for diagnostic in (payload.get("tracklets") or {}).values():
        eval_id = str(
            diagnostic.get("display_track_id", diagnostic.get("eval_track_id", ""))
        )
        keys = by_eval.get(eval_id, [])
        if len(keys) != 1:
            continue
        numbers = {
            integer(detection.get("number"))
            for detection in diagnostic.get("detections", [])
            if integer(detection.get("number")) is not None
        }
        key = keys[0]
        output[key] = predictions[key]["gt"] in numbers
    return output


def validate_surfaces(predictions):
    reference_name, reference = next(iter(predictions.items()))
    reference_keys = set(reference)
    for name, rows in predictions.items():
        keys = set(rows)
        if keys != reference_keys:
            raise ValueError(
                f"run surfaces differ: {reference_name} vs {name}; "
                f"missing={sorted(reference_keys - keys)[:10]} "
                f"extra={sorted(keys - reference_keys)[:10]}"
            )
        mismatches = [key for key in keys if rows[key]["gt"] != reference[key]["gt"]]
        if mismatches:
            raise ValueError(f"ground-truth mismatch in {name}: {mismatches[:10]}")


def build_per_track(predictions, availability, split_name):
    rows = []
    for method, values in predictions.items():
        for key in sorted(values):
            row = values[key]
            pred = row["pred"]
            truth = row["gt"]
            rows.append({
                "split": split_name,
                "method": method,
                "sequence": key[0],
                "gt_track_id": key[1],
                "eval_track_id": row["eval_track_id"],
                "gt_jersey": truth,
                "pred_jersey": "" if pred is None else pred,
                "assigned": int(pred is not None),
                "correct": int(pred is not None and pred == truth),
                "wrong": int(pred is not None and pred != truth),
                "abstained": int(pred is None),
                "gt_in_selected_candidates": int(bool(availability[method].get(key))),
                "votes": numeric(row.get("votes")),
                "winner_margin": numeric(row.get("winner_margin")),
                "confidence": numeric(
                    row.get("confidence", row.get("pred_confidence"))
                ),
            })
    return rows


def build_per_sequence(per_track):
    groups = defaultdict(list)
    for row in per_track:
        groups[(row["split"], row["method"], row["sequence"])].append(row)
    output = []
    for (split_name, method, sequence), rows in sorted(groups.items()):
        tracklets = len(rows)
        assigned = sum(row["assigned"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        wrong = sum(row["wrong"] for row in rows)
        output.append({
            "split": split_name,
            "method": method,
            "sequence": sequence,
            "tracklets": tracklets,
            "assigned": assigned,
            "correct": correct,
            "wrong": wrong,
            "abstained": tracklets - assigned,
            "coverage": ratio(assigned, tracklets),
            "accuracy_assigned": ratio(correct, assigned),
            "accuracy_all": ratio(correct, tracklets),
            "gt_in_selected_candidates": sum(
                row["gt_in_selected_candidates"] for row in rows
            ),
            "gt_in_selected_candidates_rate": ratio(
                sum(row["gt_in_selected_candidates"] for row in rows), tracklets
            ),
        })
    return output


def build_aggregate(per_track, per_sequence):
    track_groups = defaultdict(list)
    sequence_groups = defaultdict(list)
    for row in per_track:
        track_groups[(row["split"], row["method"])].append(row)
    for row in per_sequence:
        sequence_groups[(row["split"], row["method"])].append(row)
    output = []
    for key, rows in sorted(track_groups.items()):
        seq_rows = sequence_groups[key]
        tracklets = len(rows)
        assigned = sum(row["assigned"] for row in rows)
        correct = sum(row["correct"] for row in rows)
        wrong = sum(row["wrong"] for row in rows)
        sequence_accuracy = [row["accuracy_all"] for row in seq_rows]
        output.append({
            "split": key[0],
            "method": key[1],
            "sequences": len(seq_rows),
            "tracklets": tracklets,
            "assigned": assigned,
            "correct": correct,
            "wrong": wrong,
            "abstained": tracklets - assigned,
            "coverage": ratio(assigned, tracklets),
            "accuracy_assigned": ratio(correct, assigned),
            "accuracy_all": ratio(correct, tracklets),
            "gt_in_selected_candidates": sum(
                row["gt_in_selected_candidates"] for row in rows
            ),
            "gt_in_selected_candidates_rate": ratio(
                sum(row["gt_in_selected_candidates"] for row in rows), tracklets
            ),
            "sequence_accuracy_mean": statistics.fmean(sequence_accuracy),
            "sequence_accuracy_std": sample_std(sequence_accuracy),
            "sequence_accuracy_median": statistics.median(sequence_accuracy),
            "sequence_accuracy_q1": quantile(sequence_accuracy, 0.25),
            "sequence_accuracy_q3": quantile(sequence_accuracy, 0.75),
        })
    return output


def build_transitions(predictions, baseline, split_name):
    output = []
    before_rows = predictions[baseline]
    for method, after_rows in predictions.items():
        if method == baseline:
            continue
        for key in sorted(before_rows):
            before = before_rows[key]["pred"]
            after = after_rows[key]["pred"]
            truth = before_rows[key]["gt"]
            output.append({
                "split": split_name,
                "baseline": baseline,
                "candidate": method,
                "sequence": key[0],
                "gt_track_id": key[1],
                "eval_track_id": before_rows[key]["eval_track_id"],
                "gt_jersey": truth,
                "baseline_prediction": "" if before is None else before,
                "candidate_prediction": "" if after is None else after,
                "transition": transition(before, after, truth),
            })
    return output


def build_bootstrap(per_track, baseline, samples, seed):
    methods = sorted({row["method"] for row in per_track})
    sequences = sorted({row["sequence"] for row in per_track})
    by_method_sequence = defaultdict(list)
    for row in per_track:
        by_method_sequence[(row["method"], row["sequence"])].append(row)
    rng = random.Random(seed)
    result = {
        "unit": "sequence",
        "seed": seed,
        "samples": samples,
        "baseline": baseline,
        "sequence_count": len(sequences),
        "comparisons": {},
    }
    for method in methods:
        if method == baseline:
            continue
        distributions = {"accuracy_all": [], "coverage": [], "gt_top5_rate": []}
        for _ in range(samples):
            sampled = [rng.choice(sequences) for _ in sequences]
            for metric in distributions:
                candidate_value = pooled_metric(
                    by_method_sequence, method, sampled, metric
                )
                baseline_value = pooled_metric(
                    by_method_sequence, baseline, sampled, metric
                )
                distributions[metric].append(candidate_value - baseline_value)
        result["comparisons"][method] = {
            metric: summarize_bootstrap(values) for metric, values in distributions.items()
        }
    return result


def pooled_metric(groups, method, sequences, metric):
    rows = [row for sequence in sequences for row in groups[(method, sequence)]]
    if metric == "accuracy_all":
        return ratio(sum(row["correct"] for row in rows), len(rows))
    if metric == "coverage":
        return ratio(sum(row["assigned"] for row in rows), len(rows))
    if metric == "gt_top5_rate":
        return ratio(sum(row["gt_in_selected_candidates"] for row in rows), len(rows))
    raise KeyError(metric)


def summarize_bootstrap(values):
    ordered = sorted(values)
    return {
        "mean_delta": statistics.fmean(values),
        "ci95_low": percentile(ordered, 0.025),
        "ci95_high": percentile(ordered, 0.975),
        "probability_delta_gt_zero": ratio(sum(value > 0 for value in values), len(values)),
    }


def write_latex_tables(
    directory, aggregate, per_sequence, bootstrap, transition_summary, prefix="ocr"
):
    aggregate_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Metodo & Track & Assigned & Correct & Wrong & Coverage & Acc. all \\\\",
        "\\midrule",
    ]
    for row in aggregate:
        aggregate_lines.append(
            f"{latex_escape(row['method'])} & {row['tracklets']} & {row['assigned']} & "
            f"{row['correct']} & {row['wrong']} & {row['coverage']:.3f} & "
            f"{row['accuracy_all']:.3f} \\\\"
        )
    aggregate_lines.extend(["\\bottomrule", "\\end{tabular}"])
    aggregate_name = f"{prefix}_aggregate.tex"
    write_text(directory / aggregate_name, aggregate_lines)

    sequence_lines = [
        "\\begin{longtable}{llrrrr}",
        "\\toprule",
        "Metodo & Sequenza & Track & Assigned & Correct & Acc. all \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule Metodo & Sequenza & Track & Assigned & Correct & Acc. all \\\\ \\midrule",
        "\\endhead",
    ]
    for row in per_sequence:
        sequence_lines.append(
            f"{latex_escape(row['method'])} & {latex_escape(row['sequence'])} & "
            f"{row['tracklets']} & {row['assigned']} & {row['correct']} & "
            f"{row['accuracy_all']:.3f} \\\\"
        )
    sequence_lines.extend(["\\bottomrule", "\\end{longtable}"])
    sequence_name = f"{prefix}_per_sequence.tex"
    write_text(directory / sequence_name, sequence_lines)

    statistics_lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Candidato & Metrica & Delta medio & CI 95\\% low & CI 95\\% high \\\\",
        "\\midrule",
    ]
    for method, metrics in bootstrap["comparisons"].items():
        for metric, values in metrics.items():
            statistics_lines.append(
                f"{latex_escape(method)} & {latex_escape(metric)} & "
                f"{values['mean_delta']:.3f} & {values['ci95_low']:.3f} & "
                f"{values['ci95_high']:.3f} \\\\"
            )
    statistics_lines.extend(["\\bottomrule", "\\end{tabular}"])
    statistics_name = f"{prefix}_statistics.tex"
    write_text(directory / statistics_name, statistics_lines)

    transition_lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Candidato & C$\\rightarrow$C & W$\\rightarrow$C & C$\\rightarrow$W & Altre & $p$ paired \\\\",
        "\\midrule",
    ]
    for method, values in transition_summary["comparisons"].items():
        counts = values["counts"]
        other = sum(counts.values()) - sum(
            counts.get(name, 0)
            for name in ("unchanged_correct", "recovered_correct", "correct_to_wrong")
        )
        transition_lines.append(
            f"{latex_escape(method)} & {counts.get('unchanged_correct', 0)} & "
            f"{counts.get('recovered_correct', 0)} & "
            f"{counts.get('correct_to_wrong', 0)} & {other} & "
            f"{values['paired_binomial_p_value']:.3g} \\\\"
        )
    transition_lines.extend(["\\bottomrule", "\\end{tabular}"])
    transition_name = f"{prefix}_transitions.tex"
    write_text(directory / transition_name, transition_lines)
    return [aggregate_name, sequence_name, statistics_name, transition_name]


def build_transition_summary(transitions):
    groups = defaultdict(list)
    for row in transitions:
        groups[row["candidate"]].append(row)
    comparisons = {}
    for candidate, rows in sorted(groups.items()):
        counts = Counter(row["transition"] for row in rows)
        recoveries = counts["recovered_correct"]
        regressions = counts["correct_to_wrong"]
        discordant = recoveries + regressions
        comparisons[candidate] = {
            "counts": dict(sorted(counts.items())),
            "recoveries": recoveries,
            "regressions": regressions,
            "net_correct_gain": recoveries - regressions,
            "discordant_tracks": discordant,
            "paired_binomial_p_value": exact_binomial_two_sided(
                max(recoveries, regressions), discordant
            ),
        }
    return {"test": "exact_two_sided_binomial_on_discordant_tracks", "comparisons": comparisons}


def exact_binomial_two_sided(successes, trials):
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, value) for value in range(successes, trials + 1))
    return min(1.0, 2.0 * tail / (2 ** trials))


def transition(before, after, truth):
    before_correct = before is not None and before == truth
    after_correct = after is not None and after == truth
    if before == after:
        if before is None:
            return "both_abstain"
        return "unchanged_correct" if before_correct else "unchanged_wrong"
    if before_correct and after is None:
        return "correct_to_abstention"
    if before_correct and not after_correct:
        return "correct_to_wrong"
    if not before_correct and after_correct:
        return "recovered_correct"
    if before is None and after is not None:
        return "new_wrong_emission"
    if before is not None and after is None:
        return "wrong_to_abstention"
    return "wrong_to_wrong"


def file_record(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def integer(value):
    if value in EMPTY:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def numeric(value):
    if value in EMPTY:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def boolean(value):
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def percentile(ordered, probability):
    return quantile(ordered, probability)


def latex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


if __name__ == "__main__":
    main()
