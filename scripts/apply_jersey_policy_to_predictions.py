#!/usr/bin/env python3
"""Apply the shipped jersey decision policy to offline predictions.

The decision policy lives in the pipeline, but the jersey benchmark surface is
offline: ground-truth boxes and track ids, one row per (sequence, gt_track_id).
This adapter bridges the two **without** reimplementing the rule -- it turns
each source's predictions into Evidence, calls the same
``ft.decision.jersey_policy.resolve_jersey_assignments`` the pipeline calls, and
writes a fused predictions.csv that ``aggregate_jersey_thesis_benchmark.py``
consumes unchanged.

That constraint is the point: if the offline evaluation used its own copy of the
rule, the numbers would describe code that is not the code being promoted.

The legacy assignment mapping returned by the policy is unused here -- its keys
are pipeline display-track ids, meaningless on this surface. Only the per-subject
decisions are read, which is what carries the arbitration.

    python scripts/apply_jersey_policy_to_predictions.py \
        --source jersey_ocr_primary=evaluation_outputs/.../shared_surface \
        --source jersey_region_ctc=evaluation_outputs/.../region_ctc \
        --baseline-source jersey_ocr_primary \
        --policy-sources jersey_ocr_primary,jersey_region_ctc \
        --output-dir evaluation_outputs/.../policy_fallback
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ft.core.evidence import Evidence, EvidenceKind, SubjectType, evidence_value  # noqa: E402
from ft.decision.jersey_policy import normalize_policy, resolve_jersey_assignments  # noqa: E402


# Columns the winning source overrides on the baseline row. Anything else stays
# as the baseline wrote it (role, team, num_gt_frames, ...), because those
# describe the surface, not the prediction.
PREDICTION_FIELDS = (
    "pred_jersey_number",
    "assigned",
    "correct",
    "confidence",
    "head_confidence",
    "winner_margin",
    "votes",
    "total_detections",
    "candidates",
)


def subject_of(sequence, gt_track_id):
    """Offline surface key. Not a pipeline display-track id."""
    return f"{sequence}|{gt_track_id}"


def predictions_path(value):
    path = Path(value)
    return path / "predictions.csv" if path.is_dir() else path


def load_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for row in rows:
        key = (str(row["sequence"]), str(row["gt_track_id"]))
        if key in output:
            raise ValueError(f"duplicate prediction key in {path}: {key}")
        output[key] = row
    if not output:
        raise ValueError(f"empty predictions file: {path}")
    return output


def validate_surfaces(sources):
    """Every source must cover the same tracks with the same ground truth."""
    reference_name, reference = next(iter(sources.items()))
    reference_keys = set(reference)
    for name, rows in sources.items():
        keys = set(rows)
        if keys != reference_keys:
            raise SystemExit(
                f"surfaces differ: {reference_name} vs {name}; "
                f"missing={sorted(reference_keys - keys)[:5]} "
                f"extra={sorted(keys - reference_keys)[:5]}"
            )
        mismatched = [
            key
            for key in keys
            if integer(rows[key].get("gt_jersey_number"))
            != integer(reference[key].get("gt_jersey_number"))
        ]
        if mismatched:
            raise SystemExit(
                f"ground truth differs between {reference_name} and {name}: "
                f"{mismatched[:5]}"
            )


def build_evidence(sources, config_hash):
    """One Evidence row per (source, track). Abstentions are kept explicit."""
    evidence = []
    for name, rows in sources.items():
        for (sequence, gt_track_id), row in sorted(rows.items()):
            evidence.append(
                Evidence(
                    subject_type=SubjectType.IDENTITY_TRACKLET,
                    subject_id=subject_of(sequence, gt_track_id),
                    kind=EvidenceKind.JERSEY_NUMBER,
                    value=evidence_value(row.get("pred_jersey_number")),
                    score=number(row.get("confidence")),
                    produced_by=name,
                    config_hash=config_hash,
                    payload={"sequence": sequence, "gt_track_id": gt_track_id},
                )
            )
    return evidence


def fuse(sources, baseline_name, decisions):
    """Rewrite the baseline rows with whichever source the policy picked."""
    baseline = sources[baseline_name]
    by_subject = {
        subject_of(sequence, track): (sequence, track) for sequence, track in baseline
    }
    fused, provenance = [], []
    for decision in decisions:
        key = by_subject.get(decision["subject_id"])
        if key is None:
            continue
        row = dict(baseline[key])
        source = decision["chosen_source"]
        winner_row = sources[source][key] if source else None
        if source and source != baseline_name and winner_row is not None:
            for field in PREDICTION_FIELDS:
                row[field] = winner_row.get(field, "")
            gt = integer(row.get("gt_jersey_number"))
            pred = integer(winner_row.get("pred_jersey_number"))
            row["pred_jersey_number"] = "" if pred is None else pred
            row["assigned"] = pred is not None
            row["correct"] = pred is not None and pred == gt
        row["decision_source"] = source or ""
        row["decision_reason"] = decision["reason"]
        fused.append(row)
        provenance.append(
            {
                "sequence": key[0],
                "gt_track_id": key[1],
                "gt_jersey_number": row.get("gt_jersey_number"),
                "chosen_source": source or "",
                "reason": decision["reason"],
                "available_sources": ";".join(decision["available_sources"]),
                **{
                    f"pred_{name}": rows[key].get("pred_jersey_number", "")
                    for name, rows in sources.items()
                },
            }
        )
    return fused, provenance


def summarize(fused, baseline_rows, baseline_name):
    """Transitions against the baseline. correct->wrong is the gate that matters."""
    counts = Counter()
    for row in fused:
        key = (str(row["sequence"]), str(row["gt_track_id"]))
        gt = integer(row.get("gt_jersey_number"))
        before = integer(baseline_rows[key].get("pred_jersey_number"))
        after = integer(row.get("pred_jersey_number"))
        state = lambda value: (  # noqa: E731 - local, single use
            "abstain" if value is None else ("correct" if value == gt else "wrong")
        )
        counts[f"{state(before)}->{state(after)}"] += 1
    assigned = sum(1 for row in fused if integer(row.get("pred_jersey_number")) is not None)
    correct = sum(
        1
        for row in fused
        if integer(row.get("pred_jersey_number")) is not None
        and integer(row.get("pred_jersey_number")) == integer(row.get("gt_jersey_number"))
    )
    return {
        "baseline_source": baseline_name,
        "tracklets": len(fused),
        "assigned": assigned,
        "correct": correct,
        "coverage": ratio(assigned, len(fused)),
        "accuracy_assigned": ratio(correct, assigned),
        "accuracy_all": ratio(correct, len(fused)),
        "transitions": dict(sorted(counts.items())),
        "chosen_source": dict(
            sorted(Counter(row.get("decision_source", "") for row in fused).items())
        ),
    }


def integer(value):
    if value in (None, "", "None", "null", "nan", "-1"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def number(value):
    if value in (None, "", "None", "null", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def write_csv(path, rows):
    if not rows:
        raise SystemExit(f"refusing to write an empty {path}")
    fields = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_record(path):
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def parse_sources(values):
    output = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --source {value!r}; expected NAME=DIR_OR_CSV")
        name, location = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise SystemExit(f"empty or duplicate source name: {name!r}")
        output[name] = predictions_path(location)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--policy-sources", required=True, help="comma-separated, in priority order")
    parser.add_argument("--on-abstain", default="fallback", choices=["fallback", "abstain"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-note", default="", help="recorded in the manifest, e.g. 'cpu: driver mismatch'")
    args = parser.parse_args()

    paths = parse_sources(args.source)
    if args.baseline_source not in paths:
        raise SystemExit(f"--baseline-source {args.baseline_source} is not among --source names")
    policy = normalize_policy(
        {
            "sources": [name.strip() for name in args.policy_sources.split(",") if name.strip()],
            "on_abstain": args.on_abstain,
        }
    )
    unknown = [name for name in policy["sources"] if name not in paths]
    if unknown:
        raise SystemExit(f"policy references sources with no --source: {unknown}")

    sources = {name: load_predictions(path) for name, path in paths.items()}
    validate_surfaces(sources)

    config_hash = hashlib.sha256(
        json.dumps({"policy": policy, "sources": sorted(paths)}, sort_keys=True).encode()
    ).hexdigest()
    evidence = build_evidence(sources, config_hash)
    _unused_assignments, diagnostics = resolve_jersey_assignments(evidence, policy)

    fused, provenance = fuse(sources, args.baseline_source, diagnostics["decisions"])
    metrics = summarize(fused, sources[args.baseline_source], args.baseline_source)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "predictions.csv", fused)
    write_csv(output / "policy_provenance.csv", provenance)
    manifest = {
        "policy": policy,
        "baseline_source": args.baseline_source,
        "config_hash": config_hash,
        "device_note": args.device_note,
        "inputs": {name: file_record(path) for name, path in paths.items()},
        "metrics": metrics,
        "rule_implementation": "ft.decision.jersey_policy.resolve_jersey_assignments",
    }
    (output / "metrics.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(metrics, indent=2, sort_keys=True))

    # The zero-regression guarantee holds only when the comparison baseline is
    # also the highest-priority source: then no later source can override a
    # track it already decided. If the baseline sits lower in the list, a
    # regression is an expected outcome of the ordering, not a defect.
    regressions = metrics["transitions"].get("correct->wrong", 0)
    baseline_is_first = policy["sources"][0] == args.baseline_source
    if baseline_is_first and regressions:
        print(
            f"\nERROR: {regressions} correct->wrong transitions, but "
            f"{args.baseline_source} is the first policy source, so no later "
            "source can override what it decided. This is a defect in the "
            "adapter or the surfaces, not a result: do not report any number "
            "until it is explained.",
            file=sys.stderr,
        )
        return 1
    if regressions:
        print(
            f"\nNote: {regressions} correct->wrong transitions, expected because "
            f"{policy['sources'][0]} outranks the baseline {args.baseline_source}. "
            "Report them alongside the gains.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
