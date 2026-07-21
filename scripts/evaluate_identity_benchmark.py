#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ft.evaluation.identity_benchmark import (
    EMPTY,
    bootstrap_clustered,
    compare_runs,
    evaluate_identity_units,
    identity_metrics,
    paired_precision_delta_interval,
    promotion_gate,
    read_csv,
    read_json,
    sha256_file,
    wilson_interval,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FT identity runs against frozen benchmark V1.")
    parser.add_argument("--benchmark-dir", default="evaluation_outputs/identity_benchmark_v1")
    parser.add_argument("--ground-truth-dir", default="evaluation/identity_benchmark_v1")
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--baseline", action="append", required=True, metavar="VIDEO=RUN")
    parser.add_argument("--candidate", action="append", required=True, metavar="VIDEO=RUN")
    parser.add_argument("--output-dir", default="evaluation_outputs/identity_benchmark_v1/report")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    gt_dir = Path(args.ground_truth_dir)
    artifacts_root = Path(args.artifacts_root)
    manifest_path = benchmark_dir / "benchmark_manifest.json"
    gt_path = gt_dir / "ground_truth.csv"
    gt_manifest_path = gt_dir / "ground_truth_manifest.json"
    manifest = read_json(manifest_path)
    ground_truth = read_csv(gt_path)
    gt_manifest = read_json(gt_manifest_path)
    hashes_match = validate_frozen_inputs(manifest, manifest_path, gt_manifest)
    baseline_runs = parse_run_map(args.baseline)
    candidate_runs = parse_run_map(args.candidate)
    expected_videos = {row["video_id"] for row in manifest["identity_units"]}
    if set(baseline_runs) != expected_videos or set(candidate_runs) != expected_videos:
        raise SystemExit(
            f"baseline/candidate must specify exactly {sorted(expected_videos)}"
        )

    baseline_rows, baseline_meta = load_runs(baseline_runs, artifacts_root, manifest["rosters"])
    candidate_rows, candidate_meta = load_runs(candidate_runs, artifacts_root, manifest["rosters"])
    baseline_results = evaluate_identity_units(
        manifest, ground_truth, baseline_rows,
        iou_threshold=args.iou_threshold, ambiguity_margin=args.ambiguity_margin,
    )
    candidate_results = evaluate_identity_units(
        manifest, ground_truth, candidate_rows,
        iou_threshold=args.iou_threshold, ambiguity_margin=args.ambiguity_margin,
    )
    final_ids = {
        row["item_id"] for row in manifest["identity_units"]
        if row["split"] in {"test", "external"}
    }
    baseline_final = [row for row in baseline_results if row["item_id"] in final_ids]
    candidate_final = [row for row in candidate_results if row["item_id"] in final_ids]
    baseline_metrics = identity_metrics(baseline_final)
    candidate_metrics = identity_metrics(candidate_final)
    delta = compare_runs(baseline_final, candidate_final, ground_truth)
    pair_metrics = evaluate_pairs(manifest, ground_truth, candidate_runs, artifacts_root)
    duplicate_delta = candidate_meta["duplicates"] - baseline_meta["duplicates"]
    violation_delta = violation_total(candidate_meta) - violation_total(baseline_meta)
    preconstraint = evaluate_preconstraint(
        manifest, ground_truth, candidate_runs, artifacts_root, candidate_final,
        args.iou_threshold, args.ambiguity_margin, manifest["rosters"],
    )
    precision_delta_ci95 = paired_precision_delta_interval(
        baseline_final,
        candidate_final,
        iterations=args.bootstrap_iterations,
        seed=manifest["seed"],
    )
    gate = promotion_gate(
        baseline_metrics,
        candidate_metrics,
        delta,
        duplicate_delta=duplicate_delta,
        violation_delta=violation_delta,
        pair_false_positives=pair_metrics["accepted_false_positives"],
        pair_indeterminate=pair_metrics["accepted_indeterminate"] + pair_metrics["uncovered_accepted_links"],
        hashes_match=hashes_match and preconstraint.get("status") == "ok",
        precision_delta_ci95=precision_delta_ci95,
    )
    report = {
        "benchmark_sha256": manifest["benchmark_sha256"],
        "hashes_match": hashes_match,
        "splits_evaluated": ["test", "external"],
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_bootstrap_ci95": bootstrap_clustered(
            candidate_final, iterations=args.bootstrap_iterations, seed=manifest["seed"]
        ),
        "per_video": {
            video: {
                "baseline": identity_metrics([row for row in baseline_final if row["video_id"] == video]),
                "candidate": identity_metrics([row for row in candidate_final if row["video_id"] == video]),
            }
            for video in sorted({row["video_id"] for row in candidate_final})
        },
        "delta": delta,
        "precision_delta_ci95": precision_delta_ci95,
        "pair_metrics": pair_metrics,
        "constraints": {
            "baseline": baseline_meta,
            "candidate": candidate_meta,
            "duplicate_delta": duplicate_delta,
            "violation_delta": violation_delta,
            "preconstraint": preconstraint,
        },
        "promotion_gate": gate,
        "limitations": [
            "Int-Ata is development-only and excluded from final promotion metrics.",
            "Cross-scene generalization remains unproven without a second frozen test video with real cuts.",
        ],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(report, output / "report.json")
    write_csv(candidate_results, output / "candidate_unit_results.csv")
    write_csv(baseline_results, output / "baseline_unit_results.csv")
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"promotion_gate": gate, "baseline": baseline_metrics, "candidate": candidate_metrics}, indent=2))
    print(f"report={output / 'report.json'}")


def parse_run_map(values):
    output = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected VIDEO=RUN, got {value!r}")
        video, run = value.split("=", 1)
        if video in output:
            raise SystemExit(f"duplicate run mapping for {video}")
        output[video] = run
    return output


def load_runs(run_map, artifacts_root, rosters):
    rows_by_video = {}
    duplicate_total = 0
    violations = {}
    artifact_hashes = {}
    for video, run in run_map.items():
        metadata = artifacts_root / run / "metadata"
        tracklets_path = metadata / f"{video}_tracklets.csv"
        manifest_path = metadata / f"{video}_run_manifest.json"
        constraints_path = metadata / f"{video}_constraints.json"
        for required in (tracklets_path, manifest_path, constraints_path):
            if not required.is_file():
                raise FileNotFoundError(f"required evaluation artifact not found: {required}")
            artifact_hashes[str(required)] = sha256_file(required)
        rows = read_csv(tracklets_path)
        rows_by_video[video] = rows
        constraints = read_json(constraints_path)
        duplicates = (
            int(constraints.get("remaining_duplicate_team_jersey_count", 0) or 0)
            + int(constraints.get("remaining_duplicate_player_id_count", 0) or 0)
        )
        duplicate_total += duplicates
        violations[video] = state_violations(rows, rosters.get(video, []))
    return rows_by_video, {
        "duplicates": duplicate_total,
        "violations": violations,
        "artifact_hashes": artifact_hashes,
    }


def state_violations(rows, roster=None):
    players = [row for row in rows if row.get("track_group", "players") == "players" or "track_group" not in row]
    roster_ids = {str(row.get("player_id")) for row in (roster or [])}
    duplicate_player = 0
    duplicate_team_jersey = 0
    invalid_roster_player_rows = sum(
        str(row.get("player_id")) not in roster_ids
        for row in players
        if row.get("player_id") not in EMPTY
    )
    cardinality_violation_frames = 0
    by_frame = defaultdict(list)
    for row in players:
        by_frame[row.get("frame")].append(row)
    for frame_rows in by_frame.values():
        player_counts = Counter(row.get("player_id") for row in frame_rows if row.get("player_id") not in EMPTY)
        duplicate_player += sum(count - 1 for count in player_counts.values() if count > 1)
        jersey_counts = Counter(
            (row.get("team_id"), row.get("jersey_number"))
            for row in frame_rows
            if row.get("team_id") not in EMPTY and row.get("jersey_number") not in EMPTY
        )
        duplicate_team_jersey += sum(count - 1 for count in jersey_counts.values() if count > 1)
        teams = defaultdict(set)
        for row in frame_rows:
            if row.get("team_id") not in EMPTY and row.get("player_id") not in EMPTY:
                teams[str(row["team_id"])].add(str(row["player_id"]))
        cardinality_violation_frames += sum(len(ids) > 11 for ids in teams.values())
    return {
        "duplicate_player_rows": duplicate_player,
        "duplicate_team_jersey_rows": duplicate_team_jersey,
        "invalid_roster_player_rows": invalid_roster_player_rows,
        "team_cardinality_violation_frames": cardinality_violation_frames,
    }


def violation_total(metadata):
    return sum(
        int(value or 0)
        for video in metadata.get("violations", {}).values()
        for value in video.values()
    )


def evaluate_preconstraint(manifest, ground_truth, run_map, artifacts_root, final_results, iou, margin, rosters):
    rows_by_phase = defaultdict(dict)
    missing = []
    violations = defaultdict(dict)
    for video, run in run_map.items():
        path = artifacts_root / run / "metadata" / f"{video}_identity_preconstraint_state.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        state_rows = read_csv(path)
        for phase in sorted({row.get("phase") for row in state_rows}):
            phase_rows = [row for row in state_rows if row.get("phase") == phase]
            rows_by_phase[phase][video] = phase_rows
            violations[phase][video] = state_violations(phase_rows, rosters.get(video, []))
    for video in run_map:
        if video not in rows_by_phase.get("post_assignment_pre_constraints", {}):
            missing.append(f"{video}:post_assignment_pre_constraints")
    if missing:
        return {"status": "missing_artifact", "missing": missing}
    final_by_id = {row["item_id"]: row for row in final_results}
    false_invalidations = set()
    phase_metrics = {}
    for phase, rows in rows_by_phase.items():
        pre = evaluate_identity_units(manifest, ground_truth, rows, iou_threshold=iou, ambiguity_margin=margin)
        phase_metrics[phase] = identity_metrics([
            row for row in pre
            if row["split"] in {"test", "external"} and row["video_id"] in rows
        ])
        false_invalidations.update(
            row["item_id"] for row in pre
            if row["unit_correct"]
            and row["item_id"] in final_by_id
            and not final_by_id[row["item_id"]]["unit_correct"]
        )
    return {
        "status": "ok",
        "false_invalidation_units": len(false_invalidations),
        "false_invalidation_item_ids": sorted(false_invalidations),
        "violations": dict(violations),
        "phase_metrics": phase_metrics,
    }


def evaluate_pairs(manifest, ground_truth, candidate_runs, artifacts_root):
    gt_by_id = {
        row["item_id"]: row for row in ground_truth
        if row.get("item_type") == "pair"
    }
    pairs = [
        row for row in manifest.get("pairs", [])
        if candidate_runs.get(row["video_id"]) == row.get("run")
    ]
    accepted = [
        (row, gt_by_id.get(row["item_id"]))
        for row in pairs if row["status"] == "accepted"
    ]
    accepted_decided = [
        (row, gt) for row, gt in accepted
        if gt and gt.get("pair_label") in {"same", "different"}
    ]
    false_positives = sum(gt["pair_label"] == "different" for _, gt in accepted_decided)
    true_positives = sum(gt["pair_label"] == "same" for _, gt in accepted_decided)
    indeterminate = len(accepted) - len(accepted_decided)
    weighted_same_total = 0.0
    weighted_same_accepted = 0.0
    for row in pairs:
        gt = gt_by_id.get(row["item_id"])
        if not gt or gt.get("pair_label") != "same":
            continue
        weight = 1.0 / max(float(row.get("sampling_probability") or 1.0), 1e-12)
        weighted_same_total += weight
        if row["status"] == "accepted":
            weighted_same_accepted += weight
    uncovered = 0
    represented = {(row["video_id"], row["run"], row["mechanism"]) for row in pairs}
    for source in manifest.get("pair_source_specs", []):
        video = source["video_id"]
        run = candidate_runs.get(video)
        if not run or (video, run, source["mechanism"]) in represented:
            continue
        path = artifacts_root / run / "metadata" / source["artifact"]
        if not path.is_file():
            continue
        accepted_rows, _rejected = pair_records_for_evaluation(
            read_json(path), source["mechanism"]
        )
        uncovered += len(accepted_rows)
    return {
        "accepted_pairs": len(accepted),
        "accepted_decided": len(accepted_decided),
        "accepted_true_positives": true_positives,
        "accepted_false_positives": false_positives,
        "accepted_indeterminate": indeterminate,
        "uncovered_accepted_links": uncovered,
        "precision": true_positives / len(accepted_decided) if accepted_decided else None,
        "precision_ci95": wilson_interval(true_positives, len(accepted_decided)),
        "weighted_recall_estimate": (
            weighted_same_accepted / weighted_same_total if weighted_same_total else None
        ),
    }


def pair_records_for_evaluation(payload, mechanism):
    if mechanism == "prtreid_linking":
        return payload.get("accepted_links", []), payload.get("rejected_links", [])
    if mechanism == "prtreid_identity_bridge":
        return payload.get("applied_links", []), payload.get("rejected_links", [])
    if mechanism == "jersey_identity_linking":
        return payload.get("accepted_links", []), payload.get("rejected_links", [])
    return payload.get("propagations", []), payload.get("rejected_propagations", [])


def validate_frozen_inputs(manifest, manifest_path, gt_manifest):
    if gt_manifest.get("benchmark_sha256") != manifest.get("benchmark_sha256"):
        raise SystemExit("ground truth benchmark hash does not match benchmark manifest")
    if gt_manifest.get("benchmark_manifest_sha256") != sha256_file(manifest_path):
        raise SystemExit("ground truth references a different benchmark_manifest.json")
    mismatches = []
    for path, expected in manifest.get("artifact_hashes", {}).items():
        file_path = Path(path)
        if not file_path.is_file() or sha256_file(file_path) != expected:
            mismatches.append(path)
    if mismatches:
        raise SystemExit("frozen benchmark artifacts changed or are missing:\n- " + "\n- ".join(mismatches))
    return True


def markdown_report(report):
    gate = report["promotion_gate"]
    base = report["baseline"]
    candidate = report["candidate"]
    return f"""# Identity Benchmark V1

- Promotion status: **{gate['status']}**
- Benchmark: `{report['benchmark_sha256']}`
- Baseline unit precision: {format_metric(base.get('identity_precision_unit'))}
- Candidate unit precision: {format_metric(candidate.get('identity_precision_unit'))}
- Baseline correct coverage: {format_metric(base.get('correct_coverage'))}
- Candidate correct coverage: {format_metric(candidate.get('correct_coverage'))}
- New false positives: {report['delta']['new_false_positives']}
- New indeterminate decisions: {report['delta']['new_indeterminate']}
- Pair false positives: {report['pair_metrics']['accepted_false_positives']}

## Promotion checks

""" + "\n".join(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in gate["checks"].items()
    ) + "\n"


def format_metric(value):
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


if __name__ == "__main__":
    main()
