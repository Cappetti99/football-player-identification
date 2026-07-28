#!/usr/bin/env python3
"""Offline sweep: does lowering identity.reliable_jersey_min_candidate_score
recover coverage on Int-Ata (development split) without hurting precision?

Motivation: a review of ft/identity/hungarian.py found that the config knobs
which look like they gate the assignment "reliable_jersey" decision
(reliable_jersey_min_votes=5, reliable_jersey_min_winner_margin=0.10, consumed
by _tracklet_jersey_reading_is_reliable, renamed from _has_reliable_jersey in
the 2026-07-25 cleanup) are bypassed in the common path. The gate that
assignment_gate() actually calls is _jersey_evidence_supports_player()
(renamed from _has_reliable_jersey_match), which
compares jersey_candidate_score(tracklet, expected_number) against
reliable_jersey_min_candidate_score (default 0.45, +narrow_region_jersey_
score_penalty=0.15 if jersey_full_body_sufficient is False). This is the real
lever. This script replays that comparison at lower thresholds using
already-computed run artifacts -- no new GPU run -- and scores the result
against the frozen Int-Ata ground truth with the project's own
evaluate_identity_units(), the same function evaluate_identity_benchmark.py
uses for real promotion decisions.

Only Int-Ata is used (development split) -- Inter-Juve/Inter-Atalanta are
test/external and must not be touched for tuning, per project convention.

The sweep is monotonic downward: lowering the threshold only adds newly-
passing tracks, it never removes an already-assigned one (the assignment
gate is an OR of reliable_jersey / goalkeeper_singleton / strong_combined,
and cost-matrix pairing does not depend on this threshold at all -- only
whether an already-chosen pairing gets accepted does). So this only needs to
find tracks that flip unknown -> assigned at each threshold and patch those.

Run from ~/FT (PYTHONPATH=/home/cappetti/FT), jersey-yolo-ocr env:
    python3 scripts/sweep_reliable_jersey_min_candidate_score.py \
      --run Int-Ata_identity_evidence_v1_resetbytetrack_cutsensitive_1200f \
      --benchmark-dir evaluation_outputs/identity_benchmark_v1_full \
      --ground-truth-dir evaluation/identity_benchmark_v1_full
"""
import argparse
import ast
import json
from pathlib import Path

from ft.evaluation.identity_benchmark import evaluate_identity_units, identity_metrics, read_csv, read_json
from ft.identity.hungarian import jersey_candidate_score, scene_identity_tracklet_id

THRESHOLDS = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
NARROW_REGION_PENALTY = 0.15


def decode_assignment_key(key):
    try:
        value = int(key)
    except ValueError:
        return None
    return value if value >= 0 else (-value - 1) // 100000


def parse_listy_field(value):
    if isinstance(value, (list, dict)):
        return value
    if not value:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except Exception:
            continue
    return None


def summary_group_key(row, assignment_scope):
    if assignment_scope in {"scene_segment", "scene", "segment"}:
        display_id = row.get("display_track_id") or row.get("track_id")
        segment_id = row.get("scene_segment_id")
        if display_id not in (None, "") and segment_id not in (None, ""):
            return scene_identity_tracklet_id(int(display_id), int(segment_id))
    identity_id = row.get("identity_tracklet_id") or row.get("display_track_id") or row.get("track_id")
    return int(identity_id)


def load_track_evidence(metadata_dir, video_id):
    """Per track_id: current status + best (lowest-cost) candidate + raw OCR evidence."""
    identity_data = read_json(metadata_dir / f"{video_id}_identity_assignments.json")
    assignments = identity_data.get("assignments") or {}

    candidates_path = metadata_dir / f"{video_id}_candidate_scores.csv"
    best_by_track = {}
    for row in read_csv(candidates_path):
        try:
            track_id = int(row["track_id"])
            cost = float(row.get("cost") or 1.0)
        except (KeyError, ValueError, TypeError):
            continue
        if track_id not in best_by_track or cost < best_by_track[track_id]["cost"]:
            best_by_track[track_id] = {**row, "cost": cost}

    jersey_path = metadata_dir / f"{video_id}_jersey_ocr.json"
    jersey_tracks = {}
    if jersey_path.is_file():
        jersey_tracks = read_json(jersey_path).get("tracklets") or {}

    evidence = {}
    for key, entry in assignments.items():
        track_id = decode_assignment_key(key)
        if track_id is None:
            continue
        best = best_by_track.get(track_id)
        full_body_sufficient = (jersey_tracks.get(str(track_id)) or {}).get("full_body_sufficient")
        evidence[track_id] = {
            "already_assigned": entry.get("identity_status") == "assigned",
            "best": best,
            "full_body_sufficient": full_body_sufficient,
        }
    return evidence


def would_pass_at_threshold(item, threshold):
    best = item["best"]
    if not best:
        return False
    expected = best.get("player_jersey_number")
    if expected in (None, ""):
        return False
    raw = parse_listy_field(best.get("tracklet_raw_jersey_distribution"))
    if not raw:
        return False
    score = jersey_candidate_score({"raw_jersey_distribution": raw}, int(float(expected)), field="raw_jersey_distribution")
    if score is None:
        return False
    min_score = threshold
    if item["full_body_sufficient"] is False:
        min_score += NARROW_REGION_PENALTY
    return score >= min_score


def already_assigned_track_ids(rows, assignment_scope):
    """Ground truth for "is this track already identified": read straight from
    tracklets.csv, the actual final pipeline output. identity_assignments.json
    only reflects the raw Hungarian gate stage -- for runs where a later
    propagation/linking stage also fills in player_id, trusting that file
    alone would make already-identified tracks look "unknown" and this script
    would overwrite their real (possibly correct) identity with a guess."""
    assigned = set()
    for row in rows:
        if row.get("track_group", "players") not in ("players", None) and "track_group" in row:
            continue
        if row.get("player_id") in (None, "", "unknown"):
            continue
        try:
            track_id = summary_group_key(row, assignment_scope)
        except (TypeError, ValueError):
            continue
        assigned.add(track_id)
    return assigned


def patch_rows(rows, assignment_scope, flips):
    """flips: track_id -> {player_id, player_name, team_id, jersey_number}."""
    patched = []
    touched = 0
    for row in rows:
        new_row = dict(row)
        if row.get("track_group", "players") == "players" or "track_group" not in row:
            try:
                track_id = summary_group_key(row, assignment_scope)
            except (TypeError, ValueError):
                track_id = None
            if track_id in flips:
                fix = flips[track_id]
                new_row["player_id"] = fix["player_id"]
                new_row["identity_status"] = "assigned"
                new_row["identity_confidence"] = new_row.get("identity_confidence") or "0.5"
                touched += 1
        patched.append(new_row)
    return patched, touched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Existing Int-Ata run dir name under artifacts-root")
    parser.add_argument("--video-id", default="Int-Ata")
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--benchmark-dir", default="evaluation_outputs/identity_benchmark_v1_full")
    parser.add_argument("--ground-truth-dir", default="evaluation/identity_benchmark_v1_full")
    parser.add_argument("--assignment-scope", default=None, help="Override auto-detected scope (scene_segment|global)")
    args = parser.parse_args()

    metadata_dir = Path(args.artifacts_root) / args.run / "metadata"
    manifest = read_json(metadata_dir / f"{args.video_id}_run_manifest.json")
    assignment_scope = args.assignment_scope or (manifest.get("config", {}).get("identity", {}) or {}).get("assignment_scope", "global")
    print(f"assignment_scope={assignment_scope}")

    tracklets_path = metadata_dir / f"{args.video_id}_tracklets.csv"
    baseline_rows = read_csv(tracklets_path)

    evidence = load_track_evidence(metadata_dir, args.video_id)
    assigned_in_csv = already_assigned_track_ids(baseline_rows, assignment_scope)
    unassigned = {tid: item for tid, item in evidence.items() if tid not in assigned_in_csv}
    print(
        f"tracks total: {len(evidence)}, already assigned in tracklets.csv: {len(assigned_in_csv)}, "
        f"eligible for flip test: {len(unassigned)}"
    )

    benchmark = read_json(Path(args.benchmark_dir) / "benchmark_manifest.json")
    ground_truth = read_csv(Path(args.ground_truth_dir) / "ground_truth.csv")

    baseline_results = evaluate_identity_units(benchmark, ground_truth, {args.video_id: baseline_rows})
    baseline_results = [row for row in baseline_results if row["video_id"] == args.video_id]
    baseline_metrics = identity_metrics(baseline_results)
    print("\n=== baseline ===")
    print(json.dumps({k: baseline_metrics[k] for k in (
        "determinate_units", "assigned_units", "correct_units", "wrong_units",
        "identity_precision_unit", "correct_coverage",
    )}, indent=2))

    baseline_by_id = {row["item_id"]: row for row in baseline_results}

    for threshold in THRESHOLDS:
        flips = {}
        for track_id, item in unassigned.items():
            if would_pass_at_threshold(item, threshold):
                best = item["best"]
                flips[track_id] = {
                    "player_id": best["player_id"],
                    "player_name": best.get("player_name"),
                }
        candidate_rows, touched = patch_rows(baseline_rows, assignment_scope, flips)
        candidate_results = evaluate_identity_units(benchmark, ground_truth, {args.video_id: candidate_rows})
        candidate_results = [row for row in candidate_results if row["video_id"] == args.video_id]
        candidate_metrics = identity_metrics(candidate_results)

        candidate_by_id = {row["item_id"]: row for row in candidate_results}
        newly_correct = newly_wrong = 0
        for item_id, base in baseline_by_id.items():
            cand = candidate_by_id.get(item_id)
            if not cand or not base["determinate"] or base["excluded"]:
                continue
            if not base["unit_assigned"] and cand["unit_assigned"]:
                if cand["unit_correct"]:
                    newly_correct += 1
                elif cand["unit_wrong"]:
                    newly_wrong += 1

        print(f"\n--- threshold={threshold} (tracks flipped unknown->assigned: {len(flips)}, rows touched: {touched}) ---")
        print(json.dumps({
            "assigned_units": candidate_metrics["assigned_units"],
            "correct_units": candidate_metrics["correct_units"],
            "wrong_units": candidate_metrics["wrong_units"],
            "identity_precision_unit": candidate_metrics["identity_precision_unit"],
            "correct_coverage": candidate_metrics["correct_coverage"],
            "newly_correct_units": newly_correct,
            "newly_wrong_units": newly_wrong,
        }, indent=2))


if __name__ == "__main__":
    main()
