#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Compare identity evidence V1 runs against a baseline.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--output-json")
    parser.add_argument("--assigned-floor", type=int, default=5986)
    parser.add_argument("--unknown-ceiling", type=int, default=9133)
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    baseline = load_run(root, args.baseline_run, args.video_id)
    candidate = load_run(root, args.candidate_run, args.video_id)
    checks = promotion_checks(
        baseline,
        candidate,
        args.video_id,
        assigned_floor=args.assigned_floor,
        unknown_ceiling=args.unknown_ceiling,
    )
    report = {
        "video_id": args.video_id,
        "baseline_run": args.baseline_run,
        "candidate_run": args.candidate_run,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            key: candidate["overview"].get(key, 0) - baseline["overview"].get(key, 0)
            for key in ("player_rows", "assigned_rows", "unknown_rows", "jersey_rows", "propagated_rows")
        },
        "promotion_checks": checks,
        "recommendations": recommendations(candidate, checks),
    }
    print_report(report)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_run(root, run, video_id):
    metadata = root / run / "metadata"
    tracklets = read_csv(metadata / f"{video_id}_tracklets.csv")
    players = [row for row in tracklets if row.get("track_group", "players") == "players"]
    constraints = read_json(metadata / f"{video_id}_constraints.json")
    propagation = read_json(metadata / f"{video_id}_identity_propagation.json")
    prtreid_linking_path = metadata / f"{video_id}_prtreid_linking.json"
    prtreid_linking = read_json(prtreid_linking_path)
    prtreid_bridge_path = metadata / f"{video_id}_prtreid_identity_bridge.json"
    prtreid_bridge = read_json(prtreid_bridge_path)
    scene_cuts = read_json(metadata / f"{video_id}_scene_cuts.json")
    evidence_path = metadata / f"{video_id}_identity_evidence.csv"
    decisions_path = metadata / f"{video_id}_identity_decisions.csv"
    constraint_actions_path = metadata / f"{video_id}_identity_constraint_actions.csv"
    evidence_rows = read_csv(evidence_path)
    decision_rows = read_csv(decisions_path)
    constraint_actions = read_csv(constraint_actions_path)
    assigned = [row for row in players if row.get("player_id") not in ("", "None", "unknown", None)]
    unknown = [row for row in players if row.get("player_id") in ("", "None", "unknown", None)]
    jersey = [row for row in players if row.get("jersey_number") not in ("", "None", None)]
    propagated = [row for row in players if row_is_propagated(row)]
    return {
        "overview": {
            "player_rows": len(players),
            "assigned_rows": len(assigned),
            "unknown_rows": len(unknown),
            "jersey_rows": len(jersey),
            "propagated_rows": len(propagated),
            "unique_display_ids": len({row.get("display_track_id") for row in players}),
        },
        "duplicates": {
            "remaining_duplicate_team_jersey_count": int(constraints.get("remaining_duplicate_team_jersey_count", 0) or 0),
            "remaining_duplicate_player_id_count": int(constraints.get("remaining_duplicate_player_id_count", 0) or 0),
            "duplicate_player_frame_count": int(constraints.get("duplicate_player_frame_count", 0) or 0),
        },
        "risk_flags": risk_flag_counts(players, evidence_rows, decision_rows),
        "constraint_actions": Counter(row.get("action_type") for row in constraint_actions),
        "weak_propagations": weak_propagations(propagation),
        "prtreid_linking": {
            "enabled": bool(prtreid_linking.get("enabled", False)),
            "accepted_links": len(prtreid_linking.get("accepted_links", [])),
            "same_scene_links": sum(row.get("link_type") == "same_scene" for row in prtreid_linking.get("accepted_links", [])),
            "cross_scene_links": sum(row.get("link_type") == "cross_scene" for row in prtreid_linking.get("accepted_links", [])),
        },
        "prtreid_identity_bridge": {
            "enabled": bool(prtreid_bridge.get("enabled", False)),
            "apply": bool(prtreid_bridge.get("apply", False)),
            "proposed_links": len(prtreid_bridge.get("proposed_links", [])),
            "applied_links": len(prtreid_bridge.get("applied_links", [])),
            "applied_rows": int(prtreid_bridge.get("applied_rows", 0) or 0),
        },
        "scene_cut_frames": scene_cuts.get("cut_frames", []),
        "unknown_high_quality": unknown_high_quality_display_ids(unknown),
        "low_confidence_decisions": low_confidence_decisions(decision_rows),
        "artifact_presence": {
            "identity_evidence": evidence_path.exists(),
            "identity_decisions": decisions_path.exists(),
            "constraint_actions": constraint_actions_path.exists(),
            "prtreid_linking": prtreid_linking_path.exists(),
            "prtreid_identity_bridge": prtreid_bridge_path.exists(),
        },
    }


def risk_flag_counts(players, evidence_rows, decision_rows):
    counts = Counter()
    for row in players + evidence_rows + decision_rows:
        for flag in parse_list(row.get("identity_risk_flags") or row.get("risk_flags")):
            counts[str(flag)] += 1
    return dict(counts.most_common())


def row_is_propagated(row):
    if row.get("identity_status") == "propagated":
        return True
    evidence = row.get("identity_evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except (TypeError, ValueError, json.JSONDecodeError):
            evidence = None
    return isinstance(evidence, dict) and evidence.get("status") == "propagated"


def weak_propagations(propagation):
    rows = []
    for item in propagation.get("propagations", []) if isinstance(propagation, dict) else []:
        confidence = float(item.get("assignment_confidence", 0.0) or 0.0)
        partial_fraction = float(item.get("partial_fraction", 1.0) or 0.0)
        conflicts = int(item.get("conflict_frames", 0) or 0)
        if confidence < 0.60 or partial_fraction < 0.50 or conflicts > 0:
            rows.append(
                {
                    "source_display_id": item.get("source_display_id"),
                    "target_display_id": item.get("target_display_id"),
                    "player_id": item.get("player_id"),
                    "assignment_confidence": confidence,
                    "partial_fraction": partial_fraction,
                    "conflict_frames": conflicts,
                }
            )
    return rows[:20]


def unknown_high_quality_display_ids(rows):
    grouped = {}
    for row in rows:
        display_id = row.get("display_track_id")
        grouped.setdefault(display_id, []).append(row)
    output = []
    for display_id, items in grouped.items():
        qualities = [float(row.get("crop_quality", 0.0) or 0.0) for row in items]
        mean_quality = sum(qualities) / len(qualities) if qualities else 0.0
        if len(items) >= 30 and mean_quality >= 0.20:
            output.append({"display_track_id": display_id, "frames": len(items), "mean_crop_quality": round(mean_quality, 4)})
    return sorted(output, key=lambda row: (-row["frames"], -row["mean_crop_quality"]))[:20]


def low_confidence_decisions(rows):
    output = []
    for row in rows:
        status = row.get("identity_status")
        if status not in {"assigned", "propagated"}:
            continue
        confidence = float(row.get("identity_confidence", 0.0) or 0.0)
        if confidence < 0.70:
            output.append(
                {
                    "tracklet_id": row.get("tracklet_id"),
                    "player_id": row.get("player_id"),
                    "identity_status": status,
                    "identity_confidence": confidence,
                    "identity_risk_flags": parse_list(row.get("identity_risk_flags")),
                }
            )
    return sorted(output, key=lambda row: row["identity_confidence"])[:20]


def recommendations(candidate, promotion_checks=None):
    recs = []
    failed_checks = sorted(
        key
        for key, passed in (promotion_checks or {}).items()
        if not passed
    )
    if failed_checks:
        recs.append(
            "Promotion metadata gate failed: "
            + ", ".join(failed_checks)
            + ". Explain the delta with a matched ablation before promotion."
        )
    duplicates = candidate["duplicates"]
    if any(value for value in duplicates.values()):
        recs.append("Inspect duplicate identity/team-jersey constraints before promotion.")
    if candidate["weak_propagations"]:
        recs.append("Inspect weak propagations in video overlay before trusting candidate run.")
    if not all(candidate["artifact_presence"].values()):
        recs.append("Candidate run is missing one or more V1 evidence artifacts.")
    if not recs:
        recs.append("No blocking audit issue detected from metadata.")
    return recs


def promotion_checks(baseline, candidate, video_id, assigned_floor=5986, unknown_ceiling=9133):
    baseline_overview = baseline["overview"]
    candidate_overview = candidate["overview"]
    checks = {
        "assigned_not_lower": candidate_overview["assigned_rows"] >= baseline_overview["assigned_rows"],
        "unknown_not_higher": candidate_overview["unknown_rows"] <= baseline_overview["unknown_rows"],
        "duplicates_zero": not any(candidate["duplicates"].values()),
        "propagation_zero": candidate_overview["propagated_rows"] == 0,
    }
    bridge = candidate.get("prtreid_identity_bridge", {})
    if bridge.get("apply", False):
        assigned_delta = candidate_overview["assigned_rows"] - baseline_overview["assigned_rows"]
        unknown_delta = baseline_overview["unknown_rows"] - candidate_overview["unknown_rows"]
        checks["bridge_rows_match_assigned_delta"] = assigned_delta == int(bridge.get("applied_rows", 0))
        checks["bridge_rows_match_unknown_delta"] = unknown_delta == int(bridge.get("applied_rows", 0))
    if video_id == "Int-Ata":
        checks["expected_scene_cuts"] = candidate.get("scene_cut_frames") == [294, 517, 715, 925, 1052]
        checks[f"assigned_at_least_{int(assigned_floor)}"] = candidate_overview["assigned_rows"] >= int(assigned_floor)
        checks[f"unknown_at_most_{int(unknown_ceiling)}"] = candidate_overview["unknown_rows"] <= int(unknown_ceiling)
    checks["metadata_gate_pass"] = all(checks.values())
    return checks


def print_report(report):
    print(f"=== {report['video_id']} ===")
    print(f"baseline:  {report['baseline_run']}")
    print(f"candidate: {report['candidate_run']}")
    print("\n-- overview --")
    for side in ("baseline", "candidate"):
        print(side, report[side]["overview"])
    print("delta", report["delta"])
    print("promotion_checks", report["promotion_checks"])
    print("\n-- candidate duplicates --")
    print(report["candidate"]["duplicates"])
    print("\n-- candidate risk flags --")
    print(report["candidate"]["risk_flags"])
    print("\n-- weak propagations --")
    for row in report["candidate"]["weak_propagations"][:10]:
        print(row)
    print("\n-- high quality unknown display ids --")
    for row in report["candidate"]["unknown_high_quality"][:10]:
        print(row)
    print("\n-- recommendations --")
    for item in report["recommendations"]:
        print("-", item)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return [item.strip() for item in value.split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else []
    return []


if __name__ == "__main__":
    main()
