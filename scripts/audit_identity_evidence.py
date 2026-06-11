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
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    baseline = load_run(root, args.baseline_run, args.video_id)
    candidate = load_run(root, args.candidate_run, args.video_id)
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
        "recommendations": recommendations(candidate),
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
    evidence_rows = read_csv(metadata / f"{video_id}_identity_evidence.csv")
    decision_rows = read_csv(metadata / f"{video_id}_identity_decisions.csv")
    constraint_actions = read_csv(metadata / f"{video_id}_identity_constraint_actions.csv")
    assigned = [row for row in players if row.get("player_id") not in ("", "None", "unknown", None)]
    unknown = [row for row in players if row.get("player_id") in ("", "None", "unknown", None)]
    jersey = [row for row in players if row.get("jersey_number") not in ("", "None", None)]
    propagated = [row for row in players if row.get("identity_status") == "propagated"]
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
        "unknown_high_quality": unknown_high_quality_display_ids(unknown),
        "low_confidence_decisions": low_confidence_decisions(decision_rows),
        "artifact_presence": {
            "identity_evidence": bool(evidence_rows),
            "identity_decisions": bool(decision_rows),
            "constraint_actions": bool(constraint_actions),
        },
    }


def risk_flag_counts(players, evidence_rows, decision_rows):
    counts = Counter()
    for row in players + evidence_rows + decision_rows:
        for flag in parse_list(row.get("identity_risk_flags") or row.get("risk_flags")):
            counts[str(flag)] += 1
    return dict(counts.most_common())


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


def recommendations(candidate):
    recs = []
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


def print_report(report):
    print(f"=== {report['video_id']} ===")
    print(f"baseline:  {report['baseline_run']}")
    print(f"candidate: {report['candidate_run']}")
    print("\n-- overview --")
    for side in ("baseline", "candidate"):
        print(side, report[side]["overview"])
    print("delta", report["delta"])
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
