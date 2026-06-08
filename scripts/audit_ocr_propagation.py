#!/usr/bin/env python3
"""Explain why OCR jersey assignments did or did not reach final identities."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {"", "None", "unknown", None}


def nonempty(value):
    return value not in EMPTY


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_json_cell(value, default=None):
    if default is None:
        default = {}
    if value in (None, "", "None"):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def player_rows_by_display(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        display_id = str(row.get("display_track_id") or row.get("track_id"))
        grouped[display_id].append(row)
    return grouped


def selected_displays(grouped, ocr, requested, only_final_unknown):
    if requested:
        return [str(display_id) for display_id in requested]
    displays = []
    raw_assignments = ocr.get("assigned_tracklets") or {}
    for display_id, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        if only_final_unknown:
            if any(nonempty(row.get("player_id")) for row in rows):
                continue
            if any(nonempty(row.get("jersey_number")) for row in rows):
                continue
        if str(display_id) in raw_assignments or ocr_entries_for_display(ocr, display_id):
            displays.append(str(display_id))
    return displays


def ocr_entries_for_display(ocr, display_id):
    tracklets = ocr.get("tracklets") or {}
    exact = tracklets.get(str(display_id))
    if exact:
        return [exact]
    prefix = f"{display_id}:"
    return [entry for key, entry in tracklets.items() if str(key).startswith(prefix)]


def ocr_assignment_for_display(ocr, display_id):
    assignments = ocr.get("assigned_tracklets") or {}
    exact = assignments.get(str(display_id))
    if exact:
        return [exact]
    prefix = f"{display_id}:"
    return [assignment for key, assignment in assignments.items() if str(key).startswith(prefix)]


def filter_payload_for_display(filter_data, display_id, bucket):
    values = filter_data.get(bucket) or {}
    exact = values.get(str(display_id))
    if exact:
        return [exact]
    prefix = f"{display_id}:"
    return [payload for key, payload in values.items() if str(key).startswith(prefix)]


def summarize_rows(rows):
    frames = [int(row.get("frame", 0) or 0) for row in rows]
    identity_ids = [
        str(row.get("identity_tracklet_id") or row.get("display_track_id") or row.get("track_id"))
        for row in rows
    ]
    constraints = Counter()
    identity_evidence = Counter()
    for row in rows:
        constraint = parse_json_cell(row.get("jersey_constraint"), {})
        if isinstance(constraint, dict) and constraint.get("reason"):
            constraints[str(constraint.get("reason"))] += 1
        evidence = parse_json_cell(row.get("identity_evidence"), {})
        if isinstance(evidence, dict):
            status = evidence.get("status")
            gate = (evidence.get("assignment_gate") or {}).get("reason") if isinstance(evidence.get("assignment_gate"), dict) else None
            if status or gate:
                identity_evidence[str(gate or status)] += 1
    return {
        "frames": len(rows),
        "start_frame": min(frames) if frames else None,
        "end_frame": max(frames) if frames else None,
        "team_counts": Counter(row.get("team_id") or "unknown" for row in rows).most_common(),
        "role_counts": Counter(row.get("role_detection") or "unknown" for row in rows).most_common(),
        "final_jerseys": Counter(row.get("jersey_number") or "unknown" for row in rows).most_common(),
        "final_player_ids": Counter(row.get("player_id") or "unknown" for row in rows).most_common(),
        "identity_tracklet_ids": Counter(identity_ids).most_common(),
        "jersey_constraint_reasons": constraints.most_common(),
        "identity_evidence_reasons": identity_evidence.most_common(),
    }


def summarize_ocr(ocr, display_id):
    assignments = ocr_assignment_for_display(ocr, display_id)
    entries = ocr_entries_for_display(ocr, display_id)
    decisions = Counter()
    raw_numbers = Counter()
    voting_numbers = Counter()
    raw_rejections = Counter()
    voting_rejections = Counter()
    for entry in entries:
        decision = entry.get("decision") or {}
        decisions[str(decision.get("status") or "missing")] += 1
        raw_numbers.update(as_counter(decision.get("raw_number_counts")))
        voting_numbers.update(as_counter(decision.get("voting_number_counts")))
        raw_rejections.update(as_counter(decision.get("raw_rejection_reasons")))
        voting_rejections.update(as_counter(decision.get("voting_rejection_reasons")))
    return {
        "raw_ocr_assignments": [
            compact_assignment(assignment)
            for assignment in assignments
        ],
        "ocr_decisions": decisions.most_common(),
        "raw_number_counts": raw_numbers.most_common(10),
        "voting_number_counts": voting_numbers.most_common(10),
        "raw_rejections": raw_rejections.most_common(10),
        "voting_rejections": voting_rejections.most_common(10),
    }


def as_counter(value):
    out = Counter()
    if isinstance(value, dict):
        for key, count in value.items():
            try:
                out[str(key)] += int(count)
            except Exception:
                continue
    return out


def compact_assignment(assignment):
    candidates = assignment.get("candidates") or assignment.get("raw_jersey_distribution") or []
    return {
        "jersey_number": assignment.get("jersey_number"),
        "confidence": round_float(assignment.get("confidence")),
        "head_confidence": round_float(assignment.get("head_confidence")),
        "winner_margin": round_float(assignment.get("winner_margin")),
        "winner_score_ratio": round_float(assignment.get("winner_score_ratio")),
        "votes": assignment.get("votes"),
        "total_detections": assignment.get("total_detections"),
        "top_candidates": [
            {
                "jersey_number": candidate.get("jersey_number"),
                "confidence": round_float(candidate.get("confidence")),
                "votes": candidate.get("votes"),
                "score": round_float(candidate.get("score")),
            }
            for candidate in candidates[:8]
        ],
    }


def round_float(value):
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return value


def candidate_rows_for_display(candidate_scores, rows, display_id, limit=5):
    ids = {str(display_id)}
    for row in rows:
        value = row.get("identity_tracklet_id")
        if nonempty(value):
            ids.add(str(value))
    matches = [row for row in candidate_scores if str(row.get("track_id")) in ids]
    matches.sort(key=lambda row: (float(row.get("cost", 1.0) or 1.0), str(row.get("player_id"))))
    out = []
    for row in matches[:limit]:
        gate = parse_json_cell(row.get("assignment_gate"), {})
        components = parse_json_cell(row.get("components"), {})
        out.append(
            {
                "track_id": row.get("track_id"),
                "player_id": row.get("player_id"),
                "player_team_id": row.get("player_team_id"),
                "player_jersey_number": row.get("player_jersey_number"),
                "tracklet_team_id": row.get("tracklet_team_id"),
                "tracklet_jersey_number": row.get("tracklet_jersey_number"),
                "jersey_score_source": row.get("jersey_score_source"),
                "cost": round_float(row.get("cost")),
                "confidence": round_float(row.get("confidence")),
                "gate_reason": gate.get("reason"),
                "gate_pass": gate.get("pass"),
                "components": {
                    key: round_float(value)
                    for key, value in components.items()
                    if abs(float(value or 0.0)) > 1e-9
                },
            }
        )
    return out


def constraint_hits_for_display(constraints, display_id):
    hits = []
    for key, value in constraints.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            if str(item.get("display_track_id")) == str(display_id):
                hits.append(
                    {
                        "constraint": key,
                        "reason": item.get("reason"),
                        "frame": item.get("frame"),
                        "jersey_number": item.get("jersey_number") or item.get("invalid_jersey_number"),
                        "team_id": item.get("team_id") or item.get("previous_team_id"),
                    }
                )
    return hits[:20]


def likely_cause(detail):
    roster_drop = detail.get("roster_filter_dropped") or []
    if roster_drop:
        return "roster_filter:" + str(roster_drop[0].get("reason"))
    gk_drop = detail.get("goalkeeper_ocr_dropped") or []
    if gk_drop:
        return "goalkeeper_ocr_filter:" + str(gk_drop[0].get("reason"))
    row_reasons = detail["final_rows"].get("jersey_constraint_reasons") or []
    if row_reasons:
        return "constraint:" + str(row_reasons[0][0])
    constraint_hits = detail.get("constraint_hits") or []
    if constraint_hits:
        return "constraint:" + str(constraint_hits[0].get("constraint"))
    candidates = detail.get("top_hungarian_candidates") or []
    if candidates:
        reason = candidates[0].get("gate_reason")
        passed = candidates[0].get("gate_pass")
        if not passed and reason:
            return "hungarian_gate:" + str(reason)
    if detail["ocr"].get("raw_ocr_assignments"):
        return "ocr_assigned_but_not_propagated_unclear"
    return "no_ocr_assignment"


def detail_for_display(display_id, rows, ocr, constraints, candidate_scores):
    roster_filter = ocr.get("roster_filter") or {}
    goalkeeper_filter = ocr.get("goalkeeper_ocr_filter") or {}
    detail = {
        "display_track_id": str(display_id),
        "final_rows": summarize_rows(rows),
        "ocr": summarize_ocr(ocr, display_id),
        "roster_filter_dropped": filter_payload_for_display(roster_filter, display_id, "dropped"),
        "roster_filter_degraded": filter_payload_for_display(roster_filter, display_id, "degraded"),
        "goalkeeper_ocr_dropped": filter_payload_for_display(goalkeeper_filter, display_id, "dropped"),
        "constraint_hits": constraint_hits_for_display(constraints, display_id),
        "top_hungarian_candidates": candidate_rows_for_display(candidate_scores, rows, display_id),
    }
    detail["likely_cause"] = likely_cause(detail)
    return detail


def print_summary(details, limit):
    print("displays", len(details))
    print("likely_causes", Counter(item["likely_cause"] for item in details).most_common())
    print()
    for item in details[:limit]:
        rows = item["final_rows"]
        ocr_assignments = item["ocr"]["raw_ocr_assignments"]
        top_ocr = ocr_assignments[0] if ocr_assignments else {}
        top_candidate = (item.get("top_hungarian_candidates") or [{}])[0]
        print("=" * 100)
        print(
            "display",
            item["display_track_id"],
            "frames",
            rows["frames"],
            f"{rows['start_frame']}-{rows['end_frame']}",
            "cause",
            item["likely_cause"],
        )
        print("final_jerseys", rows["final_jerseys"])
        print("final_player_ids", rows["final_player_ids"][:5])
        print("team", rows["team_counts"][:3], "role", rows["role_counts"][:3])
        print("ocr", top_ocr)
        print("ocr_decisions", item["ocr"]["ocr_decisions"])
        print("raw_counts", item["ocr"]["raw_number_counts"])
        print("voting_counts", item["ocr"]["voting_number_counts"])
        print("roster_dropped", item["roster_filter_dropped"][:2])
        print("gk_dropped", item["goalkeeper_ocr_dropped"][:2])
        print("row_constraints", rows["jersey_constraint_reasons"][:5])
        print("constraint_hits", item["constraint_hits"][:5])
        print("best_hungarian", top_candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/costume-video"))
    parser.add_argument("--display", action="append", default=[])
    parser.add_argument("--all-ocr", action="store_true", help="Audit all display IDs with OCR diagnostics, not only final unknowns.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    root = args.artifacts_root / args.run / "metadata"
    rows = load_csv(root / f"{args.video_id}_tracklets.csv")
    ocr = load_json(root / f"{args.video_id}_jersey_ocr.json")
    constraints = load_json(root / f"{args.video_id}_constraints.json")
    candidate_scores = load_csv(root / f"{args.video_id}_candidate_scores.csv")

    grouped = player_rows_by_display(rows)
    display_ids = selected_displays(grouped, ocr, args.display, only_final_unknown=not args.all_ocr)
    details = [
        detail_for_display(display_id, grouped.get(str(display_id), []), ocr, constraints, candidate_scores)
        for display_id in display_ids
    ]
    details.sort(
        key=lambda item: (
            item["likely_cause"],
            -int(item["final_rows"]["frames"] or 0),
            int(item["display_track_id"]),
        )
    )
    print_summary(details, args.limit)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.output_json)


if __name__ == "__main__":
    main()
