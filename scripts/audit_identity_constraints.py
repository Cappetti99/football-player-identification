#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


csv.field_size_limit(sys.maxsize)
UNKNOWN = {None, "", "None", "unknown"}
GROUP_FIELDS = [
    "action_type", "team_id", "jersey_number", "cleared_player_id", "kept_player_id",
    "preconstraint_assignment_player_ids", "preconstraint_assignment_statuses", "classification",
    "action_records", "unique_rows_cleared", "identity_rows_cleared", "jersey_rows_cleared",
    "cleared_display_ids", "kept_display_ids", "overlap_frame_count", "overlap_frames",
    "cleared_first_frame", "cleared_last_frame", "cleared_identity_confidence",
    "kept_identity_confidence", "cleared_jersey_confidence", "kept_jersey_confidence",
    "cleared_jersey_votes", "kept_jersey_votes", "cleared_jersey_winner_margin",
    "kept_jersey_winner_margin", "final_assigned_rows_on_cleared_displays",
    "final_unknown_rows_on_cleared_displays",
]


def main():
    parser = argparse.ArgumentParser(description="Audit identity rows cleared by hard constraints.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--roster-path", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    metadata = Path(args.artifacts_root) / args.run / "metadata"
    output_json = Path(args.output_json) if args.output_json else metadata / f"{args.video_id}_identity_constraint_audit.json"
    output_csv = Path(args.output_csv) if args.output_csv else metadata / f"{args.video_id}_identity_constraint_audit.csv"
    report = audit_run(metadata, args.video_id, Path(args.roster_path))
    write_json(report, output_json)
    write_csv(report["groups"], output_csv, GROUP_FIELDS)
    print_report(report, output_json, output_csv)


def audit_run(metadata, video_id, roster_path):
    actions_path = metadata / f"{video_id}_identity_constraint_actions.csv"
    tracklets_path = metadata / f"{video_id}_tracklets.csv"
    for required in (actions_path, tracklets_path, roster_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required audit input not found: {required}")
    actions = read_csv(actions_path)
    final_rows = [
        row
        for row in read_csv(tracklets_path)
        if row.get("track_group", "players") == "players"
    ]
    assignments = read_json(metadata / f"{video_id}_identity_assignments.json")
    constraints = read_json(metadata / f"{video_id}_constraints.json")
    linker = read_json(metadata / f"{video_id}_jersey_identity_linking.json")
    provenance = read_json(metadata / f"{video_id}_source_provenance.json")
    manifest = read_json(metadata / f"{video_id}_run_manifest.json")
    roster = read_json(roster_path)
    roster_duplicates = duplicate_roster_jerseys(roster)
    frames_by_display = display_frames(final_rows)
    final_by_display = group_final_rows(final_rows)
    normalized = normalize_actions(actions, frames_by_display)
    groups = aggregate_actions(normalized, final_by_display, assignment_lookup(assignments))
    classifications = Counter(row["classification"] for row in normalized)
    unique_row_keys = {
        (key, row["reason"])
        for row in normalized
        for key in row["row_keys"]
    }
    identity_row_keys = {
        (key, row["reason"])
        for row in normalized
        if not is_unknown(row["cleared_player_id"])
        for key in row["row_keys"]
    }
    jersey_row_keys = {
        (key, row["reason"])
        for row in normalized
        if row.get("jersey_number") is not None
        for key in row["row_keys"]
    }
    return {
        "video_id": video_id,
        "artifact_metadata": str(metadata),
        "summary": {
            "input_actions": len(actions),
            "normalized_actions": len(normalized),
            "unique_cleared_rows": len(unique_row_keys),
            "identity_rows_cleared": len(identity_row_keys),
            "jersey_rows_cleared": len(jersey_row_keys),
            "classifications": dict(sorted(classifications.items())),
            "remaining_duplicate_team_jersey_count": to_int(constraints.get("remaining_duplicate_team_jersey_count"), 0),
            "remaining_duplicate_player_id_count": to_int(constraints.get("remaining_duplicate_player_id_count"), 0),
        },
        "roster": {
            "path": str(roster_path),
            "players": len(roster) if isinstance(roster, list) else 0,
            "duplicate_team_jerseys": roster_duplicates,
            "valid": not roster_duplicates,
        },
        "source_provenance": provenance or manifest.get("source_provenance", {}),
        "jersey_identity_linking": {
            "enabled": bool(linker.get("enabled", False)),
            "accepted_links": len(linker.get("accepted_links", [])),
            "rejection_counts": linker.get("rejection_counts", {}),
            "accepted": linker.get("accepted_links", []),
        },
        "groups": groups,
    }


def normalize_actions(actions, frames_by_display):
    normalized = []
    seen = set()
    for index, action in enumerate(actions):
        reason = str(action.get("reason") or action.get("action_type") or "unknown")
        cleared_player = scalar(action.get("cleared_player_id"), "unknown")
        kept_player = scalar(action.get("kept_player_id"), "unknown")
        row_keys = action_row_keys(action)
        display_ids = int_list(action.get("cleared_display_track_ids"))
        if not display_ids:
            value = to_int(action.get("cleared_display_track_id"))
            display_ids = [value] if value is not None else []
        kept_display_ids = int_list(action.get("kept_display_track_ids"))
        if not kept_display_ids:
            value = to_int(action.get("kept_display_track_id"))
            kept_display_ids = [value] if value is not None else []
        overlap_frames = int_list(action.get("overlap_frames"))
        if not overlap_frames and display_ids and kept_display_ids:
            cleared_frames = set().union(*(frames_by_display.get(value, set()) for value in display_ids))
            kept_frames = set().union(*(frames_by_display.get(value, set()) for value in kept_display_ids))
            overlap_frames = sorted(cleared_frames & kept_frames)
        if not row_keys:
            count = to_int(action.get("cleared_num_rows"), 0)
            row_keys = [("aggregate", index, offset, reason) for offset in range(count)]
        deduped_keys = []
        for key in row_keys:
            dedupe_key = tuple(key) + (reason,)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            deduped_keys.append(tuple(key))
        if not deduped_keys:
            continue
        normalized.append(
            {
                "action_type": str(action.get("action_type") or reason),
                "reason": reason,
                "team_id": to_int(action.get("team_id")),
                "jersey_number": to_int(action.get("jersey_number") or action.get("duplicate_jersey_number")),
                "cleared_player_id": cleared_player,
                "kept_player_id": kept_player,
                "cleared_display_ids": display_ids,
                "kept_display_ids": kept_display_ids,
                "row_keys": deduped_keys,
                "overlap_frames": overlap_frames,
                "classification": classify_action(reason, cleared_player, kept_player),
                "cleared_identity_confidence": to_float(action.get("cleared_identity_confidence")),
                "kept_identity_confidence": to_float(action.get("kept_identity_confidence")),
                "cleared_jersey_confidence": to_float(action.get("cleared_jersey_confidence")),
                "kept_jersey_confidence": to_float(action.get("kept_jersey_confidence")),
                "cleared_jersey_votes": to_float(action.get("cleared_jersey_votes")),
                "kept_jersey_votes": to_float(action.get("kept_jersey_votes")),
                "cleared_jersey_winner_margin": to_float(action.get("cleared_jersey_winner_margin")),
                "kept_jersey_winner_margin": to_float(action.get("kept_jersey_winner_margin")),
            }
        )
    return normalized


def action_row_keys(action):
    keys = []
    for item in json_value(action.get("cleared_row_keys"), []):
        if not isinstance(item, dict):
            continue
        frame = to_int(item.get("frame"))
        raw_id = to_int(item.get("raw_track_id"))
        if frame is not None and raw_id is not None:
            keys.append((frame, raw_id))
    if keys:
        return keys
    frame = to_int(action.get("frame"))
    raw_id = to_int(action.get("cleared_raw_track_id"))
    return [(frame, raw_id)] if frame is not None and raw_id is not None else []


def classify_action(reason, cleared_player, kept_player):
    if reason == "global_duplicate_team_jersey_owner":
        if not is_unknown(cleared_player) and cleared_player == kept_player:
            return "same_player_global_clear_bug"
        if is_unknown(cleared_player) and not is_unknown(kept_player):
            return "unknown_owner_vs_known_owner"
        if not is_unknown(cleared_player) and not is_unknown(kept_player):
            return "competing_known_owners"
        return "global_owner_without_known_identity"
    if reason == "duplicate_team_jersey_same_frame":
        return "same_frame_team_jersey_conflict"
    if reason == "duplicate_player_id_same_frame":
        return "same_frame_player_identity_conflict"
    return "other_constraint_action"


def aggregate_actions(actions, final_by_display, assignments=None):
    assignments = assignments or {}
    grouped = defaultdict(list)
    for row in actions:
        key = (
            row["reason"],
            row["team_id"],
            row["jersey_number"],
            row["cleared_player_id"],
            row["kept_player_id"],
        )
        grouped[key].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(value) for value in pair[0])):
        reason, team, jersey, cleared_player, kept_player = key
        cleared_displays = sorted({value for item in items for value in item["cleared_display_ids"]})
        kept_displays = sorted({value for item in items for value in item["kept_display_ids"]})
        overlap_frames = sorted({value for item in items for value in item["overlap_frames"]})
        row_keys = {value for item in items for value in item["row_keys"]}
        concrete_frames = sorted(
            int(value[0])
            for value in row_keys
            if len(value) == 2 and isinstance(value[0], int)
        )
        final_rows = [row for display in cleared_displays for row in final_by_display.get(display, [])]
        source_assignments = [assignments[display] for display in cleared_displays if display in assignments]
        output.append(
            {
                "action_type": reason,
                "team_id": team,
                "jersey_number": jersey,
                "cleared_player_id": cleared_player,
                "kept_player_id": kept_player,
                "preconstraint_assignment_player_ids": sorted(
                    {
                        str(row.get("player_id"))
                        for row in source_assignments
                        if not is_unknown(row.get("player_id"))
                    }
                ),
                "preconstraint_assignment_statuses": sorted(
                    {str(row.get("identity_status") or row.get("status") or "unknown") for row in source_assignments}
                ),
                "classification": items[0]["classification"],
                "action_records": len(items),
                "unique_rows_cleared": len(row_keys),
                "identity_rows_cleared": len(row_keys) if not is_unknown(cleared_player) else 0,
                "jersey_rows_cleared": len(row_keys) if jersey is not None else 0,
                "cleared_display_ids": cleared_displays,
                "kept_display_ids": kept_displays,
                "overlap_frame_count": len(overlap_frames),
                "overlap_frames": overlap_frames,
                "cleared_first_frame": concrete_frames[0] if concrete_frames else None,
                "cleared_last_frame": concrete_frames[-1] if concrete_frames else None,
                "cleared_identity_confidence": mean_present(items, "cleared_identity_confidence"),
                "kept_identity_confidence": mean_present(items, "kept_identity_confidence"),
                "cleared_jersey_confidence": max_present(items, "cleared_jersey_confidence"),
                "kept_jersey_confidence": max_present(items, "kept_jersey_confidence"),
                "cleared_jersey_votes": max_present(items, "cleared_jersey_votes"),
                "kept_jersey_votes": max_present(items, "kept_jersey_votes"),
                "cleared_jersey_winner_margin": max_present(items, "cleared_jersey_winner_margin"),
                "kept_jersey_winner_margin": max_present(items, "kept_jersey_winner_margin"),
                "final_assigned_rows_on_cleared_displays": sum(not is_unknown(row.get("player_id")) for row in final_rows),
                "final_unknown_rows_on_cleared_displays": sum(is_unknown(row.get("player_id")) for row in final_rows),
            }
        )
    return output


def duplicate_roster_jerseys(roster):
    seen = {}
    duplicates = []
    for player in roster if isinstance(roster, list) else []:
        team = to_int(player.get("team_id"))
        jersey = to_int(player.get("jersey_number"))
        if team is None or jersey is None:
            continue
        key = (team, jersey)
        if key in seen:
            duplicates.append({"team_id": team, "jersey_number": jersey, "player_ids": [seen[key], player.get("player_id")]})
        else:
            seen[key] = player.get("player_id")
    return duplicates


def display_frames(rows):
    output = defaultdict(set)
    for row in rows:
        display = to_int(row.get("display_track_id"))
        frame = to_int(row.get("frame"))
        if display is not None and frame is not None:
            output[display].add(frame)
    return output


def group_final_rows(rows):
    output = defaultdict(list)
    for row in rows:
        display = to_int(row.get("display_track_id"))
        if display is not None:
            output[display].append(row)
    return output


def assignment_lookup(payload):
    raw = payload.get("assignments", payload) if isinstance(payload, dict) else {}
    output = {}
    for key, value in raw.items() if isinstance(raw, dict) else []:
        display = to_int(key)
        if display is not None and isinstance(value, dict):
            output[display] = value
    return output


def mean_present(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return float(sum(values) / len(values)) if values else None


def max_present(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return float(max(values)) if values else None


def int_list(value):
    parsed = json_value(value, [])
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [number for item in parsed if (number := to_int(item)) is not None]


def json_value(value, default):
    if isinstance(value, (list, dict)):
        return value
    if value in UNKNOWN:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def scalar(value, default=None):
    return default if value in UNKNOWN else str(value)


def is_unknown(value):
    return value in UNKNOWN


def to_int(value, default=None):
    if value in UNKNOWN:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value):
    if value in UNKNOWN:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not path.exists():
        return {} if path.suffix == ".json" else []
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(rows, path, fields=None):
    fields = list(fields or (rows[0] if rows else []))
    if not fields:
        raise ValueError("CSV output requires rows or explicit fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def print_report(report, output_json, output_csv):
    print(json.dumps(report["summary"], indent=2))
    print(f"roster_valid={report['roster']['valid']}")
    print(f"json={output_json}")
    print(f"csv={output_csv}")


if __name__ == "__main__":
    main()
