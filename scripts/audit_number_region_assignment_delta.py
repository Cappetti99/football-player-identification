#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(
        description="Audit assignment deltas caused by scene-scoped Hungarian and number-region cost."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--focus-display-ids", default="9,15,12")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    root = Path(args.artifacts_root)
    baseline_meta = root / args.baseline_run / "metadata"
    candidate_meta = root / args.candidate_run / "metadata"

    baseline_rows = read_csv(baseline_meta / f"{args.video_id}_tracklets.csv")
    candidate_rows = read_csv(candidate_meta / f"{args.video_id}_tracklets.csv")
    candidate_scores = read_csv(candidate_meta / f"{args.video_id}_candidate_scores.csv")
    candidate_summaries = read_csv(candidate_meta / f"{args.video_id}_tracklet_summaries.csv")

    baseline_by_key = row_assignment_by_frame_display(baseline_rows)
    candidate_groups = group_candidate_rows(candidate_rows, baseline_by_key)
    best_scores = best_score_by_track(candidate_scores)
    summaries = {as_int(row.get("track_id")): row for row in candidate_summaries}

    rows = []
    for key, group in candidate_groups.items():
        display_id, scene_id = key
        track_id = scene_track_id(display_id, scene_id)
        score = best_scores.get(track_id) or best_scores.get(display_id) or {}
        summary = summaries.get(track_id) or summaries.get(display_id) or {}
        components = parse_json(score.get("components"), {})
        gate = parse_json(score.get("assignment_gate"), {})
        row = {
            "display_track_id": display_id,
            "scene_segment_id": scene_id,
            "track_id": track_id,
            "start_frame": group["start_frame"],
            "end_frame": group["end_frame"],
            "candidate_rows": group["rows"],
            "candidate_assigned_rows": group["assigned_rows"],
            "baseline_assigned_rows_same_display_frames": group["baseline_assigned_rows"],
            "new_assigned_rows": group["new_assigned_rows"],
            "candidate_player_id": group["player_id"] or "",
            "baseline_player_ids_same_display_frames": sorted(group["baseline_player_ids"]),
            "changed_player_rows": group["changed_player_rows"],
            "top_player_id": score.get("player_id", ""),
            "top_player_jersey": score.get("player_jersey_number", ""),
            "top_cost": to_float(score.get("cost")),
            "top_confidence": to_float(score.get("confidence")),
            "raw_component_sum": sum(float(v or 0.0) for v in components.values()),
            "component_base": components.get("base", 0.0),
            "component_team": components.get("team", 0.0),
            "component_jersey": components.get("jersey", 0.0),
            "component_team_jersey_constraint": components.get("team_jersey_constraint", 0.0),
            "component_number_region": components.get("number_region", 0.0),
            "component_position_prior": components.get("position_prior", 0.0),
            "component_crop_quality": components.get("crop_quality", 0.0),
            "number_region_helped": float(components.get("number_region", 0.0) or 0.0) < 0.0,
            "zero_cost": to_float(score.get("cost")) == 0.0,
            "assignment_gate_pass": gate.get("pass"),
            "assignment_gate_reason": gate.get("reason"),
            "tracklet_team_id": score.get("tracklet_team_id", ""),
            "tracklet_jersey_number": score.get("tracklet_jersey_number", ""),
            "tracklet_number_region_winner": score.get("tracklet_number_region_winner", ""),
            "tracklet_number_region_votes": score.get("tracklet_number_region_votes", ""),
            "tracklet_number_region_confidence": score.get("tracklet_number_region_confidence", ""),
            "tracklet_number_region_consecutive_support": score.get("tracklet_number_region_consecutive_support", ""),
            "summary_frames": summary.get("num_frames", ""),
            "mean_crop_quality": score.get("mean_crop_quality", ""),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -int(row["new_assigned_rows"]),
            not bool(row["number_region_helped"]),
            float(row["top_cost"] if row["top_cost"] is not None else 999.0),
            int(row["display_track_id"]),
            int(row["scene_segment_id"]),
        )
    )

    focus_ids = {int(item) for item in args.focus_display_ids.split(",") if item.strip()}
    focus_rows = [row for row in rows if int(row["display_track_id"]) in focus_ids]
    summary = build_summary(rows, focus_rows)

    out_json = candidate_meta / f"{args.video_id}_number_region_assignment_delta.json"
    out_csv = candidate_meta / f"{args.video_id}_number_region_assignment_delta.csv"
    write_json({"summary": summary, "rows": rows, "focus_rows": focus_rows}, out_json)
    write_csv(rows, out_csv)

    print(json.dumps(summary, indent=2))
    print("\nTop changed groups:")
    for row in rows[: max(0, args.top)]:
        print(
            "display={display_track_id} scene={scene_segment_id} new_rows={new_assigned_rows} "
            "player={candidate_player_id} top={top_player_id} cost={top_cost} "
            "nr={component_number_region} gate={assignment_gate_reason} "
            "nr_winner={tracklet_number_region_winner} votes={tracklet_number_region_votes} "
            "conf={tracklet_number_region_confidence}".format(**row)
        )
    if focus_rows:
        print("\nFocus display IDs:")
        for row in focus_rows:
            print(
                "display={display_track_id} scene={scene_segment_id} new_rows={new_assigned_rows} "
                "player={candidate_player_id} top={top_player_id} cost={top_cost} "
                "raw_sum={raw_component_sum:.4f} nr={component_number_region} "
                "jersey={component_jersey} team={component_team} gate={assignment_gate_reason}".format(**row)
            )
    print(f"\njson={out_json}\ncsv={out_csv}")


def row_assignment_by_frame_display(rows):
    out = {}
    for row in rows:
        display_id = as_int(row.get("display_track_id") or row.get("track_id"))
        frame = as_int(row.get("frame"))
        if display_id is None or frame is None:
            continue
        out[(frame, display_id)] = row.get("player_id") or ""
    return out


def group_candidate_rows(rows, baseline_by_key):
    groups = {}
    for row in rows:
        display_id = as_int(row.get("display_track_id") or row.get("track_id"))
        scene_id = as_int(row.get("scene_segment_id"), default=0)
        frame = as_int(row.get("frame"))
        if display_id is None or frame is None:
            continue
        key = (display_id, scene_id)
        group = groups.setdefault(
            key,
            {
                "rows": 0,
                "assigned_rows": 0,
                "baseline_assigned_rows": 0,
                "new_assigned_rows": 0,
                "changed_player_rows": 0,
                "player_counts": Counter(),
                "baseline_player_ids": set(),
                "start_frame": frame,
                "end_frame": frame,
            },
        )
        group["rows"] += 1
        group["start_frame"] = min(group["start_frame"], frame)
        group["end_frame"] = max(group["end_frame"], frame)
        candidate_player = normalized_player_id(row.get("player_id"))
        baseline_player = normalized_player_id(baseline_by_key.get((frame, display_id), ""))
        if candidate_player:
            group["assigned_rows"] += 1
            group["player_counts"][candidate_player] += 1
        if baseline_player:
            group["baseline_assigned_rows"] += 1
            group["baseline_player_ids"].add(baseline_player)
        if candidate_player and not baseline_player:
            group["new_assigned_rows"] += 1
        if candidate_player and baseline_player and candidate_player != baseline_player:
            group["changed_player_rows"] += 1
    for group in groups.values():
        group["player_id"] = group["player_counts"].most_common(1)[0][0] if group["player_counts"] else ""
    return groups


def best_score_by_track(rows):
    best = {}
    for row in rows:
        track_id = as_int(row.get("track_id"))
        if track_id is None:
            continue
        current = best.get(track_id)
        if current is None or score_sort_key(row) < score_sort_key(current):
            best[track_id] = row
    return best


def score_sort_key(row):
    return (to_float(row.get("cost"), default=999.0), str(row.get("player_id") or ""))


def build_summary(rows, focus_rows):
    total_new = sum(int(row["new_assigned_rows"]) for row in rows)
    helped = [row for row in rows if row["number_region_helped"]]
    zero = [row for row in rows if row["zero_cost"]]
    changed = [row for row in rows if int(row["new_assigned_rows"]) > 0 or int(row["changed_player_rows"]) > 0]
    return {
        "groups": len(rows),
        "changed_groups": len(changed),
        "new_assigned_rows": total_new,
        "changed_player_rows": sum(int(row["changed_player_rows"]) for row in rows),
        "number_region_helped_groups": len(helped),
        "number_region_helped_new_rows": sum(int(row["new_assigned_rows"]) for row in helped),
        "zero_cost_groups": len(zero),
        "zero_cost_new_rows": sum(int(row["new_assigned_rows"]) for row in zero),
        "top_number_region_winners": Counter(
            str(row["tracklet_number_region_winner"])
            for row in rows
            if row["number_region_helped"] and row["tracklet_number_region_winner"] not in ("", None)
        ).most_common(10),
        "top_candidate_players_by_new_rows": weighted_counter(
            (row["candidate_player_id"], int(row["new_assigned_rows"]))
            for row in rows
            if row["candidate_player_id"]
        ).most_common(10),
        "focus_rows": focus_rows,
    }


def scene_track_id(display_id, scene_id):
    return int(display_id) * 100000 + int(scene_id)


def parse_json(value, default):
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def normalized_player_id(value):
    value = str(value or "").strip()
    if value.lower() in {"", "unknown", "none", "nan"}:
        return ""
    return value


def weighted_counter(items):
    counter = Counter()
    for key, weight in items:
        counter[key] += int(weight)
    return counter


def to_float(value, default=None):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(payload, path):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, set)) else value for key, value in row.items()})


if __name__ == "__main__":
    main()
