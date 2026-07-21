#!/usr/bin/env python3
"""Audit field-coordinate gates for raw-track linking without mutating tracks."""

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--frame-matches", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max-gap", type=int, default=90)
    parser.add_argument("--max-speed-mps", type=float, default=12.0)
    parser.add_argument("--pitch-length", type=float, default=105.0)
    parser.add_argument("--pitch-width", type=float, default=68.0)
    args = parser.parse_args()

    tracks = read_tracklets(Path(args.tracklets))
    gt_by_raw = read_gt_majority(Path(args.frame_matches)) if args.frame_matches else {}
    calibration = read_json(Path(args.calibration)) if args.calibration else {}
    rows = build_audit(
        tracks,
        gt_by_raw=gt_by_raw,
        fps=args.fps,
        max_gap=args.max_gap,
        max_speed_mps=args.max_speed_mps,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        calibration_source=calibration.get("source"),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pitch_link_candidates.csv", rows)
    summary = summarize(rows, calibration, args)
    (output / "pitch_link_audit.json").write_text(
        json.dumps({"summary": summary, "candidates": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def read_tracklets(path):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("track_group") or "players") != "players":
                continue
            raw_id = integer(row.get("raw_track_id") or row.get("track_id"))
            if raw_id is None:
                continue
            grouped[raw_id].append({
                "frame": int(row["frame"]),
                "display_id": integer(row.get("display_track_id")) or raw_id,
                "pixel": vector(row.get("position")),
                "pitch": vector(row.get("position_pitch")),
            })
    summaries = {}
    for raw_id, items in grouped.items():
        items.sort(key=lambda item: item["frame"])
        summaries[raw_id] = {
            "raw_id": raw_id,
            "display_id": Counter(item["display_id"] for item in items).most_common(1)[0][0],
            "start": items[0]["frame"],
            "end": items[-1]["frame"],
            "frames": len(items),
            "first_pixel": items[0]["pixel"],
            "last_pixel": items[-1]["pixel"],
            "first_pitch": items[0]["pitch"],
            "last_pitch": items[-1]["pitch"],
        }
    return summaries


def read_gt_majority(path):
    votes = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = integer(row.get("raw_pred_track_id") or row.get("raw_track_id"))
            gt = row.get("gt_track_id")
            if raw is not None and gt not in {None, "", "None", "null"}:
                votes[raw][str(gt)] += 1
    return {raw: counts.most_common(1)[0][0] for raw, counts in votes.items() if counts}


def build_audit(tracks, gt_by_raw=None, fps=25.0, max_gap=90, max_speed_mps=12.0,
                pitch_length=105.0, pitch_width=68.0, calibration_source=None):
    gt_by_raw = gt_by_raw or {}
    ordered = sorted(tracks.values(), key=lambda row: (row["start"], row["raw_id"]))
    rows = []
    for current in ordered:
        for previous in ordered:
            if previous["raw_id"] == current["raw_id"]:
                break
            gap = current["start"] - previous["end"]
            if gap <= 0 or gap > max_gap:
                continue
            pixel_distance = distance(previous["last_pixel"], current["first_pixel"])
            pitch_distance = distance(previous["last_pitch"], current["first_pitch"])
            duration = gap / float(fps) if fps > 0 else None
            speed = pitch_distance / duration if pitch_distance is not None and duration else None
            endpoints_in_bounds = (
                in_bounds(previous["last_pitch"], pitch_length, pitch_width)
                and in_bounds(current["first_pitch"], pitch_length, pitch_width)
            )
            pitch_usable = bool(pitch_distance is not None and endpoints_in_bounds)
            pitch_gate_pass = None if not pitch_usable else speed <= max_speed_mps
            currently_linked = previous["display_id"] == current["display_id"]
            previous_gt, current_gt = gt_by_raw.get(previous["raw_id"]), gt_by_raw.get(current["raw_id"])
            gt_same = None if previous_gt is None or current_gt is None else previous_gt == current_gt
            rows.append({
                "from_raw_track_id": previous["raw_id"],
                "to_raw_track_id": current["raw_id"],
                "from_display_track_id": previous["display_id"],
                "to_display_track_id": current["display_id"],
                "gap_frames": gap,
                "gap_seconds": duration,
                "pixel_distance": pixel_distance,
                "pitch_distance_m": pitch_distance,
                "required_speed_mps": speed,
                "calibration_source": calibration_source,
                "pitch_endpoints_in_bounds": endpoints_in_bounds,
                "pitch_usable": pitch_usable,
                "pitch_gate_pass": pitch_gate_pass,
                "currently_linked": currently_linked,
                "would_block_current_link": bool(currently_linked and pitch_gate_pass is False),
                "from_gt_track_id_offline": previous_gt,
                "to_gt_track_id_offline": current_gt,
                "gt_same_identity_offline": gt_same,
            })
    return rows


def summarize(rows, calibration, args):
    linked = [row for row in rows if row["currently_linked"]]
    blocked = [row for row in linked if row["would_block_current_link"]]
    labeled_blocked = [row for row in blocked if row["gt_same_identity_offline"] is not None]
    return {
        "mode": "audit_only",
        "mutates_tracks": False,
        "calibration": calibration,
        "settings": {"fps": args.fps, "max_gap": args.max_gap, "max_speed_mps": args.max_speed_mps},
        "candidate_pairs": len(rows),
        "pitch_usable_pairs": sum(row["pitch_usable"] for row in rows),
        "currently_linked_pairs": len(linked),
        "would_block_current_links": len(blocked),
        "blocked_wrong_links_offline": sum(row["gt_same_identity_offline"] is False for row in labeled_blocked),
        "blocked_correct_links_offline": sum(row["gt_same_identity_offline"] is True for row in labeled_blocked),
        "offline_gt_usage": "evaluation_only; never used for ranking or gate computation",
    }


def vector(value):
    if value is None or (isinstance(value, str) and value in {"", "None", "null"}):
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def distance(first, second):
    if first is None or second is None:
        return None
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def in_bounds(point, length, width, tolerance=2.0):
    return bool(point is not None and -tolerance <= point[0] <= length + tolerance
                and -tolerance <= point[1] <= width + tolerance)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_csv(path, rows):
    fields = list(rows[0]) if rows else ["from_raw_track_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
