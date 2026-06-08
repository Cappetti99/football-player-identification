#!/usr/bin/env python3
"""Merge a secondary OCR run into a baseline run conservatively.

The script is designed for experiments where an auxiliary OCR preprocessing
recovers some numbers but also corrupts already-correct baseline assignments.
It never overwrites a known baseline identity by default; it only promotes
auxiliary evidence for display IDs that are unknown in the baseline and pass
roster and per-frame duplicate checks.
"""

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ft.identity.roster import load_roster


EMPTY = {"", "None", "unknown", None}


def nonempty(value):
    return value not in EMPTY


def to_int(value):
    if value in EMPTY:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values, q):
    """Return a deterministic percentile for small confidence samples."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def confidence_for_stat(winner, stat):
    if stat == "median":
        return float(winner.get("median_jersey_confidence", 0.0) or 0.0)
    if stat == "p75":
        return float(winner.get("p75_jersey_confidence", 0.0) or 0.0)
    return float(winner.get("mean_jersey_confidence", 0.0) or 0.0)


def load_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def roster_index(roster):
    index = defaultdict(list)
    names = {}
    for player in roster:
        team_id = player.get("team_id")
        jersey = player.get("jersey_number")
        if team_id is None or jersey is None:
            continue
        key = (int(team_id), int(jersey))
        index[key].append(str(player["player_id"]))
        names[str(player["player_id"])] = player.get("name", str(player["player_id"]))
    return index, names


def group_by_display(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        display_id = row.get("display_track_id")
        if display_id is not None:
            grouped[str(display_id)].append(row)
    return grouped


def display_is_unknown(rows):
    if not rows:
        return False
    return all(not nonempty(row.get("player_id")) for row in rows) and all(
        not nonempty(row.get("jersey_number")) for row in rows
    )


def display_winner(rows):
    votes = Counter()
    confidence_sum = defaultdict(float)
    identity_confidence_sum = defaultdict(float)
    confidences = defaultdict(list)
    identity_confidences = defaultdict(list)
    for row in rows:
        team_id = to_int(row.get("team_id"))
        jersey = to_int(row.get("jersey_number"))
        if team_id is None or jersey is None:
            continue
        key = (team_id, jersey)
        confidence = to_float(row.get("jersey_confidence"))
        identity_confidence = to_float(row.get("identity_confidence"))
        votes[key] += 1
        confidence_sum[key] += confidence
        identity_confidence_sum[key] += identity_confidence
        confidences[key].append(confidence)
        identity_confidences[key].append(identity_confidence)
    if not votes:
        return None
    (team_id, jersey), frames = votes.most_common(1)[0]
    key = (team_id, jersey)
    return {
        "team_id": team_id,
        "jersey_number": jersey,
        "support_frames": int(frames),
        "mean_jersey_confidence": confidence_sum[key] / max(1, frames),
        "median_jersey_confidence": percentile(confidences[key], 0.50),
        "p75_jersey_confidence": percentile(confidences[key], 0.75),
        "mean_identity_confidence": identity_confidence_sum[key] / max(1, frames),
        "median_identity_confidence": percentile(identity_confidences[key], 0.50),
    }


def winner_spread(aux_grouped):
    spread = defaultdict(set)
    for display_id, rows in aux_grouped.items():
        winner = display_winner(rows)
        if winner is None:
            continue
        spread[(winner["team_id"], winner["jersey_number"])].add(display_id)
    return {key: len(values) for key, values in spread.items()}


def frame_conflicts(candidate, baseline_rows, target_display_id):
    conflicts = []
    _, conflicts = valid_frames_without_conflicts(candidate, baseline_rows, target_display_id)
    return conflicts


def valid_frames_without_conflicts(candidate, baseline_rows, target_display_id):
    valid_frames = set()
    conflicts = []
    player_id = candidate["player_id"]
    team_id = str(candidate["team_id"])
    jersey = str(candidate["jersey_number"])
    frame_rows = defaultdict(list)
    target_frames = set()
    for row in baseline_rows:
        if row.get("track_group", "players") == "players":
            frame_rows[row.get("frame")].append(row)
            if str(row.get("display_track_id")) == str(target_display_id):
                target_frames.add(row.get("frame"))
    for frame in target_frames:
        frame_has_conflict = False
        for row in frame_rows.get(frame, []):
            if str(row.get("display_track_id")) == str(target_display_id):
                continue
            if nonempty(row.get("player_id")) and row.get("player_id") == player_id:
                frame_has_conflict = True
                conflicts.append(
                    {
                        "frame": to_int(row.get("frame")),
                        "display_track_id": to_int(row.get("display_track_id")),
                        "reason": "duplicate_player_id",
                        "player_id": player_id,
                    }
                )
            if str(row.get("team_id")) == team_id and str(row.get("jersey_number")) == jersey:
                frame_has_conflict = True
                conflicts.append(
                    {
                        "frame": to_int(row.get("frame")),
                        "display_track_id": to_int(row.get("display_track_id")),
                        "reason": "duplicate_team_jersey",
                        "team_id": candidate["team_id"],
                        "jersey_number": candidate["jersey_number"],
                    }
                )
        if not frame_has_conflict:
            valid_frames.add(str(frame))
    return valid_frames, conflicts


def compact_frame_ranges(frames):
    values = sorted(to_int(frame) for frame in frames if to_int(frame) is not None)
    if not values:
        return []
    ranges = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append({"start": int(start), "end": int(previous), "num_frames": int(previous - start + 1)})
        start = previous = value
    ranges.append({"start": int(start), "end": int(previous), "num_frames": int(previous - start + 1)})
    return ranges


def candidate_player_display_counts(aux_grouped, roster_by_team_jersey, spread, args):
    """Count plausible auxiliary display winners per roster player.

    Single digit OCR is especially prone to fragments of two-digit numbers.
    Counting independent display winners for the same player gives a simple
    automatic corroboration signal without hard-coding player IDs or jerseys.
    """
    counts = Counter()
    for rows in aux_grouped.values():
        winner = display_winner(rows)
        if winner is None:
            continue
        confidence = confidence_for_stat(winner, args.jersey_confidence_stat)
        support_score = winner["support_frames"] * confidence
        if winner["support_frames"] < args.min_support_frames and (
            args.min_support_score <= 0.0 or support_score < args.min_support_score
        ):
            continue
        if confidence < args.min_jersey_confidence:
            continue
        key = (winner["team_id"], winner["jersey_number"])
        if spread.get(key, 0) > args.max_display_spread:
            continue
        player_ids = roster_by_team_jersey.get(key, [])
        if args.require_unique_roster_player and len(player_ids) != 1:
            continue
        if player_ids:
            counts[player_ids[0]] += 1
    return counts


def candidate_for_display(
    display_id,
    baseline_rows,
    aux_rows,
    roster_by_team_jersey,
    player_names,
    spread,
    player_display_counts,
    args,
):
    if args.only_unknown and not display_is_unknown(baseline_rows):
        return None, {"reason": "baseline_not_unknown"}
    winner = display_winner(aux_rows)
    if winner is None:
        return None, {"reason": "no_aux_winner"}
    filter_confidence = confidence_for_stat(winner, args.jersey_confidence_stat)
    support_score = winner["support_frames"] * filter_confidence
    winner["filter_jersey_confidence"] = filter_confidence
    winner["jersey_confidence_stat"] = args.jersey_confidence_stat
    winner["support_score"] = support_score
    if winner["support_frames"] < args.min_support_frames:
        if args.min_support_score <= 0.0:
            return None, {"reason": "below_min_support_frames", **winner}
        if support_score < args.min_support_score:
            return None, {
                "reason": "below_min_support_score",
                "min_support_score": float(args.min_support_score),
                **winner,
            }
    if filter_confidence < args.min_jersey_confidence:
        return None, {"reason": "below_min_jersey_confidence", **winner}
    key = (winner["team_id"], winner["jersey_number"])
    if spread.get(key, 0) > args.max_display_spread:
        return None, {"reason": "jersey_display_spread", "display_spread": spread.get(key, 0), **winner}
    player_ids = roster_by_team_jersey.get(key, [])
    if args.require_unique_roster_player and len(player_ids) != 1:
        return None, {"reason": "not_unique_roster_player", "roster_matches": player_ids, **winner}
    if not player_ids:
        return None, {"reason": "not_in_roster", **winner}
    if (
        1 <= int(winner["jersey_number"]) <= 9
        and filter_confidence < args.single_digit_min_jersey_confidence
        and player_display_counts.get(player_ids[0], 0) < args.single_digit_min_player_displays
    ):
        return None, {
            "reason": "single_digit_low_confidence_single_fragment",
            "player_display_count": int(player_display_counts.get(player_ids[0], 0)),
            "single_digit_min_jersey_confidence": float(args.single_digit_min_jersey_confidence),
            "single_digit_min_player_displays": int(args.single_digit_min_player_displays),
            **winner,
        }
    candidate = {
        **winner,
        "display_track_id": int(display_id),
        "player_id": player_ids[0],
        "player_name": player_names.get(player_ids[0], player_ids[0]),
        "source": "auxiliary_ocr_merge",
    }
    return candidate, None


def apply_candidate(row, candidate, apply_player_id):
    row["candidate_player_id"] = candidate["player_id"]
    row["candidate_player_name"] = candidate["player_name"]
    row["candidate_team_id"] = candidate["team_id"]
    row["candidate_jersey_number"] = candidate["jersey_number"]
    row["candidate_confidence"] = round(float(candidate["mean_jersey_confidence"]), 6)
    row["candidate_reason"] = "auxiliary_ocr_unknown_only"
    row["candidate_evidence"] = json.dumps(
        {
            "source": candidate["source"],
            "support_frames": candidate["support_frames"],
            "support_score": candidate.get("support_score"),
            "mean_jersey_confidence": candidate["mean_jersey_confidence"],
            "median_jersey_confidence": candidate.get("median_jersey_confidence"),
            "p75_jersey_confidence": candidate.get("p75_jersey_confidence"),
            "filter_jersey_confidence": candidate.get("filter_jersey_confidence"),
            "jersey_confidence_stat": candidate.get("jersey_confidence_stat"),
            "mean_identity_confidence": candidate["mean_identity_confidence"],
        }
    )
    if not apply_player_id:
        return
    row["team_id"] = candidate["team_id"]
    row["jersey_number"] = candidate["jersey_number"]
    row["jersey_confidence"] = round(float(candidate["mean_jersey_confidence"]), 6)
    row["jersey_votes"] = candidate["support_frames"]
    row["player_id"] = candidate["player_id"]
    row["player_name"] = candidate["player_name"]
    row["identity_confidence"] = round(float(candidate["mean_identity_confidence"]), 6)
    row["identity_evidence"] = json.dumps(
        {
            "source": candidate["source"],
            "reason": "baseline_unknown_auxiliary_high_support",
            "support_frames": candidate["support_frames"],
            "support_score": candidate.get("support_score"),
        }
    )


def copy_metadata_sidecars(source_dir, output_dir, video_id):
    output_dir.mkdir(parents=True, exist_ok=True)
    skip = {
        f"{video_id}_tracklets.csv",
        f"{video_id}_tracklets.json",
        f"{video_id}_constraints.json",
        f"{video_id}_auxiliary_ocr_merge.json",
    }
    for path in source_dir.glob(f"{video_id}_*"):
        if path.name in skip or not path.is_file():
            continue
        shutil.copy2(path, output_dir / path.name)


def merged_constraints(rows, baseline_constraints):
    duplicate_player_frames = []
    duplicate_team_jersey_frames = []
    by_frame = defaultdict(list)
    for row in rows:
        if row.get("track_group", "players") == "players":
            by_frame[row.get("frame")].append(row)
    for frame, frame_rows in by_frame.items():
        player_counts = Counter(row.get("player_id") for row in frame_rows if nonempty(row.get("player_id")))
        for player_id, count in player_counts.items():
            if count > 1:
                duplicate_player_frames.append({"frame": to_int(frame), "player_id": player_id, "count": int(count)})
        team_jersey_counts = Counter(
            (row.get("team_id"), row.get("jersey_number"))
            for row in frame_rows
            if nonempty(row.get("team_id")) and nonempty(row.get("jersey_number"))
        )
        for (team_id, jersey), count in team_jersey_counts.items():
            if count > 1:
                duplicate_team_jersey_frames.append(
                    {
                        "frame": to_int(frame),
                        "team_id": to_int(team_id),
                        "jersey_number": to_int(jersey),
                        "count": int(count),
                    }
                )
    diagnostics = dict(baseline_constraints or {})
    diagnostics.update(
        {
            "source": "auxiliary_ocr_merge",
            "duplicate_player_id": duplicate_player_frames,
            "duplicate_team_jersey": duplicate_team_jersey_frames,
            "duplicate_player_id_count": len(duplicate_player_frames),
            "duplicate_player_frame_count": len(duplicate_player_frames),
            "remaining_duplicate_player_id_count": len(duplicate_player_frames),
            "duplicate_team_jersey_count": len(duplicate_team_jersey_frames),
            "remaining_duplicate_team_jersey_count": len(duplicate_team_jersey_frames),
        }
    )
    return diagnostics


def merge_runs(args):
    metadata_root = args.artifacts_root
    baseline_dir = metadata_root / args.baseline_run / "metadata"
    auxiliary_dir = metadata_root / args.auxiliary_run / "metadata"
    output_dir = metadata_root / args.output_run / "metadata"
    baseline_path = baseline_dir / f"{args.video_id}_tracklets.csv"
    auxiliary_path = auxiliary_dir / f"{args.video_id}_tracklets.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not auxiliary_path.exists():
        raise FileNotFoundError(auxiliary_path)

    baseline_rows = load_csv(baseline_path)
    auxiliary_rows = load_csv(auxiliary_path)
    fieldnames = list(baseline_rows[0].keys()) if baseline_rows else []
    baseline_grouped = group_by_display(baseline_rows)
    auxiliary_grouped = group_by_display(auxiliary_rows)
    roster_by_team_jersey, player_names = roster_index(load_roster(args.roster_path))
    spread = winner_spread(auxiliary_grouped)
    player_display_counts = candidate_player_display_counts(
        auxiliary_grouped,
        roster_by_team_jersey,
        spread,
        args,
    )

    accepted = []
    rejected = {}
    valid_frames_by_display = {}
    for display_id in sorted(set(baseline_grouped) & set(auxiliary_grouped), key=lambda value: int(value)):
        candidate, rejection = candidate_for_display(
            display_id,
            baseline_grouped[display_id],
            auxiliary_grouped[display_id],
            roster_by_team_jersey,
            player_names,
            spread,
            player_display_counts,
            args,
        )
        if rejection:
            rejected[str(display_id)] = rejection
            continue
        valid_frames, conflicts = valid_frames_without_conflicts(candidate, baseline_rows, display_id)
        if conflicts:
            valid_support_score = len(valid_frames) * float(candidate.get("filter_jersey_confidence", 0.0) or 0.0)
            valid_support_pass = len(valid_frames) >= args.min_support_frames or (
                args.min_support_score > 0.0 and valid_support_score >= args.min_support_score
            )
            if args.allow_partial_conflicts and valid_support_pass:
                candidate["partial_conflict_merge"] = True
                candidate["valid_support_frames"] = len(valid_frames)
                candidate["valid_support_score"] = valid_support_score
                candidate["conflict_frame_count"] = len({item.get("frame") for item in conflicts})
                candidate["valid_frame_ranges"] = compact_frame_ranges(valid_frames)
                candidate["conflict_sample"] = conflicts[:20]
                valid_frames_by_display[str(display_id)] = valid_frames
                accepted.append(candidate)
                continue
            rejected[str(display_id)] = {
                "reason": "frame_conflict",
                "conflicts": conflicts[:20],
                "num_conflicts": len(conflicts),
                **candidate,
            }
            continue
        valid_frames_by_display[str(display_id)] = valid_frames
        accepted.append(candidate)

    accepted_by_display = {str(candidate["display_track_id"]): candidate for candidate in accepted}
    merged_rows = []
    applied_rows = 0
    candidate_rows = 0
    for row in baseline_rows:
        out = dict(row)
        candidate = accepted_by_display.get(str(row.get("display_track_id")))
        if candidate and row.get("track_group", "players") == "players":
            valid_frames = valid_frames_by_display.get(str(row.get("display_track_id")))
            if valid_frames is not None and str(row.get("frame")) not in valid_frames:
                merged_rows.append(out)
                continue
            apply_candidate(out, candidate, apply_player_id=args.apply_player_id)
            candidate_rows += 1
            if args.apply_player_id:
                applied_rows += 1
        merged_rows.append(out)

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata_sidecars(baseline_dir, output_dir, args.video_id)
    write_csv(merged_rows, output_dir / f"{args.video_id}_tracklets.csv", fieldnames)
    write_json(merged_rows, output_dir / f"{args.video_id}_tracklets.json")
    constraints = merged_constraints(
        merged_rows,
        load_json(baseline_dir / f"{args.video_id}_constraints.json"),
    )
    write_json(constraints, output_dir / f"{args.video_id}_constraints.json")
    diagnostics = {
        "baseline_run": args.baseline_run,
        "auxiliary_run": args.auxiliary_run,
        "output_run": args.output_run,
        "video_id": args.video_id,
        "apply_player_id": bool(args.apply_player_id),
        "min_support_frames": args.min_support_frames,
        "min_support_score": args.min_support_score,
        "min_jersey_confidence": args.min_jersey_confidence,
        "jersey_confidence_stat": args.jersey_confidence_stat,
        "max_display_spread": args.max_display_spread,
        "single_digit_min_jersey_confidence": args.single_digit_min_jersey_confidence,
        "single_digit_min_player_displays": args.single_digit_min_player_displays,
        "allow_partial_conflicts": bool(args.allow_partial_conflicts),
        "accepted": accepted,
        "rejected": rejected,
        "accepted_display_count": len(accepted),
        "candidate_rows": candidate_rows,
        "applied_rows": applied_rows,
        "constraints": {
            "duplicate_player_frame_count": constraints["duplicate_player_frame_count"],
            "remaining_duplicate_team_jersey_count": constraints["remaining_duplicate_team_jersey_count"],
            "remaining_duplicate_player_id_count": constraints["remaining_duplicate_player_id_count"],
        },
    }
    write_json(diagnostics, output_dir / f"{args.video_id}_auxiliary_ocr_merge.json")
    print(json.dumps(diagnostics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--auxiliary-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--roster-path", required=True, type=Path)
    parser.add_argument("--artifacts-root", default=Path("artifacts/costume-video"), type=Path)
    parser.add_argument("--min-support-frames", type=int, default=200)
    parser.add_argument("--min-support-score", type=float, default=0.0)
    parser.add_argument("--min-jersey-confidence", type=float, default=0.0)
    parser.add_argument("--jersey-confidence-stat", choices=["mean", "median", "p75"], default="mean")
    parser.add_argument("--max-display-spread", type=int, default=2)
    parser.add_argument("--single-digit-min-jersey-confidence", type=float, default=0.0)
    parser.add_argument("--single-digit-min-player-displays", type=int, default=1)
    parser.add_argument("--require-unique-roster-player", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-unknown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-partial-conflicts", action="store_true")
    parser.add_argument("--apply-player-id", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    merge_runs(args)


if __name__ == "__main__":
    main()
