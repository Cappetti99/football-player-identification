import json
from collections import defaultdict

from ft.identity.roster import goalkeeper_numbers_by_team, roster_numbers_by_team


EMPTY = {"", "None", "unknown", None}


def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value in EMPTY
    return False


def build_tracklet_evidence(rows, roster=None):
    """Build non-authoritative identity evidence grouped by identity tracklet."""
    numbers_by_team = roster_numbers_by_team(roster or [])
    goalkeeper_numbers = goalkeeper_numbers_by_team(roster or [])
    grouped = defaultdict(list)
    for row in rows or []:
        if row.get("track_group", "players") != "players":
            continue
        tracklet_id = int(row.get("identity_tracklet_id") or row.get("display_track_id", row["track_id"]))
        grouped[tracklet_id].append(row)

    evidence = []
    for tracklet_id, items in sorted(grouped.items()):
        frames = sorted({int(row.get("frame", 0) or 0) for row in items})
        team_id, team_votes = mode_count(row.get("team_id") for row in items if not is_empty(row.get("team_id")))
        jersey_number, jersey_votes = mode_count(
            row.get("jersey_number") for row in items if not is_empty(row.get("jersey_number")) and row.get("jersey_number") != -1
        )
        raw_distribution = aggregate_distribution(items, ("raw_jersey_distribution", "jersey_candidates"))
        roster_distribution = aggregate_distribution(items, ("jersey_distribution",))
        role, role_votes = mode_count(row.get("role_detection") for row in items if not is_empty(row.get("role_detection")))
        semantic_group, _semantic_votes = mode_count(
            row.get("semantic_group_id") for row in items if not is_empty(row.get("semantic_group_id"))
        )
        team_int = to_int(team_id)
        valid_numbers = numbers_by_team.get(team_int, set()) if team_int is not None else set()
        gk_numbers = goalkeeper_numbers.get(team_int, set()) if team_int is not None else set()
        risk_flags = risk_flags_for_tracklet(
            items=items,
            raw_distribution=raw_distribution,
            roster_distribution=roster_distribution,
            valid_numbers=valid_numbers,
            goalkeeper_numbers=gk_numbers,
            role=role,
        )
        evidence.append(
            {
                "tracklet_id": int(tracklet_id),
                "display_track_id": mode(row.get("display_track_id") for row in items if row.get("display_track_id") not in EMPTY),
                "raw_track_ids": sorted({int(row.get("raw_track_id", row["track_id"])) for row in items}),
                "start_frame": int(frames[0]) if frames else None,
                "end_frame": int(frames[-1]) if frames else None,
                "num_frames": int(len(items)),
                "team_evidence": {
                    "team_id": to_int(team_id),
                    "votes": int(team_votes),
                    "mean_confidence": mean(row.get("team_confidence") for row in items if not is_empty(row.get("team_id"))),
                },
                "jersey_evidence": {
                    "winner": to_int(jersey_number),
                    "votes": int(jersey_votes),
                    "mean_confidence": mean(
                        row.get("jersey_confidence") for row in items if row.get("jersey_number") == jersey_number
                    ),
                    "raw_distribution": raw_distribution,
                    "roster_distribution": roster_distribution,
                    "roster_mass": mean(row.get("jersey_roster_mass") for row in items),
                    "valid_team_numbers": sorted(valid_numbers),
                    "goalkeeper_numbers": sorted(gk_numbers),
                },
                "role_evidence": {
                    "role": role,
                    "votes": int(role_votes),
                    "reid_role": mode(row.get("reid_role_detection") for row in items if not is_empty(row.get("reid_role_detection"))),
                    "reid_role_confidence": mean(row.get("reid_role_confidence") for row in items),
                    "semantic_group_id": to_int(semantic_group),
                    "goalkeeper_palette_fraction": mean_bool(row.get("goalkeeper_palette_match") for row in items),
                    "referee_palette_fraction": mean_bool(row.get("referee_palette_match") for row in items),
                },
                "position_evidence": {
                    "mean_pitch_position": mean_position(row.get("position_pitch") for row in items),
                    "mean_image_position": mean_position(row.get("position_image") for row in items),
                },
                "visual_evidence": {
                    "has_embedding": any(not is_empty(row.get("visual_embedding")) for row in items),
                    "reid_model": mode(row.get("reid_model") for row in items if not is_empty(row.get("reid_model"))),
                    "embedding_dim": embedding_dim(row.get("visual_embedding") for row in items),
                },
                "quality_evidence": {
                    "mean_crop_quality": mean(row.get("crop_quality") for row in items),
                    "crop_count": sum(1 for row in items if row.get("crop_path")),
                },
                "risk_flags": risk_flags,
            }
        )
    return evidence


def identity_evidence_rows(evidence):
    rows = []
    for item in evidence or []:
        jersey = item.get("jersey_evidence", {})
        team = item.get("team_evidence", {})
        role = item.get("role_evidence", {})
        quality = item.get("quality_evidence", {})
        rows.append(
            {
                "tracklet_id": item.get("tracklet_id"),
                "display_track_id": item.get("display_track_id"),
                "start_frame": item.get("start_frame"),
                "end_frame": item.get("end_frame"),
                "num_frames": item.get("num_frames"),
                "team_id": team.get("team_id"),
                "team_confidence": team.get("mean_confidence"),
                "jersey_winner": jersey.get("winner"),
                "jersey_votes": jersey.get("votes"),
                "jersey_confidence": jersey.get("mean_confidence"),
                "jersey_roster_mass": jersey.get("roster_mass"),
                "role": role.get("role"),
                "reid_role": role.get("reid_role"),
                "reid_role_confidence": role.get("reid_role_confidence"),
                "semantic_group_id": role.get("semantic_group_id"),
                "reid_model": item.get("visual_evidence", {}).get("reid_model"),
                "visual_embedding_dim": item.get("visual_evidence", {}).get("embedding_dim"),
                "mean_crop_quality": quality.get("mean_crop_quality"),
                "risk_flags": item.get("risk_flags", []),
                "raw_jersey_distribution": jersey.get("raw_distribution", []),
                "roster_jersey_distribution": jersey.get("roster_distribution", []),
            }
        )
    return rows


def aggregate_distribution(items, fields):
    scores = defaultdict(float)
    votes = defaultdict(int)
    for row in items:
        for field in fields:
            distribution = parse_candidates(row.get(field))
            if not distribution:
                continue
            for candidate in distribution:
                number = to_int(candidate.get("jersey_number"))
                if number is None:
                    continue
                scores[number] += float(candidate.get("confidence", 0.0) or 0.0)
                votes[number] += int(candidate.get("votes", 0) or 0)
            break
    total = sum(scores.values())
    if total <= 0:
        return []
    return [
        {
            "jersey_number": int(number),
            "confidence": float(score / total),
            "votes": int(votes[number]),
        }
        for number, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def risk_flags_for_tracklet(items, raw_distribution, roster_distribution, valid_numbers, goalkeeper_numbers, role):
    flags = []
    if not raw_distribution:
        flags.append("missing_raw_ocr")
    if raw_distribution and not roster_distribution:
        flags.append("no_roster_compatible_ocr")
    if len(raw_distribution) >= 2:
        margin = float(raw_distribution[0].get("confidence", 0.0)) - float(raw_distribution[1].get("confidence", 0.0))
        if margin < 0.10:
            flags.append("ambiguous_ocr")
    winner = to_int(raw_distribution[0].get("jersey_number")) if raw_distribution else None
    if winner is not None and valid_numbers and winner not in valid_numbers:
        flags.append("ocr_winner_not_in_team_roster")
    role_name = str(role or "").lower()
    if winner is not None and winner in goalkeeper_numbers and role_name not in {"goalkeeper", "keeper", "gk"}:
        flags.append("goalkeeper_number_on_non_goalkeeper")
    team_values = {to_int(row.get("team_id")) for row in items if not is_empty(row.get("team_id"))}
    if len(team_values) > 1:
        flags.append("mixed_team_votes")
    if mean(row.get("crop_quality") for row in items) < 0.10:
        flags.append("low_crop_quality")
    return flags


def parse_candidates(value):
    if not value:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def mode(values):
    value, _count = mode_count(values)
    return value


def mode_count(values):
    counts = defaultdict(int)
    for value in values:
        if is_empty(value):
            continue
        counts[value] += 1
    if not counts:
        return None, 0
    value, count = max(counts.items(), key=lambda item: item[1])
    return value, int(count)


def mean(values):
    cleaned = []
    for value in values:
        if is_empty(value):
            continue
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(sum(cleaned) / len(cleaned)) if cleaned else 0.0


def mean_bool(values):
    cleaned = [bool(value) for value in values if not is_empty(value)]
    return float(sum(1 for value in cleaned if value) / len(cleaned)) if cleaned else 0.0


def mean_position(values):
    positions = []
    for value in values:
        parsed = parse_position(value)
        if parsed is not None:
            positions.append(parsed)
    if not positions:
        return None
    dims = len(positions[0])
    positions = [position for position in positions if len(position) == dims]
    if not positions:
        return None
    return [float(sum(position[index] for position in positions) / len(positions)) for index in range(dims)]


def parse_position(value):
    if is_empty(value):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def embedding_dim(values):
    for value in values:
        if is_empty(value):
            continue
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def to_int(value):
    if is_empty(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
