import json
from pathlib import Path


def load_roster(roster_path):
    if not roster_path:
        return []
    path = Path(roster_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Roster JSON must be a list of players")
    roster = []
    for row in payload:
        jersey_number = normalize_jersey_number(row.get("jersey_number"))
        metadata = dict(row.get("metadata", {}) or {})
        if "kit_hint" in row and "kit_hint" not in metadata:
            metadata["kit_hint"] = row["kit_hint"]
        kit_hint = metadata.get("kit_hint")
        if isinstance(kit_hint, dict):
            shirt = kit_hint.get("shirt") or kit_hint.get("shirt_color") or kit_hint.get("primary")
            shorts = kit_hint.get("shorts") or kit_hint.get("shorts_color")
            socks = kit_hint.get("socks") or kit_hint.get("socks_color")
            if shirt and "kit_color" not in metadata:
                metadata["kit_color"] = shirt
            if shirt and "shirt_color" not in metadata:
                metadata["shirt_color"] = shirt
            if shorts and "shorts_color" not in metadata:
                metadata["shorts_color"] = shorts
            if socks and "socks_color" not in metadata:
                metadata["socks_color"] = socks
        for key in ("kit_color", "uniform_color", "shirt_color", "referee_color", "color"):
            if key in row and key not in metadata:
                metadata[key] = row[key]
        roster.append(
            {
                "player_id": str(row["player_id"]),
                "name": row.get("name", str(row["player_id"])),
                "team_id": int(row["team_id"]) if row.get("team_id") is not None else None,
                "jersey_number": jersey_number,
                "role": row.get("role"),
                "position_prior": normalize_point(row.get("position_prior")),
                "visual_embedding": row.get("visual_embedding") or row.get("visual_profile"),
                "metadata": metadata,
            }
        )
    return roster


def normalize_jersey_number(value):
    if value is None:
        return None
    number = int(value)
    if number < 1 or number > 99:
        raise ValueError(f"Invalid jersey_number {number}: expected an integer from 1 to 99")
    return number


def normalize_point(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return [float(value["x"]), float(value["y"])] if "x" in value and "y" in value else None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [float(value[0]), float(value[1])]
    return None


def validate_unique_team_jersey(roster):
    seen = {}
    duplicates = []
    for player in roster:
        team_id = player.get("team_id")
        jersey = player.get("jersey_number")
        if team_id is None or jersey is None:
            continue
        key = (int(team_id), int(jersey))
        if key in seen:
            duplicates.append((key, seen[key], player["player_id"]))
        else:
            seen[key] = player["player_id"]
    if duplicates:
        details = ", ".join(
            f"team={team_id} jersey={jersey}: {first}/{second}"
            for (team_id, jersey), first, second in duplicates
        )
        raise ValueError(f"Roster has duplicate players with the same team_id and jersey_number: {details}")


def roster_numbers_by_team(roster):
    numbers = {}
    for player in roster:
        team_id = player.get("team_id")
        jersey = player.get("jersey_number")
        if team_id is None or jersey is None:
            continue
        numbers.setdefault(int(team_id), set()).add(int(jersey))
    return numbers


def goalkeeper_numbers_by_team(roster):
    numbers = {}
    for player in roster:
        role = str(player.get("role") or "").lower()
        team_id = player.get("team_id")
        jersey = player.get("jersey_number")
        if role not in {"goalkeeper", "keeper", "gk"}:
            continue
        if team_id is None or jersey is None:
            continue
        numbers.setdefault(int(team_id), set()).add(int(jersey))
    return numbers
