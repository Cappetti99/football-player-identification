SEMANTIC_GROUPS = {
    1: "team1_players",
    2: "team2_players",
    3: "team1_goalkeeper",
    4: "team2_goalkeeper",
    5: "referees",
}


GROUP_COLORS = {
    1: (40, 140, 255),
    2: (255, 90, 80),
    3: (40, 220, 255),
    4: (255, 170, 80),
    5: (0, 255, 255),
}


class SemanticGroupAssigner:
    """Assign five operational groups without replacing team_id.

    The groups are:
    1. team 1 outfield players
    2. team 2 outfield players
    3. team 1 goalkeeper
    4. team 2 goalkeeper
    5. referees / referee candidates
    """

    def apply(self, tracks):
        for frame_tracks in tracks.get("players", []):
            for raw_id, track in frame_tracks.items():
                group_id = player_group_id(track)
                apply_group(track, group_id)
        for frame_tracks in tracks.get("referees", []):
            for track in frame_tracks.values():
                apply_group(track, 5)
        return {
            "groups": SEMANTIC_GROUPS,
            "colors": {str(key): list(value) for key, value in GROUP_COLORS.items()},
        }


def player_group_id(track):
    role = str(track.get("role_detection") or "").lower()
    team = track.get("team")
    if role in {"referee", "referee_candidate"}:
        return 5
    if role == "goalkeeper":
        if team == 1:
            return 3
        if team == 2:
            return 4
        return 3
    if team == 1:
        return 1
    if team == 2:
        return 2
    return None


def apply_group(track, group_id):
    track["semantic_group_id"] = group_id
    track["semantic_group"] = SEMANTIC_GROUPS.get(group_id, "unknown")
    if group_id in GROUP_COLORS:
        track["semantic_group_color"] = GROUP_COLORS[group_id]
