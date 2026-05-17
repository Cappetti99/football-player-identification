from collections import defaultdict

from ft.features.groups import apply_group
from ft.identity.roster import goalkeeper_numbers_by_team, roster_numbers_by_team


def enforce_identity_constraints(
    tracks,
    roster,
    frame_team_consistency=True,
    frame_team_min_confidence=0.70,
    frame_team_split_enabled=True,
    frame_team_split_min_frames=8,
    frame_team_split_max_gap=2,
):
    """Apply hard consistency constraints to final per-frame identities."""
    numbers_by_team = roster_numbers_by_team(roster)
    goalkeeper_numbers = goalkeeper_numbers_by_team(roster)
    diagnostics = {
        "enabled": bool(roster),
        "invalid_team_jersey": [],
        "duplicate_team_jersey": [],
        "duplicate_player_id": [],
        "semantic_group_corrections": [],
        "goalkeeper_invalid_jersey": [],
        "goalkeeper_only_jersey": [],
        "frame_team_conflicts": [],
        "display_track_splits": [],
    }
    if not tracks.get("players"):
        return diagnostics

    player_roster = {str(player["player_id"]): player for player in roster}
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        if frame_team_consistency:
            _apply_frame_team_consistency(frame_num, frame_tracks, frame_team_min_confidence, diagnostics)
        _clear_invalid_team_jerseys(frame_num, frame_tracks, numbers_by_team, diagnostics)
        _clear_goalkeeper_only_jerseys(frame_num, frame_tracks, goalkeeper_numbers, diagnostics)
        _clear_goalkeeper_invalid_jerseys(frame_num, frame_tracks, goalkeeper_numbers, diagnostics)
        _clear_duplicate_player_ids(frame_num, frame_tracks, player_roster, diagnostics)
        _enforce_semantic_groups(frame_num, frame_tracks, player_roster, diagnostics)
        _clear_duplicate_team_jerseys(frame_num, frame_tracks, diagnostics)
    if frame_team_split_enabled:
        _split_persistent_frame_team_conflicts(
            tracks,
            min_frames=frame_team_split_min_frames,
            max_gap=frame_team_split_max_gap,
            diagnostics=diagnostics,
        )
    diagnostics["invalid_team_jersey_count"] = len(diagnostics["invalid_team_jersey"])
    diagnostics["duplicate_team_jersey_count"] = len(diagnostics["duplicate_team_jersey"])
    diagnostics["duplicate_player_id_count"] = len(diagnostics["duplicate_player_id"])
    diagnostics["duplicate_player_frame_count"] = len(diagnostics["duplicate_player_id"])
    diagnostics["semantic_group_correction_count"] = len(diagnostics["semantic_group_corrections"])
    diagnostics["goalkeeper_invalid_jersey_count"] = len(diagnostics["goalkeeper_invalid_jersey"])
    diagnostics["goalkeeper_only_jersey_count"] = len(diagnostics["goalkeeper_only_jersey"])
    diagnostics["frame_team_conflict_count"] = len(diagnostics["frame_team_conflicts"])
    diagnostics["display_track_split_count"] = len(diagnostics["display_track_splits"])
    diagnostics["remaining_duplicate_team_jersey_count"] = remaining_duplicate_team_jersey_count(tracks)
    diagnostics["remaining_duplicate_player_id_count"] = remaining_duplicate_player_id_count(tracks)
    return diagnostics


def _apply_frame_team_consistency(frame_num, frame_tracks, min_confidence, diagnostics):
    for raw_id, track in frame_tracks.items():
        if str(track.get("role_detection") or "").lower() in {"referee", "referee_candidate", "goalkeeper"}:
            continue
        if track.get("semantic_group_id") in {3, 4, 5}:
            continue
        team = track.get("team")
        frame_team = track.get("frame_team")
        if team in (None, "", "None") or frame_team in (None, "", "None"):
            continue
        try:
            team = int(team)
            frame_team = int(frame_team)
        except (TypeError, ValueError):
            continue
        confidence = float(track.get("frame_team_confidence", 0.0) or 0.0)
        if team == frame_team or confidence < float(min_confidence):
            continue
        diagnostics["frame_team_conflicts"].append(
            {
                "frame": int(frame_num),
                "raw_track_id": int(raw_id),
                "display_track_id": int(track.get("display_track_id", raw_id)),
                "previous_team_id": int(team),
                "frame_team_id": int(frame_team),
                "frame_team_confidence": float(confidence),
                "frame_team_margin": float(track.get("frame_team_margin", 0.0) or 0.0),
                "jersey_number": track.get("jersey_number"),
                "player_id": track.get("player_id", "unknown"),
            }
        )
        track["frame_team_conflict"] = True
        previous_team = track.get("team")
        track["team"] = int(frame_team)
        track["team_confidence"] = max(float(track.get("team_confidence", 0.0) or 0.0), confidence)
        track["team_evidence"] = {
            "source": "frame_team_consistency",
            "previous_team": previous_team,
            "frame_team": int(frame_team),
            "confidence": float(confidence),
            "margin": float(track.get("frame_team_margin", 0.0) or 0.0),
        }
        if track.get("jersey_number") not in (None, "", "None", -1):
            _clear_jersey(
                track,
                {
                    "status": "cleared",
                    "reason": "frame_team_conflict",
                    "previous_team_id": int(team),
                    "frame_team_id": int(frame_team),
                    "frame_team_confidence": float(confidence),
                },
            )
        if track.get("player_id") not in (None, "unknown"):
            _clear_identity(
                track,
                {
                    "status": "cleared",
                    "reason": "frame_team_conflict",
                    "previous_team_id": int(team),
                    "frame_team_id": int(frame_team),
                    "frame_team_confidence": float(confidence),
                },
            )


def _split_persistent_frame_team_conflicts(tracks, min_frames, max_gap, diagnostics):
    min_frames = max(1, int(min_frames or 1))
    max_gap = max(0, int(max_gap or 0))
    next_display_id = max_display_track_id(tracks) + 1
    by_display = defaultdict(list)
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            by_display[display_id].append((frame_num, raw_id, track))

    for display_id, items in sorted(by_display.items()):
        items.sort(key=lambda item: item[0])
        runs = conflict_runs(items, max_gap=max_gap)
        for run in runs:
            if len(run) < min_frames:
                continue
            new_display_id = next_display_id
            next_display_id += 1
            frame_team_counts = defaultdict(int)
            bridge_frames = 0
            for frame_num, raw_id, track in run:
                is_bridge = not bool(track.get("frame_team_conflict", False))
                if is_bridge:
                    bridge_frames += 1
                frame_team = track.get("frame_team")
                if frame_team is not None:
                    frame_team_counts[int(frame_team)] += 1
                track["previous_display_track_id"] = int(display_id)
                track["display_track_id"] = int(new_display_id)
                track["display_split"] = {
                    "status": "split",
                    "reason": "persistent_frame_team_conflict",
                    "previous_display_track_id": int(display_id),
                    "new_display_track_id": int(new_display_id),
                    "start_frame": int(run[0][0]),
                    "end_frame": int(run[-1][0]),
                    "num_frames": int(len(run)),
                }
                if is_bridge:
                    if track.get("jersey_number") not in (None, "", "None", -1):
                        _clear_jersey(
                            track,
                            {
                                "status": "cleared",
                                "reason": "persistent_frame_team_conflict_bridge",
                                "previous_display_track_id": int(display_id),
                                "new_display_track_id": int(new_display_id),
                            },
                        )
                    if track.get("player_id") not in (None, "unknown"):
                        _clear_identity(
                            track,
                            {
                                "status": "cleared",
                                "reason": "persistent_frame_team_conflict_bridge",
                                "previous_display_track_id": int(display_id),
                                "new_display_track_id": int(new_display_id),
                            },
                        )
            diagnostics["display_track_splits"].append(
                {
                    "reason": "persistent_frame_team_conflict",
                    "previous_display_track_id": int(display_id),
                    "new_display_track_id": int(new_display_id),
                    "start_frame": int(run[0][0]),
                    "end_frame": int(run[-1][0]),
                    "num_frames": int(len(run)),
                    "bridge_frames": int(bridge_frames),
                    "frame_team_counts": {str(team): int(count) for team, count in sorted(frame_team_counts.items())},
                }
            )


def conflict_runs(items, max_gap):
    runs = []
    current = []
    bridge = []
    previous_conflict_frame = None
    previous_conflict_team = None
    for frame_num, raw_id, track in items:
        is_conflict = bool(track.get("frame_team_conflict", False))
        if not is_conflict:
            if current:
                bridge.append((frame_num, raw_id, track))
                if len(bridge) > max_gap:
                    runs.append(current)
                    current = []
                    bridge = []
                    previous_conflict_frame = None
                    previous_conflict_team = None
            continue
        frame_team = track.get("frame_team")
        if current and previous_conflict_frame is not None:
            same_conflict_team = previous_conflict_team == frame_team
            within_gap = frame_num - previous_conflict_frame <= max_gap + 1
            if not same_conflict_team or not within_gap:
                runs.append(current)
                current = []
                bridge = []
        if current and bridge:
            current.extend(bridge)
            bridge = []
        elif not current:
            bridge = []
        if current and previous_conflict_frame is not None and frame_num - previous_conflict_frame > max_gap + 1:
            runs.append(current)
            current = []
        current.append((frame_num, raw_id, track))
        previous_conflict_frame = frame_num
        previous_conflict_team = frame_team
    if current:
        runs.append(current)
    return runs


def max_display_track_id(tracks):
    max_id = 0
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            try:
                max_id = max(max_id, int(track.get("display_track_id", raw_id)))
            except (TypeError, ValueError):
                continue
    return max_id


def _clear_invalid_team_jerseys(frame_num, frame_tracks, numbers_by_team, diagnostics):
    for raw_id, track in frame_tracks.items():
        team = track.get("team")
        jersey = track.get("jersey_number")
        if team in (None, "", "None") or jersey in (None, "", "None", -1):
            continue
        valid_numbers = numbers_by_team.get(int(team))
        if not valid_numbers:
            continue
        try:
            jersey = int(jersey)
        except (TypeError, ValueError):
            continue
        if jersey in valid_numbers:
            continue

        diagnostics["invalid_team_jersey"].append(
            {
                "frame": int(frame_num),
                "raw_track_id": int(raw_id),
                "display_track_id": int(track.get("display_track_id", raw_id)),
                "team_id": int(team),
                "jersey_number": int(jersey),
                "valid_numbers": sorted(valid_numbers),
                "player_id": track.get("player_id", "unknown"),
            }
        )
        track["jersey_number"] = None
        track["jersey_confidence"] = 0.0
        track["jersey_votes"] = 0
        track["jersey_constraint"] = {
            "status": "cleared",
            "reason": "number_not_in_team_roster",
            "team_id": int(team),
            "invalid_jersey_number": int(jersey),
            "valid_numbers": sorted(valid_numbers),
        }
        if track.get("player_id") not in (None, "unknown"):
            _clear_identity(
                track,
                {
                    "status": "cleared",
                    "reason": "assigned_jersey_not_in_track_team_roster",
                    "team_id": int(team),
                    "invalid_jersey_number": int(jersey),
                },
            )


def _clear_duplicate_team_jerseys(frame_num, frame_tracks, diagnostics):
    by_team_jersey = defaultdict(list)
    for raw_id, track in frame_tracks.items():
        team = track.get("team")
        jersey = track.get("jersey_number")
        if team in (None, "", "None") or jersey in (None, "", "None", -1):
            continue
        try:
            key = (int(team), int(jersey))
        except (TypeError, ValueError):
            continue
        by_team_jersey[key].append((raw_id, track))

    for (team, jersey), items in by_team_jersey.items():
        if len(items) <= 1:
            continue
        keep_raw_id, keep_track = max(items, key=lambda item: jersey_rank(item[1]))
        for raw_id, track in items:
            if raw_id == keep_raw_id:
                continue
            diagnostics["duplicate_team_jersey"].append(
                {
                    "frame": int(frame_num),
                    "team_id": int(team),
                    "jersey_number": int(jersey),
                    "cleared_raw_track_id": int(raw_id),
                    "kept_raw_track_id": int(keep_raw_id),
                    "cleared_display_track_id": int(track.get("display_track_id", raw_id)),
                    "kept_display_track_id": int(keep_track.get("display_track_id", keep_raw_id)),
                }
            )
            _clear_jersey(
                track,
                {
                    "status": "cleared",
                    "reason": "duplicate_team_jersey_same_frame",
                    "team_id": int(team),
                    "duplicate_jersey_number": int(jersey),
                    "kept_raw_track_id": int(keep_raw_id),
                },
            )
            if track.get("player_id") not in (None, "unknown"):
                _clear_identity(
                    track,
                    {
                        "status": "cleared",
                        "reason": "duplicate_team_jersey_same_frame",
                        "team_id": int(team),
                        "duplicate_jersey_number": int(jersey),
                        "kept_raw_track_id": int(keep_raw_id),
                    },
                )


def _clear_duplicate_player_ids(frame_num, frame_tracks, player_roster, diagnostics):
    by_player = defaultdict(list)
    for raw_id, track in frame_tracks.items():
        player_id = track.get("player_id")
        if player_id in (None, "unknown"):
            continue
        by_player[str(player_id)].append((raw_id, track))

    for player_id, items in by_player.items():
        if len(items) <= 1:
            continue
        keep_raw_id, keep_track = max(items, key=lambda item: identity_rank(item[1]))
        for raw_id, track in items:
            if raw_id == keep_raw_id:
                continue
            diagnostics["duplicate_player_id"].append(
                {
                    "frame": int(frame_num),
                    "player_id": player_id,
                    "cleared_raw_track_id": int(raw_id),
                    "kept_raw_track_id": int(keep_raw_id),
                    "cleared_display_track_id": int(track.get("display_track_id", raw_id)),
                    "kept_display_track_id": int(keep_track.get("display_track_id", keep_raw_id)),
                    "player_roster": player_roster.get(player_id, {}),
                }
            )
            _clear_identity(
                track,
                {
                    "status": "cleared",
                    "reason": "duplicate_player_id_same_frame",
                    "duplicate_player_id": player_id,
                    "kept_raw_track_id": int(keep_raw_id),
                },
            )


def identity_rank(track):
    confidence = float(track.get("identity_confidence", 0.0) or 0.0)
    crop_quality = float(track.get("crop_quality", 0.0) or 0.0)
    bbox = track.get("bbox")
    area = 0.0
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return confidence, crop_quality, area


def jersey_rank(track):
    evidence = track.get("jersey_evidence") or {}
    confidence = float(evidence.get("confidence", track.get("jersey_confidence", 0.0)) or 0.0)
    head_confidence = float(evidence.get("head_confidence", 0.0) or 0.0)
    votes = int(evidence.get("votes", track.get("jersey_votes", 0)) or 0)
    winner_margin = float(evidence.get("winner_margin", 0.0) or 0.0)
    identity_confidence = float(track.get("identity_confidence", 0.0) or 0.0)
    crop_quality = float(track.get("crop_quality", 0.0) or 0.0)
    ref_penalty = -1 if track.get("role_detection") in {"referee", "referee_candidate"} or track.get("semantic_group_id") == 5 else 0
    goalkeeper_bonus = 1 if track.get("semantic_group_id") in {3, 4} or track.get("role_detection") == "goalkeeper" else 0
    bbox = track.get("bbox")
    area = 0.0
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return (
        ref_penalty,
        identity_confidence,
        goalkeeper_bonus,
        confidence,
        head_confidence,
        votes,
        winner_margin,
        crop_quality,
        area,
    )


def _enforce_semantic_groups(frame_num, frame_tracks, player_roster, diagnostics):
    for raw_id, track in frame_tracks.items():
        player_id = track.get("player_id")
        player = player_roster.get(str(player_id)) if player_id not in (None, "unknown") else None
        target_group = semantic_group_from_roster_or_track(player, track)
        if target_group is None:
            continue
        current_group = track.get("semantic_group_id")
        if current_group == target_group:
            continue
        diagnostics["semantic_group_corrections"].append(
            {
                "frame": int(frame_num),
                "raw_track_id": int(raw_id),
                "display_track_id": int(track.get("display_track_id", raw_id)),
                "player_id": player_id,
                "from_group": current_group,
                "to_group": int(target_group),
                "reason": semantic_group_reason(player, track, target_group),
            }
        )
        apply_group(track, target_group)


def _clear_goalkeeper_only_jerseys(frame_num, frame_tracks, goalkeeper_numbers, diagnostics):
    if not goalkeeper_numbers:
        return
    for raw_id, track in frame_tracks.items():
        team = track.get("team")
        jersey = track.get("jersey_number")
        if team in (None, "", "None") or jersey in (None, "", "None", -1):
            continue
        try:
            team = int(team)
            jersey = int(jersey)
        except (TypeError, ValueError):
            continue
        if jersey not in goalkeeper_numbers.get(team, set()):
            continue
        if has_goalkeeper_evidence(track):
            continue
        diagnostics["goalkeeper_only_jersey"].append(
            {
                "frame": int(frame_num),
                "raw_track_id": int(raw_id),
                "display_track_id": int(track.get("display_track_id", raw_id)),
                "team_id": int(team),
                "jersey_number": int(jersey),
                "semantic_group_id": track.get("semantic_group_id"),
                "role_detection": track.get("role_detection"),
            }
        )
        _clear_jersey(
            track,
            {
                "status": "cleared",
                "reason": "goalkeeper_only_jersey_on_non_goalkeeper",
                "team_id": int(team),
                "jersey_number": int(jersey),
            },
        )
        if track.get("player_id") not in (None, "unknown"):
            _clear_identity(
                track,
                {
                    "status": "cleared",
                    "reason": "goalkeeper_only_jersey_on_non_goalkeeper",
                    "team_id": int(team),
                    "jersey_number": int(jersey),
                },
            )


def _clear_goalkeeper_invalid_jerseys(frame_num, frame_tracks, goalkeeper_numbers, diagnostics):
    if not goalkeeper_numbers:
        return
    for raw_id, track in frame_tracks.items():
        if not has_goalkeeper_evidence(track):
            continue
        team = track.get("team")
        jersey = track.get("jersey_number")
        if team in (None, "", "None") or jersey in (None, "", "None", -1):
            continue
        try:
            team = int(team)
            jersey = int(jersey)
        except (TypeError, ValueError):
            continue
        valid_goalkeeper_numbers = goalkeeper_numbers.get(team, set())
        if not valid_goalkeeper_numbers or jersey in valid_goalkeeper_numbers:
            continue
        diagnostics["goalkeeper_invalid_jersey"].append(
            {
                "frame": int(frame_num),
                "raw_track_id": int(raw_id),
                "display_track_id": int(track.get("display_track_id", raw_id)),
                "team_id": int(team),
                "jersey_number": int(jersey),
                "valid_goalkeeper_numbers": sorted(valid_goalkeeper_numbers),
                "semantic_group_id": track.get("semantic_group_id"),
                "role_detection": track.get("role_detection"),
            }
        )
        _clear_jersey(
            track,
            {
                "status": "cleared",
                "reason": "non_goalkeeper_jersey_on_goalkeeper",
                "team_id": int(team),
                "jersey_number": int(jersey),
                "valid_goalkeeper_numbers": sorted(valid_goalkeeper_numbers),
            },
        )
        if track.get("player_id") not in (None, "unknown"):
            _clear_identity(
                track,
                {
                    "status": "cleared",
                    "reason": "non_goalkeeper_jersey_on_goalkeeper",
                    "team_id": int(team),
                    "jersey_number": int(jersey),
                },
            )


def is_goalkeeper_track(track):
    role = str(track.get("role_detection") or "").lower()
    if role in {"goalkeeper", "keeper", "gk"}:
        return True
    return track.get("semantic_group_id") in {3, 4}


def has_goalkeeper_evidence(track):
    role = str(track.get("role_detection") or "").lower()
    if role in {"goalkeeper", "keeper", "gk"}:
        return True
    return bool(track.get("goalkeeper_palette_match", False))


def semantic_group_from_roster_or_track(player, track):
    role = str((player or {}).get("role") or track.get("role_detection") or "").lower()
    team = (player or {}).get("team_id", track.get("team"))
    if role in {"referee", "referee_candidate"}:
        return 5
    if role in {"goalkeeper", "keeper", "gk"}:
        if team == 1:
            return 3
        if team == 2:
            return 4
        return None
    if track.get("role_detection") in {"referee", "referee_candidate"}:
        return 5
    if team == 1:
        return 1
    if team == 2:
        return 2
    return None


def semantic_group_reason(player, track, target_group):
    if player and str(player.get("role") or "").lower() in {"goalkeeper", "keeper", "gk"}:
        return "roster_goalkeeper_role"
    if target_group == 5:
        return "referee_role"
    if player and player.get("team_id") is not None:
        return "roster_team_role"
    return "track_team_role"


def _clear_identity(track, evidence):
    track["player_id"] = "unknown"
    track["player_name"] = "unknown"
    track["identity_confidence"] = 0.0
    track["identity_evidence"] = evidence


def _clear_jersey(track, evidence):
    track["jersey_number"] = None
    track["jersey_confidence"] = 0.0
    track["jersey_votes"] = 0
    track["jersey_evidence"] = None
    track["jersey_candidates"] = None
    track["jersey_distribution"] = None
    track["jersey_roster_mass"] = 0.0
    track["jersey_constraint"] = evidence


def remaining_duplicate_team_jersey_count(tracks):
    count = 0
    for frame_tracks in tracks.get("players", []):
        by_key = defaultdict(int)
        for track in frame_tracks.values():
            team = track.get("team")
            jersey = track.get("jersey_number")
            if team in (None, "", "None") or jersey in (None, "", "None", -1):
                continue
            try:
                by_key[(int(team), int(jersey))] += 1
            except (TypeError, ValueError):
                continue
        count += sum(value - 1 for value in by_key.values() if value > 1)
    return int(count)


def remaining_duplicate_player_id_count(tracks):
    count = 0
    for frame_tracks in tracks.get("players", []):
        by_player = defaultdict(int)
        for track in frame_tracks.values():
            player_id = track.get("player_id")
            if player_id in (None, "unknown"):
                continue
            by_player[str(player_id)] += 1
        count += sum(value - 1 for value in by_player.values() if value > 1)
    return int(count)
