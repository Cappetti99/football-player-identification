from collections import defaultdict

REFEREE_ROLES = {"referee", "referee_candidate", "official", "match_official"}


def build_identity_candidates(
    candidate_scores,
    tracks=None,
    enabled=True,
    min_confidence=0.35,
    max_cost=0.85,
    min_margin=0.0,
    max_jersey_display_spread=None,
    display_spread_scope="team",
    display_spread_only_unknown=True,
):
    """Select one diagnostic identity candidate for each still-unknown track.

    Candidates are deliberately non-authoritative. They expose the best roster
    hypothesis for analysis and annotation review, but never overwrite the real
    player identity assigned by Hungarian plus constraints.
    """
    if not enabled:
        return {}

    assigned_track_ids = assigned_display_track_ids(tracks) if tracks is not None else set()
    blocked_track_ids = referee_like_track_ids(tracks) if tracks is not None else set()
    grouped = defaultdict(list)
    for row in candidate_scores or []:
        try:
            track_id = int(row["track_id"])
        except Exception:
            continue
        if track_id in assigned_track_ids:
            continue
        if track_id in blocked_track_ids:
            continue
        if is_referee_roster_candidate(row):
            continue
        grouped[track_id].append(row)

    selected = {}
    for track_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (float(row.get("cost", 1.0) or 1.0), str(row.get("player_id"))))
        if not rows:
            continue
        best = rows[0]
        second_cost = float(rows[1].get("cost", 1.0) or 1.0) if len(rows) > 1 else 1.0
        cost = float(best.get("cost", 1.0) or 1.0)
        confidence = float(best.get("confidence", 0.0) or 0.0)
        margin = max(0.0, second_cost - cost)
        if confidence < float(min_confidence):
            continue
        if cost > float(max_cost):
            continue
        if margin < float(min_margin):
            continue

        selected[track_id] = {
            "candidate_player_id": best.get("player_id"),
            "candidate_player_name": best.get("player_name", best.get("player_id")),
            "candidate_team_id": best.get("player_team_id"),
            "candidate_jersey_number": best.get("player_jersey_number"),
            "candidate_confidence": confidence,
            "candidate_cost": cost,
            "candidate_margin": margin,
            "candidate_reason": candidate_reason(best),
            "candidate_evidence": {
                "assignment_gate": best.get("assignment_gate"),
                "components": best.get("components"),
                "tracklet_team_id": best.get("tracklet_team_id"),
                "tracklet_team_confidence": best.get("tracklet_team_confidence"),
                "tracklet_jersey_number": best.get("tracklet_jersey_number"),
                "tracklet_jersey_confidence": best.get("tracklet_jersey_confidence"),
                "tracklet_jersey_votes": best.get("tracklet_jersey_votes"),
                "tracklet_frames": best.get("tracklet_frames"),
                "position_prior_distance": best.get("position_prior_distance"),
                "visual_similarity": best.get("visual_similarity"),
            },
        }
    spread = candidate_jersey_display_spread(selected, scope=display_spread_scope)
    candidates = {}
    for track_id, candidate in selected.items():
        jersey = candidate.get("candidate_jersey_number")
        if max_jersey_display_spread not in (None, "", 0, "0") and jersey not in (None, "", "None"):
            key = candidate_jersey_spread_key(candidate, scope=display_spread_scope)
            if int(spread.get(key, 0)) > int(max_jersey_display_spread):
                continue
        if spread:
            evidence = candidate.setdefault("candidate_evidence", {})
            evidence["candidate_jersey_display_spread"] = int(
                spread.get(candidate_jersey_spread_key(candidate, scope=display_spread_scope), 0)
            )
            evidence["candidate_jersey_display_spread_scope"] = str(display_spread_scope or "global")
            evidence["candidate_jersey_display_spread_only_unknown"] = bool(display_spread_only_unknown)
        candidates[track_id] = candidate
    return candidates


def candidate_jersey_display_spread(candidates, scope="team"):
    """Count how many unknown tracklets selected each jersey candidate.

    Candidate fallback is diagnostic and should not promote numbers that behave
    like OCR attractors across many display IDs. Counting after the first-pass
    best-candidate selection catches cases where the same jersey is proposed
    for many unrelated unknown tracklets in one video.
    """
    by_key = defaultdict(set)
    for track_id, candidate in (candidates or {}).items():
        key = candidate_jersey_spread_key(candidate, scope=scope)
        if key is None:
            continue
        by_key[key].add(int(track_id))
    return {key: len(track_ids) for key, track_ids in by_key.items()}


def candidate_jersey_spread_key(candidate, scope="team"):
    jersey = candidate.get("candidate_jersey_number")
    if jersey in (None, "", "None"):
        return None
    try:
        jersey = int(jersey)
    except (TypeError, ValueError):
        return None
    if str(scope or "team").lower() == "team":
        team = candidate.get("candidate_team_id")
        try:
            team = int(team)
        except (TypeError, ValueError):
            team = None
        return (team, jersey)
    return jersey


def apply_identity_candidates(tracks, candidates):
    """Attach non-authoritative identity candidates to still-unknown tracks."""
    if not candidates:
        return
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            tracklet_id = int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id))
            candidate = candidates.get(tracklet_id)
            if not candidate:
                continue
            if track.get("player_id") not in (None, "", "unknown"):
                continue
            track.update(candidate)


def identity_candidate_rows(candidates):
    """Convert candidate diagnostics into stable CSV/JSON rows."""
    rows = []
    for track_id, candidate in sorted((candidates or {}).items()):
        row = {"track_id": int(track_id)}
        row.update(candidate)
        rows.append(row)
    return rows


def assigned_display_track_ids(tracks):
    """Return tracklet ids that already have an authoritative identity."""
    assigned = set()
    for frame_tracks in (tracks or {}).get("players", []):
        for raw_id, track in frame_tracks.items():
            if track.get("player_id") in (None, "", "unknown"):
                continue
            assigned.add(int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id)))
    return assigned


def referee_like_track_ids(tracks, min_referee_score=0.25):
    """Find tracklets where player candidates would likely describe the referee.

    Referee detections can pass through the players branch so downstream
    exports remain uniform. Candidate identities are only diagnostic, but
    writing player candidates on referee-like display ids makes audits noisy
    and was observed to turn yellow referee tracks into roster players.
    """
    stats = defaultdict(lambda: {"total": 0, "referee_like": 0, "strong": False})
    for group in ("players", "referees"):
        for frame_tracks in (tracks or {}).get(group, []):
            for raw_id, track in frame_tracks.items():
                tracklet_id = int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id))
                stats[tracklet_id]["total"] += 1
                role = str(track.get("role_detection") or "").lower()
                referee_like = (
                    group == "referees"
                    or role in REFEREE_ROLES
                    or track.get("semantic_group_id") == 5
                    or bool(track.get("referee_palette_match", False))
                    or float(track.get("referee_like_score") or 0.0) >= float(min_referee_score)
                )
                if referee_like:
                    stats[tracklet_id]["referee_like"] += 1
                if group == "referees" or role == "referee":
                    stats[tracklet_id]["strong"] = True
    return {
        tracklet_id
        for tracklet_id, values in stats.items()
        if values["strong"] or values["referee_like"] / max(1, values["total"]) >= 0.5
    }


def is_referee_roster_candidate(row):
    """Return whether a candidate-score row points at a referee roster entry."""
    return str(row.get("player_role") or "").lower() in REFEREE_ROLES


def candidate_reason(row):
    """Explain why a non-authoritative candidate was selected."""
    gate = row.get("assignment_gate") or {}
    if gate.get("pass") and gate.get("reason"):
        return str(gate["reason"])

    player_team = row.get("player_team_id")
    track_team = row.get("tracklet_team_id")
    if player_team is not None and track_team is not None and int(player_team) == int(track_team):
        expected = row.get("player_jersey_number")
        observed = row.get("tracklet_jersey_number")
        if expected is not None and observed is not None and int(expected) == int(observed):
            return "team_jersey_candidate"
        distance = row.get("position_prior_distance")
        if distance is not None and float(distance) <= 18.0:
            return "team_position_candidate"
        return "team_candidate"

    if row.get("tracklet_jersey_number") is not None and row.get("player_jersey_number") is not None:
        if int(row["tracklet_jersey_number"]) == int(row["player_jersey_number"]):
            return "jersey_candidate"

    if gate.get("reason"):
        return f"gate_{gate['reason']}"
    return "low_cost_candidate"
