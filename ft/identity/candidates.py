from collections import defaultdict


def build_identity_candidates(
    candidate_scores,
    tracks=None,
    enabled=True,
    min_confidence=0.35,
    max_cost=0.85,
    min_margin=0.0,
):
    """Select one diagnostic identity candidate for each still-unknown track.

    Candidates are deliberately non-authoritative. They expose the best roster
    hypothesis for analysis and annotation review, but never overwrite the real
    player identity assigned by Hungarian plus constraints.
    """
    if not enabled:
        return {}

    assigned_track_ids = assigned_display_track_ids(tracks) if tracks is not None else set()
    grouped = defaultdict(list)
    for row in candidate_scores or []:
        try:
            track_id = int(row["track_id"])
        except Exception:
            continue
        if track_id in assigned_track_ids:
            continue
        grouped[track_id].append(row)

    candidates = {}
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

        candidates[track_id] = {
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
    return candidates


def apply_identity_candidates(tracks, candidates):
    if not candidates:
        return
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            candidate = candidates.get(display_id)
            if not candidate:
                continue
            if track.get("player_id") not in (None, "", "unknown"):
                continue
            track.update(candidate)


def identity_candidate_rows(candidates):
    rows = []
    for track_id, candidate in sorted((candidates or {}).items()):
        row = {"track_id": int(track_id)}
        row.update(candidate)
        rows.append(row)
    return rows


def assigned_display_track_ids(tracks):
    assigned = set()
    for frame_tracks in (tracks or {}).get("players", []):
        for raw_id, track in frame_tracks.items():
            if track.get("player_id") in (None, "", "unknown"):
                continue
            assigned.add(int(track.get("display_track_id", raw_id)))
    return assigned


def candidate_reason(row):
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
