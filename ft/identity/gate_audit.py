from collections import Counter, defaultdict

from ft.identity.hungarian import jersey_candidate_score


def build_identity_gate_audit(
    summaries,
    candidate_scores,
    assignments,
    roster,
    identity_config=None,
    near_miss_window=0.10,
    final_identity_states=None,
):
    identity_config = identity_config or {}
    min_candidate_score = float(identity_config.get("reliable_jersey_min_candidate_score", 0.20))
    roster_by_id = {str(row["player_id"]): row for row in roster or []}
    scores_by_track = defaultdict(list)
    for row in candidate_scores or []:
        scores_by_track[int(row["track_id"])].append(row)
    rows = []
    for summary in summaries or []:
        track_id = int(summary["track_id"])
        assignment = (assignments or {}).get(track_id, {})
        final_state = (final_identity_states or {}).get(track_id, {})
        scores = sorted(scores_by_track.get(track_id, []), key=lambda row: (float(row["cost"]), str(row["player_id"])))
        selected_player = assignment.get("player_id")
        selected = next((row for row in scores if row.get("player_id") == selected_player), None)
        best = selected or (scores[0] if scores else {})
        second = next((row for row in scores if row is not best), None)
        player = roster_by_id.get(str(best.get("player_id")), {})
        candidate_score = jersey_candidate_score(summary, player.get("jersey_number"), field="raw_jersey_distribution")
        if candidate_score is None:
            candidate_score = jersey_candidate_score(summary, player.get("jersey_number"), field="jersey_distribution")
        gate = best.get("assignment_gate") if isinstance(best.get("assignment_gate"), dict) else {}
        strong_failures = strong_gate_failures(gate, identity_config)
        hungarian_status = str(
            assignment.get("identity_status")
            or assignment.get("evidence", {}).get("status")
            or "unknown"
        )
        status = str(final_state.get("status") or hungarian_status)
        rows.append({
            "tracklet_id": track_id,
            "display_track_id": summary.get("display_track_id", track_id),
            "frames": int(summary.get("num_frames") or 0),
            "assignment_status": status,
            "hungarian_assignment_status": hungarian_status,
            "assigned_player_id": selected_player or "unknown",
            "final_player_ids": final_state.get("player_ids", []),
            "final_assigned_rows": int(final_state.get("assigned_rows", 0)),
            "final_unknown_rows": int(final_state.get("unknown_rows", 0)),
            "post_assignment_reasons": final_state.get("reasons", []),
            "best_player_id": best.get("player_id"),
            "best_player_name": best.get("player_name"),
            "best_cost": float(best["cost"]) if best.get("cost") is not None else None,
            "best_confidence": float(best["confidence"]) if best.get("confidence") is not None else None,
            "cost_margin": (
                float(second["cost"]) - float(best["cost"])
                if second and second.get("cost") is not None and best.get("cost") is not None
                else None
            ),
            "gate_reason": gate.get("reason") or assignment.get("evidence", {}).get("status"),
            "reliable_jersey": bool(gate.get("reliable_jersey", False)),
            "goalkeeper_singleton": bool(gate.get("goalkeeper_singleton", False)),
            "strong_combined": bool(gate.get("strong_combined", False)),
            "strong_gate_failures": strong_failures,
            "tracklet_visual_available": summary.get("visual_embedding") is not None,
            "roster_visual_available": bool(player.get("visual_embedding") or player.get("visual_profile")),
            "visual_similarity": best.get("visual_similarity"),
            "tracklet_position_available": summary.get("mean_pitch_position") is not None,
            "roster_position_available": player.get("position_prior") is not None,
            "position_prior_distance": best.get("position_prior_distance"),
            "team_id": summary.get("team_id"),
            "team_confidence": float(summary.get("mean_team_confidence") or 0.0),
            "best_player_team_id": player.get("team_id"),
            "best_player_jersey_number": player.get("jersey_number"),
            "jersey_candidate_score": candidate_score,
            "jersey_candidate_score_gap": (
                float(min_candidate_score) - float(candidate_score)
                if candidate_score is not None and candidate_score < float(min_candidate_score)
                else 0.0
            ),
            "jersey_number": summary.get("jersey_number"),
            "jersey_confidence": float(summary.get("jersey_confidence") or 0.0),
            "jersey_votes": int(summary.get("jersey_votes") or 0),
            "jersey_winner_margin": float(summary.get("jersey_winner_margin") or 0.0),
            "raw_jersey_distribution": summary.get("raw_jersey_distribution") or [],
            "crop_paths": summary.get("crop_paths") or [],
            "near_miss_reliable_jersey": bool(
                status == "unknown"
                and candidate_score is not None
                and candidate_score < float(min_candidate_score)
                and candidate_score >= max(0.0, float(min_candidate_score) - float(near_miss_window))
            ),
        })
    return {
        "summary": gate_audit_summary(rows, roster),
        "tracklets": rows,
        "near_misses": sorted(
            [row for row in rows if row["near_miss_reliable_jersey"]],
            key=lambda row: (row["jersey_candidate_score_gap"], -row["frames"]),
        ),
    }


def build_final_identity_states(tracks):
    """Aggregate the actual per-frame state after constraints and bridges."""
    grouped = defaultdict(list)
    for frame_tracks in (tracks or {}).get("players", []):
        for raw_id, track in frame_tracks.items():
            tracklet_id = int(
                track.get("identity_tracklet_id")
                or track.get("display_track_id", raw_id)
            )
            grouped[tracklet_id].append(track)

    states = {}
    for tracklet_id, items in grouped.items():
        player_ids = sorted({
            str(item.get("player_id"))
            for item in items
            if item.get("player_id") not in (None, "unknown")
        })
        assigned_rows = sum(
            item.get("player_id") not in (None, "unknown")
            for item in items
        )
        unknown_rows = len(items) - assigned_rows
        if assigned_rows and unknown_rows:
            status = "partially_assigned"
        elif assigned_rows:
            status = "assigned"
        else:
            status = "unknown"
        reasons = sorted({
            str(reason)
            for item in items
            if item.get("player_id") in (None, "unknown")
            for reason in [identity_state_reason(item)]
            if reason
        })
        states[int(tracklet_id)] = {
            "status": status,
            "player_ids": player_ids,
            "assigned_rows": int(assigned_rows),
            "unknown_rows": int(unknown_rows),
            "reasons": reasons,
        }
    return states


def identity_state_reason(track):
    evidence = track.get("identity_evidence")
    if not isinstance(evidence, dict):
        return None
    return evidence.get("reason") or evidence.get("status")


def strong_gate_failures(gate, identity_config=None):
    identity_config = identity_config or {}
    failures = []
    if not gate.get("team_match", False):
        failures.append("team_mismatch")
    if float(gate.get("team_confidence") or 0.0) < float(identity_config.get("strong_evidence_min_team_confidence", 0.75)):
        failures.append("team_confidence")
    if gate.get("visual_similarity") is None:
        failures.append("missing_visual_similarity")
    if int(gate.get("tracklet_frames") or 0) < int(identity_config.get("strong_evidence_min_tracklet_frames", 45)):
        failures.append("tracklet_frames")
    if gate.get("position_prior_distance") is None:
        failures.append("missing_position_prior")
    elif float(gate["position_prior_distance"]) > float(identity_config.get("strong_evidence_max_position_distance", 18.0)):
        failures.append("position_distance")
    return failures


def gate_audit_summary(rows, roster):
    roster_visual = sum(bool(row.get("visual_embedding") or row.get("visual_profile")) for row in roster or [])
    roster_position = sum(row.get("position_prior") is not None for row in roster or [])
    return {
        "tracklets": len(rows),
        "assigned_tracklets": sum(row["assignment_status"] == "assigned" for row in rows),
        "partially_assigned_tracklets": sum(row["assignment_status"] == "partially_assigned" for row in rows),
        "unknown_tracklets": sum(
            row["assignment_status"] not in {"assigned", "partially_assigned"}
            for row in rows
        ),
        "changed_after_hungarian_tracklets": sum(
            row["hungarian_assignment_status"] == "assigned"
            and row["assignment_status"] != "assigned"
            for row in rows
        ),
        "final_assigned_rows": sum(row["final_assigned_rows"] for row in rows),
        "final_unknown_rows": sum(row["final_unknown_rows"] for row in rows),
        "near_miss_tracklets": sum(row["near_miss_reliable_jersey"] for row in rows),
        "gate_reasons": dict(Counter(row["gate_reason"] or "unknown" for row in rows)),
        "strong_gate_failures": dict(Counter(reason for row in rows for reason in row["strong_gate_failures"])),
        "tracklet_visual_available": sum(row["tracklet_visual_available"] for row in rows),
        "roster_players": len(roster or []),
        "roster_visual_available": roster_visual,
        "roster_position_available": roster_position,
        "strong_visual_gate_available": bool(roster_visual > 0),
    }
