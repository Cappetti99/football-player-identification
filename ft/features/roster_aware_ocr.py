from collections import defaultdict

from ft.identity.roster import goalkeeper_numbers_by_team, roster_numbers_by_team


class RosterAwareOCRFilter:
    """Validate OCR jersey numbers against the known roster per team."""

    def __init__(
        self,
        roster,
        mode="degrade",
        unknown_team_policy="keep",
        confidence_scale=0.60,
        promote_roster_candidate=True,
        min_promoted_candidate_confidence=0.12,
        min_promoted_candidate_votes=1,
        preserve_dropped_evidence=False,
    ):
        self.numbers_by_team = roster_numbers_by_team(roster)
        self.goalkeeper_numbers_by_team = goalkeeper_numbers_by_team(roster)
        self.mode = str(mode)
        self.unknown_team_policy = str(unknown_team_policy)
        self.confidence_scale = float(confidence_scale)
        self.promote_roster_candidate = bool(promote_roster_candidate)
        self.min_promoted_candidate_confidence = float(min_promoted_candidate_confidence)
        self.min_promoted_candidate_votes = int(min_promoted_candidate_votes)
        self.preserve_dropped_evidence = bool(preserve_dropped_evidence)

    def apply(self, assignments, rows):
        """Filter OCR assignments by roster, returning filtered values and diagnostics."""
        if not assignments or not self.numbers_by_team:
            return assignments, {
                "enabled": bool(self.numbers_by_team),
                "mode": self.mode,
                "kept": len(assignments or {}),
                "dropped": {},
                "degraded": {},
                "reason": "missing_assignments_or_roster",
            }

        teams_by_tracklet = summarize_team_by_tracklet(rows)
        roles_by_tracklet = summarize_role_by_tracklet(rows)
        filtered = {}
        dropped = {}
        degraded = {}
        goalkeeper_rejections = {}
        for track_id, assignment in sorted(assignments.items(), key=lambda item: assignment_key_string(item[0])):
            if assignment.get("raw_jersey_distribution") is None:
                assignment = dict(assignment)
                assignment["raw_jersey_distribution"] = assignment.get("candidates")
            display_id = assignment_display_id(track_id, assignment)
            scoped_rows = assignment_rows(rows, display_id, assignment)
            team_id = summarize_team_by_tracklet(scoped_rows).get(display_id, teams_by_tracklet.get(display_id))
            role = summarize_role_by_tracklet(scoped_rows).get(display_id, roles_by_tracklet.get(display_id))
            jersey = int(assignment["jersey_number"])
            if team_id is None:
                if self.unknown_team_policy == "drop":
                    dropped[assignment_key_string(track_id)] = rejection_payload(assignment, team_id, "unknown_team")
                    continue
                filtered[track_id] = assignment
                continue

            valid_numbers = self.numbers_by_team.get(int(team_id), set())
            if role in {"goalkeeper", "keeper", "gk"}:
                valid_goalkeeper_numbers = self.goalkeeper_numbers_by_team.get(int(team_id), set())
                if valid_goalkeeper_numbers:
                    assignment = apply_roster_distribution(assignment, valid_goalkeeper_numbers)
                    jersey = int(assignment["jersey_number"])
                    if jersey not in valid_goalkeeper_numbers:
                        promoted = self._promote_candidate(assignment, team_id, valid_goalkeeper_numbers)
                        if promoted is not None:
                            filtered[track_id] = promoted
                            degraded[assignment_key_string(track_id)] = promoted["roster_filter"]
                            continue
                        rejection = rejection_payload(
                            assignment,
                            team_id,
                            "goalkeeper_number_not_in_goalkeeper_roster",
                            valid_goalkeeper_numbers,
                        )
                        goalkeeper_rejections[assignment_key_string(track_id)] = rejection
                        dropped[assignment_key_string(track_id)] = rejection
                        if self.preserve_dropped_evidence:
                            filtered[track_id] = preserved_evidence_assignment(assignment, rejection)
                        continue

            assignment = apply_roster_distribution(assignment, valid_numbers)
            jersey = int(assignment["jersey_number"])
            if jersey in valid_numbers:
                filtered[track_id] = assignment
                continue

            promoted = self._promote_candidate(assignment, team_id, valid_numbers)
            if promoted is not None:
                filtered[track_id] = promoted
                degraded[assignment_key_string(track_id)] = promoted["roster_filter"]
                continue

            if self.mode == "degrade":
                updated = dict(assignment)
                updated["confidence"] = float(updated.get("confidence", 0.0)) * self.confidence_scale
                updated["roster_filter"] = {
                    "status": "degraded",
                    "team_id": int(team_id),
                    "jersey_number": jersey,
                    "valid_numbers": sorted(valid_numbers),
                }
                filtered[track_id] = updated
                degraded[assignment_key_string(track_id)] = updated["roster_filter"]
            else:
                rejection = rejection_payload(
                    assignment,
                    team_id,
                    "number_not_in_team_roster",
                    valid_numbers,
                )
                dropped[assignment_key_string(track_id)] = rejection
                if self.preserve_dropped_evidence:
                    filtered[track_id] = preserved_evidence_assignment(assignment, rejection)

        return filtered, {
            "enabled": True,
            "mode": self.mode,
            "unknown_team_policy": self.unknown_team_policy,
            "preserve_dropped_evidence": self.preserve_dropped_evidence,
            "teams": {str(team): sorted(numbers) for team, numbers in self.numbers_by_team.items()},
            "input_assignments": len(assignments),
            "kept": len(filtered),
            "dropped": dropped,
            "degraded": degraded,
            "goalkeeper_rejections": goalkeeper_rejections,
        }

    def _promote_candidate(self, assignment, team_id, valid_numbers):
        """Promote a lower-ranked OCR candidate only when it is roster-valid."""
        if not self.promote_roster_candidate:
            return None
        candidates = assignment.get("candidates") or []
        for candidate in candidates:
            number = int(candidate.get("jersey_number"))
            if number not in valid_numbers:
                continue
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            votes = int(candidate.get("votes", 0) or 0)
            if confidence < self.min_promoted_candidate_confidence:
                continue
            if votes < self.min_promoted_candidate_votes:
                continue
            updated = dict(assignment)
            updated["jersey_number"] = number
            updated["confidence"] = confidence
            updated["votes"] = votes
            updated["roster_filter"] = {
                "status": "promoted_alternative",
                "team_id": int(team_id),
                "original_jersey_number": int(assignment["jersey_number"]),
                "promoted_jersey_number": number,
                "promoted_confidence": confidence,
                "valid_numbers": sorted(valid_numbers),
            }
            return updated
        return None


def summarize_team_by_tracklet(rows):
    """Return the majority team id for each display tracklet."""
    votes = defaultdict(lambda: defaultdict(int))
    for row in rows:
        display_id = int(row.get("display_track_id", row["track_id"]))
        team = row.get("team_id")
        if team in (None, "", "None"):
            continue
        votes[display_id][int(team)] += 1
    teams = {}
    for track_id, counts in votes.items():
        team, _ = max(counts.items(), key=lambda item: item[1])
        teams[track_id] = int(team)
    return teams


def summarize_role_by_tracklet(rows):
    """Return the majority role label for each display tracklet."""
    votes = defaultdict(lambda: defaultdict(int))
    for row in rows:
        display_id = int(row.get("display_track_id", row["track_id"]))
        role = str(row.get("role_detection") or "").strip().lower()
        if not role:
            continue
        votes[display_id][role] += 1
    roles = {}
    for track_id, counts in votes.items():
        role, _ = max(counts.items(), key=lambda item: item[1])
        roles[track_id] = role
    return roles


def assignment_display_id(track_id, assignment):
    """Normalize a scalar or segmented OCR key into its display track id."""
    if assignment.get("display_track_id") is not None:
        return int(assignment["display_track_id"])
    if isinstance(track_id, tuple):
        return int(track_id[0])
    return int(track_id)


def assignment_rows(rows, display_id, assignment):
    """Select source rows matching a display id and optional OCR segment window."""
    start = assignment.get("segment_start_frame")
    end = assignment.get("segment_end_frame")
    scoped = []
    for row in rows:
        row_display_id = int(row.get("display_track_id", row["track_id"]))
        if row_display_id != int(display_id):
            continue
        frame = int(row.get("frame", 0) or 0)
        if start is not None and frame < int(start):
            continue
        if end is not None and frame > int(end):
            continue
        scoped.append(row)
    return scoped


def assignment_key_string(track_id):
    """Serialize scalar and segmented track ids for diagnostics."""
    if isinstance(track_id, tuple):
        return f"{int(track_id[0])}:{int(track_id[1])}"
    return str(int(track_id))


def rejection_payload(assignment, team_id, reason, valid_numbers=None):
    """Build the diagnostic payload for a roster-filter rejection."""
    payload = {
        "status": "dropped",
        "reason": reason,
        "team_id": int(team_id) if team_id is not None else None,
        "jersey_number": int(assignment["jersey_number"]),
        "confidence": float(assignment.get("confidence", 0.0)),
    }
    if valid_numbers is not None:
        payload["valid_numbers"] = sorted(valid_numbers)
    return payload


def preserved_evidence_assignment(assignment, rejection):
    """Keep rejected OCR as candidate evidence without exposing a jersey number.

    This is intentionally not a degraded assignment: downstream identity code
    can inspect the raw distribution for candidate diagnostics, while the
    authoritative jersey field remains empty.
    """
    updated = dict(assignment)
    if updated.get("raw_jersey_distribution") is None:
        updated["raw_jersey_distribution"] = updated.get("candidates")
    updated["original_jersey_number"] = updated.get("jersey_number")
    updated["jersey_number"] = None
    updated["confidence"] = 0.0
    updated["votes"] = 0
    updated["roster_filter"] = {
        **dict(rejection),
        "status": "dropped_preserved_evidence",
        "original_status": rejection.get("status"),
    }
    return updated


def apply_roster_distribution(assignment, valid_numbers):
    """Attach the roster-valid slice of an OCR candidate distribution."""
    candidates = assignment.get("candidates") or []
    valid_candidates = [
        candidate
        for candidate in candidates
        if int(candidate.get("jersey_number")) in valid_numbers
    ]
    if not valid_candidates:
        return assignment

    total_confidence = sum(float(candidate.get("confidence", 0.0) or 0.0) for candidate in valid_candidates)
    if total_confidence <= 0:
        return assignment

    updated = dict(assignment)
    updated["jersey_distribution"] = [
        {
            "jersey_number": int(candidate["jersey_number"]),
            "confidence": float(candidate.get("confidence", 0.0) or 0.0),
            "votes": int(candidate.get("votes", 0) or 0),
        }
        for candidate in valid_candidates
    ]
    updated["jersey_roster_mass"] = float(total_confidence)
    return updated
