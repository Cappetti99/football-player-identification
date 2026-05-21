from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ft.features.visual import mean_embedding
from ft.identity.roster import load_roster, validate_unique_team_jersey


class HungarianPlayerIdentifier:
    """Assign display tracklets to roster players with a transparent cost matrix.

    The solver gives the globally best one-to-one assignment, but it is not
    trusted blindly: a separate assignment gate below blocks weak matches and
    leaves them as unknown.
    """

    def __init__(
        self,
        roster_path=None,
        unknown_threshold=0.55,
        enforce_unique_team_jersey=True,
        reliable_jersey_min_votes=5,
        reliable_jersey_min_confidence=0.20,
        reliable_jersey_min_head_confidence=0.55,
        reliable_jersey_min_winner_margin=0.10,
        goalkeeper_number_one_prior=True,
        number_one_goalkeeper_bonus=0.08,
        number_one_non_goalkeeper_penalty=0.08,
        position_prior_max_cost=0.08,
        position_prior_tiebreak_only=True,
        require_assignment_evidence=True,
        reliable_jersey_min_candidate_score=0.45,
        strong_evidence_min_team_confidence=0.75,
        strong_evidence_min_visual_similarity=0.82,
        strong_evidence_min_tracklet_frames=45,
        strong_evidence_max_position_distance=18.0,
        goalkeeper_singleton_gate=True,
        goalkeeper_singleton_min_team_confidence=0.75,
        goalkeeper_singleton_min_tracklet_frames=30,
    ):
        self.roster = load_roster(roster_path)
        self.unknown_threshold = float(unknown_threshold)
        self.enforce_unique_team_jersey = bool(enforce_unique_team_jersey)
        self.reliable_jersey_min_votes = int(reliable_jersey_min_votes)
        self.reliable_jersey_min_confidence = float(reliable_jersey_min_confidence)
        self.reliable_jersey_min_head_confidence = float(reliable_jersey_min_head_confidence)
        self.reliable_jersey_min_winner_margin = float(reliable_jersey_min_winner_margin)
        self.goalkeeper_number_one_prior = bool(goalkeeper_number_one_prior)
        self.number_one_goalkeeper_bonus = float(number_one_goalkeeper_bonus)
        self.number_one_non_goalkeeper_penalty = float(number_one_non_goalkeeper_penalty)
        self.position_prior_max_cost = float(position_prior_max_cost)
        self.position_prior_tiebreak_only = bool(position_prior_tiebreak_only)
        self.require_assignment_evidence = bool(require_assignment_evidence)
        self.reliable_jersey_min_candidate_score = float(reliable_jersey_min_candidate_score)
        self.strong_evidence_min_team_confidence = float(strong_evidence_min_team_confidence)
        self.strong_evidence_min_visual_similarity = float(strong_evidence_min_visual_similarity)
        self.strong_evidence_min_tracklet_frames = int(strong_evidence_min_tracklet_frames)
        self.strong_evidence_max_position_distance = float(strong_evidence_max_position_distance)
        self.goalkeeper_singleton_gate = bool(goalkeeper_singleton_gate)
        self.goalkeeper_singleton_min_team_confidence = float(goalkeeper_singleton_min_team_confidence)
        self.goalkeeper_singleton_min_tracklet_frames = int(goalkeeper_singleton_min_tracklet_frames)
        if self.enforce_unique_team_jersey:
            validate_unique_team_jersey(self.roster)

    def summarize(self, rows):
        """Collapse per-frame rows into the evidence unit used by Hungarian."""
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(row.get("display_track_id", row["track_id"]))].append(row)
        summaries = []
        for track_id, items in sorted(grouped.items()):
            if is_non_player_tracklet(items):
                continue
            frames = sorted({int(row["frame"]) for row in items})
            team_ids = [row.get("team_id") for row in items if row.get("team_id") is not None]
            jerseys = [row.get("jersey_number") for row in items if row.get("jersey_number") not in (None, -1)]
            jersey_number = mode(jerseys)
            jersey_distribution = aggregate_jersey_distribution(items)
            jersey_raw_candidates = aggregate_jersey_distribution(items, fields=("jersey_candidates",))
            positions = [row.get("position_pitch") for row in items if row.get("position_pitch") is not None]
            visual_values = [row.get("visual_embedding") for row in items if row.get("visual_embedding") is not None]
            roles = [row.get("role_detection") for row in items if row.get("role_detection")]
            semantic_groups = [row.get("semantic_group_id") for row in items if row.get("semantic_group_id") is not None]
            goalkeeper_matches = [bool(row.get("goalkeeper_palette_match", False)) for row in items]
            goalkeeper_scores = [row.get("goalkeeper_like_score", 0.0) for row in items if row.get("goalkeeper_like_score") is not None]
            goalkeeper_teams = [row.get("goalkeeper_like_team") for row in items if row.get("goalkeeper_like_team") is not None]
            summary = {
                "track_id": int(track_id),
                "raw_track_ids": sorted({int(row.get("raw_track_id", row["track_id"])) for row in items}),
                "role_detection": mode(roles),
                "semantic_group_id": mode(semantic_groups),
                "team_id": mode(team_ids),
                "team_votes": count_mode(team_ids)[1],
                "mean_team_confidence": mean([row.get("team_confidence", 0.0) for row in items if row.get("team_id") is not None]),
                "goalkeeper_palette_match": mean(goalkeeper_matches) >= 0.5 if goalkeeper_matches else False,
                "goalkeeper_like_score": mean(goalkeeper_scores),
                "goalkeeper_like_team": mode(goalkeeper_teams),
                "jersey_number": jersey_number,
                "jersey_distribution": jersey_distribution,
                "jersey_raw_candidates": jersey_raw_candidates,
                "jersey_roster_mass": mean(
                    row.get("jersey_roster_mass", 0.0)
                    for row in items
                    if row.get("jersey_roster_mass") is not None
                ),
                "jersey_votes": max_int(
                    row.get("jersey_votes", 0)
                    for row in items
                    if row.get("jersey_number") == jersey_number
                ),
                "jersey_confidence": mean(
                    row.get("jersey_confidence", 0.0)
                    for row in items
                    if row.get("jersey_number") == jersey_number
                ),
                "jersey_head_confidence": mean(
                    row.get("jersey_head_confidence", 0.0)
                    for row in items
                    if row.get("jersey_number") == jersey_number
                ),
                "jersey_winner_margin": mean(
                    row.get("jersey_winner_margin", 0.0)
                    for row in items
                    if row.get("jersey_number") == jersey_number
                ),
                "num_frames": len(items),
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "frame_span": frames[-1] - frames[0] + 1,
                "mean_pitch_position": mean_position(positions),
                "mean_crop_quality": mean([row.get("crop_quality", 0.0) for row in items]),
                "visual_embedding": mean_embedding(visual_values),
                "crop_paths": [row["crop_path"] for row in items if row.get("crop_path")],
            }
            summaries.append(summary)
        return summaries

    def assign(self, summaries):
        if not summaries:
            return {}, []
        if not self.roster:
            return unknown_assignments(summaries, "missing_roster"), []
        scores = self.candidate_scores(summaries)
        cost_matrix = np.asarray(
            [[self.cost_details(tracklet, player)["cost"] for player in self.roster] for tracklet in summaries],
            dtype=float,
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = unknown_assignments(summaries, "below_threshold")
        for row, col in zip(row_ind, col_ind):
            tracklet = summaries[row]
            player = self.roster[col]
            details = self.cost_details(tracklet, player)
            confidence = details["confidence"]
            track_id = int(tracklet["track_id"])
            assignment_gate = self.assignment_gate(tracklet, player, details)
            if confidence < self.unknown_threshold:
                # Keep the best rejected candidate in diagnostics. It is useful
                # when tuning thresholds, but it must not become a real identity.
                assignments[track_id]["confidence"] = confidence
                assignments[track_id]["evidence"].update(
                    {
                        "best_candidate": player["player_id"],
                        "cost": details["cost"],
                        "feature_costs": details["components"],
                        "assignment_gate": assignment_gate,
                    }
                )
                continue
            if not assignment_gate["pass"]:
                # Low cost alone is not enough. Without reliable jersey evidence
                # or strong combined cues, the conservative outcome is unknown.
                assignments[track_id]["confidence"] = confidence
                assignments[track_id]["evidence"].update(
                    {
                        "best_candidate": player["player_id"],
                        "cost": details["cost"],
                        "feature_costs": details["components"],
                        "assignment_gate": assignment_gate,
                        "status": "insufficient_assignment_evidence",
                    }
                )
                continue
            assignments[track_id] = {
                "player_id": player["player_id"],
                "player_name": player.get("name", player["player_id"]),
                "team_id": tracklet.get("team_id"),
                "jersey_number": player.get("jersey_number") or tracklet.get("jersey_number"),
                "confidence": confidence,
                "evidence": {
                    "status": "assigned",
                    "cost": details["cost"],
                    "feature_costs": details["components"],
                    "jersey_observed": tracklet.get("jersey_number"),
                    "team_match": tracklet.get("team_id") == player.get("team_id"),
                    "assignment_gate": assignment_gate,
                },
            }
        return assignments, scores

    def candidate_scores(self, summaries):
        rows = []
        for tracklet in summaries:
            for player in self.roster:
                details = self.cost_details(tracklet, player)
                rows.append(
                    {
                        "track_id": tracklet["track_id"],
                        "player_id": player["player_id"],
                        "player_name": player.get("name", player["player_id"]),
                        "player_team_id": player.get("team_id"),
                        "player_jersey_number": player.get("jersey_number"),
                        "player_role": player.get("role"),
                        "tracklet_team_id": tracklet.get("team_id"),
                        "tracklet_team_confidence": tracklet.get("mean_team_confidence"),
                        "tracklet_jersey_number": tracklet.get("jersey_number"),
                        "tracklet_jersey_confidence": tracklet.get("jersey_confidence"),
                        "tracklet_jersey_votes": tracklet.get("jersey_votes"),
                        "tracklet_jersey_distribution": tracklet.get("jersey_distribution"),
                        "tracklet_jersey_raw_candidates": tracklet.get("jersey_raw_candidates"),
                        "tracklet_jersey_roster_mass": tracklet.get("jersey_roster_mass"),
                        "tracklet_frames": tracklet.get("num_frames"),
                        "mean_crop_quality": tracklet.get("mean_crop_quality"),
                        "mean_pitch_position": tracklet.get("mean_pitch_position"),
                        "position_prior_distance": details["position_prior_distance"],
                        "visual_similarity": details["visual_similarity"],
                        "assignment_gate": self.assignment_gate(tracklet, player, details),
                        "cost": details["cost"],
                        "confidence": details["confidence"],
                        "components": details["components"],
                    }
                )
        rows.sort(key=lambda row: (row["track_id"], row["cost"], row["player_id"]))
        return rows

    def cost_details(self, tracklet, player):
        """Return total cost and each feature contribution for one candidate."""
        components = {
            "base": 0.25,
            "team": 0.0,
            "jersey": 0.0,
            "team_jersey_constraint": 0.0,
            "goalkeeper_number_one_prior": 0.0,
            "position_prior": 0.0,
            "visual": 0.0,
            "goalkeeper_role": 0.0,
            "tracklet_length": 0.0,
            "crop_quality": 0.0,
        }
        if tracklet.get("team_id") is not None and player.get("team_id") is not None:
            conf = clamp(tracklet.get("mean_team_confidence", 0.0), 0.0, 1.0)
            if int(tracklet["team_id"]) != int(player["team_id"]):
                components["team"] = 0.6 * max(0.25, conf)

        observed = tracklet.get("jersey_number")
        expected = player.get("jersey_number")
        jersey_score = jersey_candidate_score(tracklet, expected)
        if expected is not None and jersey_score is not None:
            # Candidate distributions let a lower-ranked but roster-valid OCR
            # number still influence the assignment.
            components["jersey"] = -0.45 * jersey_score
        elif observed is not None and expected is not None:
            reliability = jersey_reliability(tracklet)
            components["jersey"] = -0.40 * reliability if int(observed) == int(expected) else 0.55 * max(0.25, reliability)
            if self.enforce_unique_team_jersey and self._has_reliable_jersey(tracklet):
                same_known_team = (
                    tracklet.get("team_id") is not None
                    and player.get("team_id") is not None
                    and int(tracklet["team_id"]) == int(player["team_id"])
                )
                if same_known_team and int(observed) != int(expected):
                    components["team_jersey_constraint"] = 0.90
                elif same_known_team and int(observed) == int(expected):
                    components["team_jersey_constraint"] = -0.20 * reliability
            if self.goalkeeper_number_one_prior and int(observed) == 1:
                role = str(player.get("role") or "").lower()
                if role in {"goalkeeper", "keeper", "gk"}:
                    components["goalkeeper_number_one_prior"] = -self.number_one_goalkeeper_bonus * reliability
                elif role:
                    components["goalkeeper_number_one_prior"] = self.number_one_non_goalkeeper_penalty * reliability
        elif expected is not None:
            # Missing jersey is a weak penalty, not a blocker. Many broadcast
            # crops never expose a readable back number.
            components["jersey"] = 0.35

        position_prior_distance = None
        if tracklet.get("mean_pitch_position") is not None and player.get("position_prior") is not None:
            dist = float(
                np.linalg.norm(np.asarray(tracklet["mean_pitch_position"]) - np.asarray(player["position_prior"]))
            )
            position_prior_distance = dist
            if self.position_prior_tiebreak_only:
                scaled = dist / 120.0 * self.position_prior_max_cost
                components["position_prior"] = min(self.position_prior_max_cost, scaled)
            else:
                components["position_prior"] = min(max(0.25, self.position_prior_max_cost), dist / 120.0)

        visual_cost = visual_distance(tracklet.get("visual_embedding"), player.get("visual_embedding") or player.get("visual_profile"))
        visual_similarity = None
        if visual_cost is not None:
            visual_similarity = 1.0 - 2.0 * visual_cost
            components["visual"] = min(0.30, 0.30 * visual_cost)

        if has_goalkeeper_tracklet_evidence(tracklet):
            if is_goalkeeper_player(player) and same_team(tracklet, player):
                components["goalkeeper_role"] = -0.10
            elif same_team(tracklet, player):
                components["goalkeeper_role"] = 0.12

        if tracklet.get("num_frames", 0) < 10:
            components["tracklet_length"] = 0.15
        components["crop_quality"] = -0.15 * clamp(tracklet.get("mean_crop_quality", 0.0), 0.0, 1.0)

        raw_cost = float(sum(components.values()))
        cost = clamp(raw_cost, 0.0, 1.0)
        return {
            "cost": cost,
            "raw_cost": raw_cost,
            "confidence": clamp(1.0 - cost, 0.0, 1.0),
            "components": {key: float(value) for key, value in components.items()},
            "position_prior_distance": position_prior_distance,
            "visual_similarity": visual_similarity,
        }

    def _has_reliable_jersey(self, tracklet):
        if tracklet.get("jersey_number") is None:
            return False
        if int(tracklet.get("jersey_votes") or 0) < self.reliable_jersey_min_votes:
            return False
        if float(tracklet.get("jersey_confidence") or 0.0) < self.reliable_jersey_min_confidence:
            return False
        head_confidence = tracklet.get("jersey_head_confidence")
        if head_confidence is not None and float(head_confidence or 0.0) < self.reliable_jersey_min_head_confidence:
            return False
        winner_margin = tracklet.get("jersey_winner_margin")
        if winner_margin is not None and float(winner_margin or 0.0) < self.reliable_jersey_min_winner_margin:
            return False
        return True

    def assignment_gate(self, tracklet, player, details):
        """Decide whether a low-cost match has enough evidence to be trusted."""
        if not self.require_assignment_evidence:
            return {"pass": True, "reason": "gate_disabled"}

        reliable_jersey = self._has_reliable_jersey_match(tracklet, player)
        goalkeeper_singleton = self._has_goalkeeper_singleton_match(tracklet, player)
        team_match = same_team(tracklet, player)
        team_confidence = float(tracklet.get("mean_team_confidence", 0.0) or 0.0)
        visual_similarity = details.get("visual_similarity")
        position_distance = details.get("position_prior_distance")
        tracklet_frames = int(tracklet.get("num_frames") or 0)
        strong_combined = (
            team_match
            and team_confidence >= self.strong_evidence_min_team_confidence
            and visual_similarity is not None
            and visual_similarity >= self.strong_evidence_min_visual_similarity
            and tracklet_frames >= self.strong_evidence_min_tracklet_frames
            and position_distance is not None
            and position_distance <= self.strong_evidence_max_position_distance
        )

        if reliable_jersey:
            reason = "reliable_jersey"
        elif goalkeeper_singleton:
            reason = "goalkeeper_roster_singleton"
        elif strong_combined:
            reason = "strong_team_visual_trajectory"
        else:
            reason = "insufficient_assignment_evidence"
        return {
            "pass": bool(reliable_jersey or goalkeeper_singleton or strong_combined),
            "reason": reason,
            "reliable_jersey": bool(reliable_jersey),
            "goalkeeper_singleton": bool(goalkeeper_singleton),
            "strong_combined": bool(strong_combined),
            "team_match": bool(team_match),
            "team_confidence": float(team_confidence),
            "visual_similarity": visual_similarity,
            "position_prior_distance": position_distance,
            "tracklet_frames": int(tracklet_frames),
        }

    def _has_reliable_jersey_match(self, tracklet, player):
        expected = player.get("jersey_number")
        if expected is None:
            return False
        raw_candidates = tracklet.get("jersey_raw_candidates") or []
        candidate_score = jersey_candidate_score(tracklet, expected, field="jersey_raw_candidates")
        if raw_candidates:
            # Roster filtering can promote a valid alternative after dropping
            # impossible numbers. The gate should still look at the raw OCR
            # strength, otherwise weak second-place candidates become identities.
            return candidate_score is not None and candidate_score >= self.reliable_jersey_min_candidate_score
        candidate_score = jersey_candidate_score(tracklet, expected)
        if candidate_score is not None and candidate_score >= self.reliable_jersey_min_candidate_score:
            return True
        observed = tracklet.get("jersey_number")
        return observed is not None and int(observed) == int(expected) and self._has_reliable_jersey(tracklet)

    def _has_goalkeeper_singleton_match(self, tracklet, player):
        if not self.goalkeeper_singleton_gate:
            return False
        if not same_team(tracklet, player):
            return False
        if not is_goalkeeper_player(player):
            return False
        if not has_goalkeeper_tracklet_evidence(tracklet):
            return False
        team_id = player.get("team_id")
        if team_id is None:
            return False
        if int(tracklet.get("num_frames") or 0) < self.goalkeeper_singleton_min_tracklet_frames:
            return False
        team_confidence = float(tracklet.get("mean_team_confidence") or 0.0)
        if team_confidence < self.goalkeeper_singleton_min_team_confidence:
            return False
        goalkeeper_like_team = tracklet.get("goalkeeper_like_team")
        if goalkeeper_like_team is not None and int(goalkeeper_like_team) != int(team_id):
            return False
        goalkeepers = [
            candidate
            for candidate in self.roster
            if is_goalkeeper_player(candidate)
            and candidate.get("team_id") is not None
            and int(candidate["team_id"]) == int(team_id)
        ]
        return len(goalkeepers) == 1 and goalkeepers[0].get("player_id") == player.get("player_id")


def is_non_player_tracklet(items):
    roles = [str(row.get("role_detection") or "").lower() for row in items]
    if not roles:
        return False
    referee_like = sum(role in {"referee", "referee_candidate"} for role in roles)
    return referee_like / len(roles) >= 0.5


def same_team(tracklet, player):
    return (
        tracklet.get("team_id") is not None
        and player.get("team_id") is not None
        and int(tracklet["team_id"]) == int(player["team_id"])
    )


def is_goalkeeper_player(player):
    role = str((player or {}).get("role") or "").lower()
    return role in {"goalkeeper", "keeper", "gk"}


def has_goalkeeper_tracklet_evidence(tracklet):
    role = str(tracklet.get("role_detection") or "").lower()
    if role in {"goalkeeper", "keeper", "gk"}:
        return True
    if tracklet.get("semantic_group_id") in {3, 4}:
        return True
    if bool(tracklet.get("goalkeeper_palette_match", False)):
        return True
    return float(tracklet.get("goalkeeper_like_score") or 0.0) >= 0.20


def unknown_assignments(summaries, status):
    return {
        int(summary["track_id"]): {
            "player_id": "unknown",
            "player_name": "unknown",
            "team_id": summary.get("team_id"),
            "jersey_number": summary.get("jersey_number"),
            "confidence": 0.0,
            "evidence": {"status": status, "tracklet_frames": summary.get("num_frames", 0)},
        }
        for summary in summaries
    }


def apply_assignments(tracks, assignments):
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            assignment = assignments.get(display_id)
            if not assignment:
                continue
            track["player_id"] = assignment["player_id"]
            track["player_name"] = assignment["player_name"]
            track["jersey_number"] = assignment["jersey_number"]
            track["identity_confidence"] = assignment["confidence"]
            track["identity_evidence"] = assignment["evidence"]


def mode(values):
    value, _ = count_mode(values)
    return value


def count_mode(values):
    if not values:
        return None, 0
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    value, count = max(counts.items(), key=lambda item: item[1])
    return value, int(count)


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else 0.0


def max_int(values):
    values = [int(value or 0) for value in values]
    return max(values) if values else 0


def mean_position(values):
    if not values:
        return None
    return np.asarray(values, dtype=float).mean(axis=0).tolist()


def aggregate_jersey_distribution(items, fields=("jersey_distribution", "jersey_candidates")):
    """Merge OCR candidate distributions across the frames of one tracklet."""
    scores = defaultdict(float)
    votes = defaultdict(int)
    for row in items:
        distribution = []
        for field in fields:
            distribution = row.get(field) or []
            if distribution:
                break
        if isinstance(distribution, str):
            try:
                import json

                distribution = json.loads(distribution)
            except Exception:
                distribution = []
        for candidate in distribution:
            try:
                number = int(candidate["jersey_number"])
            except Exception:
                continue
            scores[number] += float(candidate.get("confidence", 0.0) or 0.0)
            votes[number] += int(candidate.get("votes", 0) or 0)
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


def jersey_candidate_score(tracklet, expected, field="jersey_distribution"):
    if expected is None:
        return None
    for candidate in tracklet.get(field) or []:
        if int(candidate.get("jersey_number")) == int(expected):
            confidence = clamp(candidate.get("confidence", 0.0), 0.0, 1.0)
            votes = int(candidate.get("votes", 0) or 0)
            return confidence * min(1.0, max(1, votes) / 3.0)
    return None


def jersey_reliability(tracklet):
    conf = clamp(tracklet.get("jersey_confidence", 0.0), 0.0, 1.0)
    votes = int(tracklet.get("jersey_votes") or 0)
    if votes <= 0:
        return 0.2
    return clamp(conf * min(1.0, votes / 3.0), 0.15, 1.0)


def visual_distance(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return None
    cosine = float(np.dot(a, b) / denom)
    return clamp((1.0 - cosine) / 2.0, 0.0, 1.0)


def clamp(value, low, high):
    return max(low, min(high, float(value)))
