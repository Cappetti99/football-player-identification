from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ft.features.visual import mean_embedding
from ft.identity.roster import load_roster, validate_unique_team_jersey


class HungarianPlayerIdentifier:
    """Assign display tracklets to roster players with a transparent cost matrix."""

    def __init__(
        self,
        roster_path=None,
        unknown_threshold=0.55,
        enforce_unique_team_jersey=True,
        reliable_jersey_min_votes=2,
        reliable_jersey_min_confidence=0.5,
        goalkeeper_number_one_prior=True,
        number_one_goalkeeper_bonus=0.08,
        number_one_non_goalkeeper_penalty=0.08,
    ):
        self.roster = load_roster(roster_path)
        self.unknown_threshold = float(unknown_threshold)
        self.enforce_unique_team_jersey = bool(enforce_unique_team_jersey)
        self.reliable_jersey_min_votes = int(reliable_jersey_min_votes)
        self.reliable_jersey_min_confidence = float(reliable_jersey_min_confidence)
        self.goalkeeper_number_one_prior = bool(goalkeeper_number_one_prior)
        self.number_one_goalkeeper_bonus = float(number_one_goalkeeper_bonus)
        self.number_one_non_goalkeeper_penalty = float(number_one_non_goalkeeper_penalty)
        if self.enforce_unique_team_jersey:
            validate_unique_team_jersey(self.roster)

    def summarize(self, rows):
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
            positions = [row.get("position_pitch") for row in items if row.get("position_pitch") is not None]
            visual_values = [row.get("visual_embedding") for row in items if row.get("visual_embedding") is not None]
            summary = {
                "track_id": int(track_id),
                "raw_track_ids": sorted({int(row.get("raw_track_id", row["track_id"])) for row in items}),
                "team_id": mode(team_ids),
                "team_votes": count_mode(team_ids)[1],
                "mean_team_confidence": mean([row.get("team_confidence", 0.0) for row in items if row.get("team_id") is not None]),
                "jersey_number": jersey_number,
                "jersey_distribution": jersey_distribution,
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
            if confidence < self.unknown_threshold:
                assignments[track_id]["confidence"] = confidence
                assignments[track_id]["evidence"].update(
                    {
                        "best_candidate": player["player_id"],
                        "cost": details["cost"],
                        "feature_costs": details["components"],
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
                        "tracklet_jersey_roster_mass": tracklet.get("jersey_roster_mass"),
                        "tracklet_frames": tracklet.get("num_frames"),
                        "mean_crop_quality": tracklet.get("mean_crop_quality"),
                        "mean_pitch_position": tracklet.get("mean_pitch_position"),
                        "cost": details["cost"],
                        "confidence": details["confidence"],
                        "components": details["components"],
                    }
                )
        rows.sort(key=lambda row: (row["track_id"], row["cost"], row["player_id"]))
        return rows

    def cost_details(self, tracklet, player):
        components = {
            "base": 0.25,
            "team": 0.0,
            "jersey": 0.0,
            "team_jersey_constraint": 0.0,
            "goalkeeper_number_one_prior": 0.0,
            "position_prior": 0.0,
            "visual": 0.0,
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
            components["jersey"] = 0.35

        if tracklet.get("mean_pitch_position") is not None and player.get("position_prior") is not None:
            dist = np.linalg.norm(np.asarray(tracklet["mean_pitch_position"]) - np.asarray(player["position_prior"]))
            components["position_prior"] = min(0.25, float(dist) / 120.0)

        visual_cost = visual_distance(tracklet.get("visual_embedding"), player.get("visual_embedding") or player.get("visual_profile"))
        if visual_cost is not None:
            components["visual"] = min(0.30, 0.30 * visual_cost)

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
        }

    def _has_reliable_jersey(self, tracklet):
        return (
            tracklet.get("jersey_number") is not None
            and int(tracklet.get("jersey_votes") or 0) >= self.reliable_jersey_min_votes
            and float(tracklet.get("jersey_confidence") or 0.0) >= self.reliable_jersey_min_confidence
        )


def is_non_player_tracklet(items):
    roles = [str(row.get("role_detection") or "").lower() for row in items]
    if not roles:
        return False
    referee_like = sum(role in {"referee", "referee_candidate"} for role in roles)
    return referee_like / len(roles) >= 0.5


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


def aggregate_jersey_distribution(items):
    scores = defaultdict(float)
    votes = defaultdict(int)
    for row in items:
        distribution = row.get("jersey_distribution") or row.get("jersey_candidates") or []
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


def jersey_candidate_score(tracklet, expected):
    if expected is None:
        return None
    for candidate in tracklet.get("jersey_distribution") or []:
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
