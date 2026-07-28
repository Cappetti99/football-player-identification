import json
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
        narrow_region_jersey_score_penalty=0.15,
        # strong_evidence_* / the "strong_combined" gate path are currently
        # inert on every video processed so far: they require a non-None
        # visual_similarity, which requires player.visual_embedding in the
        # roster, and no roster built so far provides one (see handoff.md,
        # identity_gate_report.py findings on Usa-Bel: visual_similarity is
        # None on 23/23 tracklets, no exceptions). Not dead code to remove --
        # this activates automatically once a roster ships visual_embedding
        # per player -- but do not assume it currently affects any decision.
        strong_evidence_min_team_confidence=0.75,
        strong_evidence_min_visual_similarity=0.82,
        strong_evidence_min_tracklet_frames=45,
        strong_evidence_max_position_distance=18.0,
        goalkeeper_singleton_gate=True,
        goalkeeper_singleton_min_team_confidence=0.75,
        goalkeeper_singleton_min_tracklet_frames=30,
        number_region_cost_enabled=False,
        number_region_bonus_weight=0.20,
        number_region_mismatch_penalty=0.20,
        number_region_min_votes=2,
        number_region_min_mean_confidence=0.20,
        number_region_min_consecutive_support=1,
        assignment_scope="global",
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
        # Raises the reliable-jersey bar for tracklets whose OCR decision only
        # reached vote quorum through narrow ROI regions (torso/number_band),
        # not full_body alone -- validated at ~20% accuracy vs ~75-84% for
        # full_body-sufficient decisions on two independent GSR samples.
        self.narrow_region_jersey_score_penalty = float(narrow_region_jersey_score_penalty)
        self.strong_evidence_min_team_confidence = float(strong_evidence_min_team_confidence)
        self.strong_evidence_min_visual_similarity = float(strong_evidence_min_visual_similarity)
        self.strong_evidence_min_tracklet_frames = int(strong_evidence_min_tracklet_frames)
        self.strong_evidence_max_position_distance = float(strong_evidence_max_position_distance)
        self.goalkeeper_singleton_gate = bool(goalkeeper_singleton_gate)
        self.goalkeeper_singleton_min_team_confidence = float(goalkeeper_singleton_min_team_confidence)
        self.goalkeeper_singleton_min_tracklet_frames = int(goalkeeper_singleton_min_tracklet_frames)
        self.number_region_cost_enabled = bool(number_region_cost_enabled)
        self.number_region_bonus_weight = float(number_region_bonus_weight)
        self.number_region_mismatch_penalty = float(number_region_mismatch_penalty)
        self.number_region_min_votes = int(number_region_min_votes)
        self.number_region_min_mean_confidence = float(number_region_min_mean_confidence)
        self.number_region_min_consecutive_support = int(number_region_min_consecutive_support)
        self.assignment_scope = str(assignment_scope or "global").lower()
        if self.enforce_unique_team_jersey:
            validate_unique_team_jersey(self.roster)

    def summarize(self, rows, number_region_evidence=None):
        """Collapse per-frame rows into the evidence unit used by Hungarian."""
        number_regions = number_region_evidence_by_display(number_region_evidence)
        grouped = defaultdict(list)
        for row in rows:
            grouped[self.summary_group_key(row)].append(row)
        summaries = []
        for track_id, items in sorted(grouped.items()):
            if is_non_player_tracklet(items):
                continue
            frames = sorted({int(row["frame"]) for row in items})
            team_ids = [row.get("team_id") for row in items if row.get("team_id") is not None]
            jerseys = [row.get("jersey_number") for row in items if row.get("jersey_number") not in (None, -1)]
            jersey_number = mode(jerseys)
            raw_jersey_distribution = aggregate_jersey_distribution(
                items,
                fields=("raw_jersey_distribution", "jersey_candidates"),
            )
            jersey_distribution = aggregate_jersey_distribution(items, fields=("jersey_distribution",))
            jersey_raw_candidates = raw_jersey_distribution
            positions = [row.get("position_pitch") for row in items if row.get("position_pitch") is not None]
            visual_values = [row.get("visual_embedding") for row in items if row.get("visual_embedding") is not None]
            roles = [row.get("role_detection") for row in items if row.get("role_detection")]
            preserved_dropped_jersey_evidence = any(
                is_preserved_dropped_jersey_evidence(row.get("jersey_roster_filter"))
                for row in items
            )
            semantic_groups = [row.get("semantic_group_id") for row in items if row.get("semantic_group_id") is not None]
            goalkeeper_matches = [bool(row.get("goalkeeper_palette_match", False)) for row in items]
            goalkeeper_scores = [row.get("goalkeeper_like_score", 0.0) for row in items if row.get("goalkeeper_like_score") is not None]
            goalkeeper_teams = [row.get("goalkeeper_like_team") for row in items if row.get("goalkeeper_like_team") is not None]
            summary = {
                "track_id": int(track_id),
                "display_track_id": mode([row.get("display_track_id") for row in items if row.get("display_track_id") is not None]),
                "scene_segment_id": mode([row.get("scene_segment_id") for row in items if row.get("scene_segment_id") is not None]),
                "jersey_segment_index": mode([row.get("jersey_segment_index") for row in items if row.get("jersey_segment_index") is not None]),
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
                "raw_jersey_distribution": raw_jersey_distribution,
                "jersey_raw_candidates": jersey_raw_candidates,
                "preserved_dropped_jersey_evidence": preserved_dropped_jersey_evidence,
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
                "jersey_full_body_sufficient": mode(
                    [
                        row.get("jersey_full_body_sufficient")
                        for row in items
                        if row.get("jersey_number") == jersey_number
                        and row.get("jersey_full_body_sufficient") is not None
                    ]
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
            display_id = summary.get("display_track_id")
            if display_id is not None and int(display_id) in number_regions:
                summary["number_region_evidence"] = number_regions[int(display_id)]
            summaries.append(summary)
        return summaries

    def summary_group_key(self, row):
        if self.assignment_scope in {"scene_segment", "scene", "segment"}:
            display_id = row.get("display_track_id", row.get("track_id"))
            segment_id = row.get("scene_segment_id")
            if display_id is not None and segment_id is not None:
                return scene_identity_tracklet_id(display_id, segment_id)
        return int(row.get("identity_tracklet_id") or row.get("display_track_id", row["track_id"]))

    def assign(self, summaries):
        if not summaries:
            return {}, []
        if not self.roster:
            return unknown_assignments(summaries, "missing_roster"), []
        if self.assignment_scope in {"scene_segment", "scene", "segment"}:
            assignments = unknown_assignments(summaries, "below_threshold")
            scores = []
            grouped = defaultdict(list)
            for summary in summaries:
                grouped[summary.get("scene_segment_id")].append(summary)
            for _segment_id, segment_summaries in sorted(grouped.items(), key=lambda item: (-1 if item[0] is None else int(item[0]))):
                segment_assignments, segment_scores = self._assign_one_scope(segment_summaries)
                assignments.update(segment_assignments)
                scores.extend(segment_scores)
            scores.sort(key=lambda row: (row["track_id"], row["cost"], row["player_id"]))
            return assignments, scores
        return self._assign_one_scope(summaries)

    def _assign_one_scope(self, summaries):
        scores = self.candidate_scores(summaries)
        cost_matrix = np.asarray(
            [[self.cost_details(tracklet, player, allow_preserved_evidence=False)["cost"] for player in self.roster] for tracklet in summaries],
            dtype=float,
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = unknown_assignments(summaries, "below_threshold")
        for row, col in zip(row_ind, col_ind):
            tracklet = summaries[row]
            player = self.roster[col]
            details = self.cost_details(tracklet, player, allow_preserved_evidence=False)
            confidence = details["confidence"]
            track_id = int(tracklet["track_id"])
            assignment_gate = self.assignment_gate(tracklet, player, details)
            if confidence < self.unknown_threshold:
                # Keep the best rejected candidate in diagnostics. It is useful
                # when tuning thresholds, but it must not become a real identity.
                assignments[track_id]["confidence"] = confidence
                assignments[track_id]["identity_confidence"] = confidence
                assignments[track_id]["identity_status"] = "unknown"
                assignments[track_id]["identity_sources"] = identity_sources(details)
                assignments[track_id]["identity_risk_flags"] = identity_risk_flags(tracklet, assignment_gate, details)
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
                assignments[track_id]["identity_confidence"] = confidence
                assignments[track_id]["identity_status"] = "unknown"
                assignments[track_id]["identity_sources"] = identity_sources(details)
                assignments[track_id]["identity_risk_flags"] = identity_risk_flags(tracklet, assignment_gate, details)
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
                "identity_confidence": confidence,
                "identity_status": "assigned",
                "identity_sources": identity_sources(details),
                "identity_risk_flags": identity_risk_flags(tracklet, assignment_gate, details),
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
                details = self.cost_details(tracklet, player, allow_preserved_evidence=True)
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
                        "tracklet_raw_jersey_distribution": tracklet.get("raw_jersey_distribution"),
                        "tracklet_jersey_raw_candidates": tracklet.get("jersey_raw_candidates"),
                        "tracklet_jersey_roster_mass": tracklet.get("jersey_roster_mass"),
                        "tracklet_number_region_winner": (tracklet.get("number_region_evidence") or {}).get("winner"),
                        "tracklet_number_region_distribution": (tracklet.get("number_region_evidence") or {}).get("raw_distribution"),
                        "tracklet_number_region_votes": (tracklet.get("number_region_evidence") or {}).get("winner_votes"),
                        "tracklet_number_region_confidence": (tracklet.get("number_region_evidence") or {}).get("winner_mean_confidence"),
                        "tracklet_number_region_consecutive_support": (tracklet.get("number_region_evidence") or {}).get("max_consecutive_support"),
                        "tracklet_frames": tracklet.get("num_frames"),
                        "mean_crop_quality": tracklet.get("mean_crop_quality"),
                        "mean_pitch_position": tracklet.get("mean_pitch_position"),
                        "position_prior_distance": details["position_prior_distance"],
                        "visual_similarity": details["visual_similarity"],
                        "jersey_score_source": details.get("jersey_score_source"),
                        "assignment_gate": self.assignment_gate(tracklet, player, details),
                        "cost": details["cost"],
                        "confidence": details["confidence"],
                        "components": details["components"],
                    }
                )
        rows.sort(key=lambda row: (row["track_id"], row["cost"], row["player_id"]))
        return rows

    def cost_details(self, tracklet, player, allow_preserved_evidence=False):
        """Return total cost and each feature contribution for one candidate."""
        components = {
            "base": 0.25,
            "team": 0.0,
            "jersey": 0.0,
            "team_jersey_constraint": 0.0,
            "number_region": 0.0,
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
        jersey_score_source = None
        use_preserved_evidence = (
            bool(tracklet.get("preserved_dropped_jersey_evidence"))
            and bool(allow_preserved_evidence)
        )
        use_trusted_raw_evidence = not bool(tracklet.get("preserved_dropped_jersey_evidence"))
        jersey_score = None
        if use_trusted_raw_evidence or use_preserved_evidence:
            # Preserved dropped evidence is useful for diagnostics/candidates,
            # but it must not become authoritative identity evidence. The
            # caller controls that via allow_preserved_evidence.
            jersey_score = jersey_candidate_score(tracklet, expected, field="raw_jersey_distribution")
            if jersey_score is not None:
                jersey_score_source = "raw_jersey_distribution"
            else:
                jersey_score = jersey_candidate_score(tracklet, expected)
                if jersey_score is not None:
                    jersey_score_source = "jersey_distribution"
        if expected is not None and jersey_score is not None:
            # Candidate distributions let a lower-ranked but roster-valid OCR
            # number still influence the assignment.
            components["jersey"] = -0.45 * jersey_score
        elif observed is not None and expected is not None:
            reliability = jersey_reliability(tracklet)
            components["jersey"] = -0.40 * reliability if int(observed) == int(expected) else 0.55 * max(0.25, reliability)
            if self.enforce_unique_team_jersey and self._tracklet_jersey_reading_is_reliable(tracklet):
                same_known_team = (
                    tracklet.get("team_id") is not None
                    and player.get("team_id") is not None
                    and int(tracklet["team_id"]) == int(player["team_id"])
                )
                if same_known_team and int(observed) != int(expected):
                    components["team_jersey_constraint"] = 0.90
                elif same_known_team and int(observed) == int(expected):
                    components["team_jersey_constraint"] = -0.20 * reliability
        elif expected is not None:
            # Missing jersey is a weak penalty, not a blocker. Many broadcast
            # crops never expose a readable back number.
            components["jersey"] = 0.35

        # Independent of which jersey-scoring branch fired above: whenever the
        # tracklet's own observed number is 1, this structural goalkeeper/#1
        # prior should still apply. It must not be gated behind jersey_score,
        # since the tracklet's own raw_jersey_distribution almost always
        # contains its own mode number as a candidate -- so jersey_score is
        # found whenever observed == expected == 1, which would otherwise
        # silently skip this prior in the common case where a distribution is
        # available (see handoff.md for the discovered gap).
        if self.goalkeeper_number_one_prior and observed is not None and int(observed) == 1:
            reliability = jersey_reliability(tracklet)
            role = str(player.get("role") or "").lower()
            if role in {"goalkeeper", "keeper", "gk"}:
                components["goalkeeper_number_one_prior"] = -self.number_one_goalkeeper_bonus * reliability
            elif role:
                components["goalkeeper_number_one_prior"] = self.number_one_non_goalkeeper_penalty * reliability

        if self.number_region_cost_enabled and expected is not None:
            number_region_score = number_region_candidate_score(
                tracklet,
                expected,
                min_votes=self.number_region_min_votes,
                min_mean_confidence=self.number_region_min_mean_confidence,
            )
            if number_region_score is not None:
                components["number_region"] = -self.number_region_bonus_weight * number_region_score
            elif self._has_reliable_number_region(tracklet):
                components["number_region"] = self.number_region_mismatch_penalty * number_region_reliability(tracklet)

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

        # player.visual_embedding is always None until a roster ships one, so
        # visual_cost/visual_similarity are currently always None in practice
        # (see the strong_evidence_* comment on the constructor).
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
            "jersey_score_source": jersey_score_source,
            "position_prior_distance": position_prior_distance,
            "visual_similarity": visual_similarity,
        }

    def _tracklet_jersey_reading_is_reliable(self, tracklet):
        """Intrinsic quality of the tracklet's own OCR reading, independent of
        any specific roster player. Gated by reliable_jersey_min_votes/
        _min_confidence/_min_head_confidence/_min_winner_margin.

        NOTE: this is NOT the check that assignment_gate() uses to decide the
        "reliable_jersey" pass/fail reason -- that is
        _jersey_evidence_supports_player() below, which uses a different
        threshold (reliable_jersey_min_candidate_score) and looks at evidence
        for one specific player, not the tracklet's own reading in isolation.
        This method is only used for the team/jersey uniqueness constraint in
        cost_details() and as a rare fallback inside
        _jersey_evidence_supports_player() when the raw OCR distribution is
        empty. Do not assume it gates the main assignment decision.
        """
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
        min_winner_margin = self.reliable_jersey_min_winner_margin
        if tracklet.get("jersey_full_body_sufficient") is False:
            min_winner_margin += self.narrow_region_jersey_score_penalty
        if winner_margin is not None and float(winner_margin or 0.0) < min_winner_margin:
            return False
        return True

    def assignment_gate(self, tracklet, player, details):
        """Decide whether a low-cost match has enough evidence to be trusted."""
        if not self.require_assignment_evidence:
            return {"pass": True, "reason": "gate_disabled"}

        reliable_jersey = self._jersey_evidence_supports_player(tracklet, player)
        goalkeeper_singleton = self._has_goalkeeper_singleton_match(tracklet, player)
        team_match = same_team(tracklet, player)
        team_confidence = float(tracklet.get("mean_team_confidence", 0.0) or 0.0)
        visual_similarity = details.get("visual_similarity")
        position_distance = details.get("position_prior_distance")
        tracklet_frames = int(tracklet.get("num_frames") or 0)
        # strong_combined is currently always False in practice: it requires
        # visual_similarity is not None, which requires roster
        # visual_embedding -- see the constructor's strong_evidence_* comment.
        strong_combined = (
            team_match
            and team_confidence >= self.strong_evidence_min_team_confidence
            and visual_similarity is not None
            and visual_similarity >= self.strong_evidence_min_visual_similarity
            and tracklet_frames >= self.strong_evidence_min_tracklet_frames
            and position_distance is not None
            and position_distance <= self.strong_evidence_max_position_distance
        )

        # Prefer the most specific provenance when more than one gate passes.
        # A singleton goalkeeper supported by palette/role evidence may also
        # happen to have a reliable number, but the roster constraint is the
        # evidence that makes this special-case assignment safe.
        if goalkeeper_singleton:
            reason = "goalkeeper_roster_singleton"
        elif reliable_jersey:
            reason = "reliable_jersey"
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

    def _jersey_evidence_supports_player(self, tracklet, player):
        """The check assignment_gate() actually uses for the "reliable_jersey"
        pass/fail reason. Unlike _tracklet_jersey_reading_is_reliable() above,
        this does not require the tracklet's own top-voted number to match
        the player -- it scores whatever candidate exists for the player's
        expected number inside the tracklet's raw OCR distribution, even if
        that candidate is not the tracklet's overall winner. A minority
        candidate can still clear reliable_jersey_min_candidate_score.
        """
        if tracklet.get("preserved_dropped_jersey_evidence"):
            return False
        expected = player.get("jersey_number")
        if expected is None:
            return False
        min_candidate_score = self.reliable_jersey_min_candidate_score
        if tracklet.get("jersey_full_body_sufficient") is False:
            min_candidate_score += self.narrow_region_jersey_score_penalty
        raw_candidates = tracklet.get("raw_jersey_distribution") or tracklet.get("jersey_raw_candidates") or []
        candidate_score = jersey_candidate_score(tracklet, expected, field="raw_jersey_distribution")
        if candidate_score is None:
            candidate_score = jersey_candidate_score(tracklet, expected, field="jersey_raw_candidates")
        if raw_candidates:
            # Roster filtering can promote a valid alternative after dropping
            # impossible numbers. The gate should still look at the raw OCR
            # strength, otherwise weak second-place candidates become identities.
            return candidate_score is not None and candidate_score >= min_candidate_score
        candidate_score = jersey_candidate_score(tracklet, expected)
        if candidate_score is not None and candidate_score >= min_candidate_score:
            return True
        observed = tracklet.get("jersey_number")
        return (
            observed is not None
            and int(observed) == int(expected)
            and self._tracklet_jersey_reading_is_reliable(tracklet)
        )

    def _has_reliable_number_region(self, tracklet):
        evidence = tracklet.get("number_region_evidence") or {}
        winner = evidence.get("winner")
        if winner is None or "?" in str(winner):
            return False
        if int(evidence.get("winner_votes") or 0) < self.number_region_min_votes:
            return False
        if float(evidence.get("winner_mean_confidence") or 0.0) < self.number_region_min_mean_confidence:
            return False
        if int(evidence.get("max_consecutive_support") or 0) < self.number_region_min_consecutive_support:
            return False
        return True

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


def is_preserved_dropped_jersey_evidence(value):
    """Detect OCR evidence kept only for candidate diagnostics after roster drop."""
    if not value:
        return False
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return False
    if not isinstance(value, dict):
        return False
    return value.get("status") == "dropped_preserved_evidence"


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
            "identity_confidence": 0.0,
            "identity_status": "unknown",
            "identity_sources": {},
            "identity_risk_flags": ["missing_roster"] if status == "missing_roster" else [],
            "evidence": {"status": status, "tracklet_frames": summary.get("num_frames", 0)},
        }
        for summary in summaries
    }


def apply_assignments(tracks, assignments, assignment_scope="global"):
    assignment_scope = str(assignment_scope or "global").lower()
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            if assignment_scope in {"scene_segment", "scene", "segment"} and track.get("scene_segment_id") is not None:
                tracklet_id = scene_identity_tracklet_id(track.get("display_track_id", raw_id), track.get("scene_segment_id"))
            else:
                tracklet_id = int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id))
            assignment = assignments.get(tracklet_id)
            if not assignment:
                continue
            if assignment_scope in {"scene_segment", "scene", "segment"}:
                track["identity_tracklet_id"] = int(tracklet_id)
            track["player_id"] = assignment["player_id"]
            track["player_name"] = assignment["player_name"]
            track["jersey_number"] = assignment["jersey_number"]
            track["identity_confidence"] = assignment["confidence"]
            track["identity_status"] = assignment.get("identity_status", assignment.get("evidence", {}).get("status"))
            track["identity_sources"] = assignment.get("identity_sources", {})
            track["identity_risk_flags"] = assignment.get("identity_risk_flags", [])
            track["identity_evidence"] = assignment["evidence"]


def scene_identity_tracklet_id(display_id, scene_segment_id):
    display_id = int(display_id)
    scene_segment_id = int(scene_segment_id)
    if display_id < 0 or scene_segment_id < 0:
        raise ValueError("scene-scoped identity IDs require non-negative display and segment IDs")
    # Scene IDs are negative so they cannot collide with global display IDs or
    # the positive IDs produced by jersey-window segmentation.
    return -(display_id * 100000 + scene_segment_id + 1)


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


def number_region_evidence_by_display(evidence):
    output = {}
    for row in evidence or []:
        try:
            display_id = int(row.get("display_track_id"))
        except Exception:
            continue
        output[display_id] = {
            "display_track_id": display_id,
            "winner": row.get("winner"),
            "winner_votes": int(row.get("winner_votes") or 0),
            "winner_mean_confidence": float(row.get("winner_mean_confidence") or 0.0),
            "recognized_regions": int(row.get("recognized_regions") or 0),
            "sampled_regions": int(row.get("sampled_regions") or 0),
            "frame_diversity": int(row.get("frame_diversity") or 0),
            "max_consecutive_support": int(row.get("max_consecutive_support") or 0),
            "raw_distribution": row.get("raw_distribution") or [],
            "localization_source": row.get("localization_source"),
            "audit_only": bool(row.get("audit_only", True)),
        }
    return output


def number_region_candidate_score(tracklet, expected, min_votes=2, min_mean_confidence=0.20):
    if expected is None:
        return None
    evidence = tracklet.get("number_region_evidence") or {}
    best = None
    for candidate in evidence.get("raw_distribution") or []:
        value = candidate.get("value")
        compatibility = number_region_value_compatibility(value, expected)
        if compatibility <= 0.0:
            continue
        votes = int(candidate.get("votes", 0) or 0)
        mean_confidence = clamp(candidate.get("mean_confidence", 0.0), 0.0, 1.0)
        if votes < int(min_votes) or mean_confidence < float(min_mean_confidence):
            continue
        score = compatibility * mean_confidence * vote_confidence_scale(votes)
        best = score if best is None else max(best, score)
    return best


def number_region_value_compatibility(value, expected):
    try:
        expected_text = str(int(expected))
    except Exception:
        return 0.0
    observed = str(value or "").strip()
    if not observed:
        return 0.0
    if "?" not in observed:
        try:
            return 1.0 if int(observed) == int(expected) else 0.0
        except Exception:
            return 0.0
    if len(observed) != len(expected_text):
        return 0.0
    known = 0
    for observed_digit, expected_digit in zip(observed, expected_text):
        if observed_digit == "?":
            continue
        known += 1
        if observed_digit != expected_digit:
            return 0.0
    return 0.5 if known else 0.0


def number_region_reliability(tracklet):
    evidence = tracklet.get("number_region_evidence") or {}
    confidence = clamp(evidence.get("winner_mean_confidence", 0.0), 0.0, 1.0)
    votes = int(evidence.get("winner_votes") or 0)
    consecutive = int(evidence.get("max_consecutive_support") or 0)
    vote_score = vote_confidence_scale(votes)
    consecutive_score = min(1.0, max(1, consecutive) / 2.0)
    return clamp(confidence * vote_score * consecutive_score, 0.0, 1.0)


def jersey_candidate_score(tracklet, expected, field="jersey_distribution"):
    if expected is None:
        return None
    for candidate in tracklet.get(field) or []:
        if int(candidate.get("jersey_number")) == int(expected):
            confidence = clamp(candidate.get("confidence", 0.0), 0.0, 1.0)
            votes = int(candidate.get("votes", 0) or 0)
            return confidence * vote_confidence_scale(votes)
    return None


def identity_sources(details):
    components = details.get("components") or {}
    sources = {}
    for key in ("team", "jersey", "number_region", "position_prior", "visual", "goalkeeper_role", "tracklet_length", "crop_quality"):
        value = float(components.get(key, 0.0) or 0.0)
        sources[key] = float(clamp(0.5 - value, 0.0, 1.0))
    sources["cost"] = float(details.get("cost", 1.0) or 1.0)
    sources["confidence"] = float(details.get("confidence", 0.0) or 0.0)
    return sources


def identity_risk_flags(tracklet, assignment_gate, details):
    flags = []
    if not assignment_gate.get("pass"):
        flags.append(str(assignment_gate.get("reason") or "assignment_gate_failed"))
    if not tracklet.get("raw_jersey_distribution") and not tracklet.get("jersey_distribution"):
        flags.append("missing_ocr_distribution")
    if tracklet.get("number_region_evidence") is not None and not (tracklet.get("number_region_evidence") or {}).get("raw_distribution"):
        flags.append("missing_number_region_distribution")
    if float(tracklet.get("mean_team_confidence") or 0.0) < 0.50:
        flags.append("low_team_confidence")
    if int(tracklet.get("num_frames") or 0) < 10:
        flags.append("short_tracklet")
    if details.get("visual_similarity") is None:
        flags.append("missing_visual_similarity")
    if float(details.get("confidence") or 0.0) < 0.70:
        flags.append("low_identity_confidence")
    return sorted(set(flags))


def jersey_reliability(tracklet):
    conf = clamp(tracklet.get("jersey_confidence", 0.0), 0.0, 1.0)
    votes = int(tracklet.get("jersey_votes") or 0)
    if votes <= 0:
        return 0.2
    reliability = clamp(conf * vote_confidence_scale(votes), 0.15, 1.0)
    if tracklet.get("jersey_full_body_sufficient") is False:
        # Decision only reached vote quorum via narrow ROI regions, not
        # full_body alone -- validated ~20% accurate vs ~75-84% (see
        # narrow_region_jersey_score_penalty). Halve the soft-cost weight
        # given to this evidence rather than only gating the hard pass/fail.
        reliability *= 0.5
    return reliability


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


# Shared by jersey_candidate_score, number_region_candidate_score,
# number_region_reliability and jersey_reliability: a vote count below this
# saturates the confidence weight linearly instead of counting at full
# strength (e.g. 1 vote -> 1/3 weight, 2 votes -> 2/3 weight, 3+ votes -> full
# weight). Kept as one constant so the four call sites cannot silently drift
# apart if this is ever retuned.
VOTE_SATURATION_COUNT = 3.0


def vote_confidence_scale(votes):
    return min(1.0, max(1, int(votes or 0)) / VOTE_SATURATION_COUNT)
