from collections import defaultdict
import math

from ft.features.visual import cosine_similarity, extract_from_frame, mean_embedding
from ft.utils.geometry import distance


def tracklet_linker_config(config):
    """Translate linking settings into TrackletLinker constructor options."""
    options = dict(config or {})
    options.pop("enabled", None)
    embedding_mode = str(options.pop("embedding_mode", "hsv")).lower()
    hsv_similarity = options.pop("appearance_min_similarity_hsv", None)
    options.pop("max_temporal_candidates", None)
    if (
        "appearance_min_similarity" not in options
        and embedding_mode == "hsv"
        and hsv_similarity is not None
    ):
        options["appearance_min_similarity"] = hsv_similarity
    return options


class TrackletLinker:
    """Link fragmented raw tracks into stable display_track_id values.

    The linker is deliberately conservative. It only merges non-overlapping raw
    tracks when motion, team and appearance are compatible, because a bad merge
    is more damaging to identity assignment than leaving two shorter tracklets.
    """

    def __init__(
        self,
        max_gap=90,
        max_distance=160.0,
        min_frames=4,
        team_gate_enabled=True,
        team_gate_min_confidence=0.65,
        appearance_gate_enabled=True,
        appearance_min_similarity=0.72,
        pitch_gate=None,
        pitch_reranking=None,
        calibration_source=None,
        max_rejection_records=5000,
    ):
        self.max_gap = int(max_gap)
        self.max_distance = float(max_distance)
        self.min_frames = int(min_frames)
        self.team_gate_enabled = bool(team_gate_enabled)
        self.team_gate_min_confidence = float(team_gate_min_confidence)
        self.appearance_gate_enabled = bool(appearance_gate_enabled)
        self.appearance_min_similarity = float(appearance_min_similarity)
        self.pitch_gate = normalize_pitch_gate(pitch_gate)
        self.pitch_reranking = normalize_pitch_reranking(pitch_reranking)
        self.calibration_source = calibration_source
        self.max_rejection_records = int(max_rejection_records)
        self.diagnostics = {}

    def apply(self, tracks, frames=None):
        summaries = self._summaries(tracks.get("players", []), frames=frames)
        display_id_by_track = {track_id: track_id for track_id in summaries}
        tracks_by_display = {track_id: {track_id} for track_id in summaries}
        ordered = sorted(summaries.values(), key=lambda row: (row["start"], row["track_id"]))
        accepted = []
        rejected = []
        candidates = []

        for current in ordered:
            if current["num_frames"] < self.min_frames:
                continue
            best = None
            best_score = None
            eligible = []
            for previous in ordered:
                if previous["track_id"] == current["track_id"]:
                    break
                if previous["num_frames"] < self.min_frames:
                    continue
                gap = current["start"] - previous["end"]
                if gap <= 0 or gap > self.max_gap:
                    self._record_rejection(rejected, current, previous, "gap", gap=gap)
                    continue
                if self._cluster_conflict(current, previous, summaries, display_id_by_track, tracks_by_display):
                    # A display_track_id cannot contain two raw tracks visible in
                    # the same frame; that would create duplicate bodies under
                    # one identity before Hungarian even runs.
                    self._record_rejection(rejected, current, previous, "overlap", gap=gap)
                    continue
                dist = tracklet_distance(previous, current)
                if dist is None:
                    self._record_rejection(rejected, current, previous, "distance", gap=gap, distance=None)
                    continue
                if dist > self.max_distance:
                    self._record_rejection(rejected, current, previous, "distance", gap=gap, distance=dist)
                    continue
                gate = self._gate(current, previous, gap)
                if not gate["pass"]:
                    self._record_rejection(
                        rejected,
                        current,
                        previous,
                        gate["reason"],
                        gap=gap,
                        distance=dist,
                        team_gate_pass=gate.get("team_gate_pass"),
                        appearance_gate_pass=gate.get("appearance_gate_pass"),
                        visual_similarity=gate.get("visual_similarity"),
                        **pitch_gate_payload(gate),
                    )
                    continue
                # Distance is the main link score; gap is a small tie-breaker so
                # a nearby continuation is preferred over a long disappearance.
                score = dist + gap * 0.5
                pitch = self._pitch_gate(previous, current, gap)
                candidate_row = {
                            "from_track_id": int(previous["track_id"]),
                            "to_track_id": int(current["track_id"]),
                            "gap": int(gap),
                            "distance": float(dist),
                            "baseline_score": float(score),
                            "team_gate_pass": gate.get("team_gate_pass"),
                            "appearance_gate_pass": gate.get("appearance_gate_pass"),
                            "visual_similarity": gate.get("visual_similarity"),
                            **pitch_gate_payload(pitch),
                        }
                if len(candidates) < self.max_rejection_records:
                    candidates.append(candidate_row)
                eligible.append({
                    "previous": previous,
                    "baseline_score": float(score),
                    "pitch": pitch,
                    "diagnostic": candidate_row,
                })
                if best_score is None or score < best_score:
                    best_score = score
                    best = previous
            selection = select_pitch_reranked_candidate(
                eligible, self.pitch_reranking
            )
            if selection["selected"] is not None:
                best = selection["selected"]["previous"]
            if best is not None:
                display_id = display_id_by_track[best["track_id"]]
                display_id_by_track[current["track_id"]] = display_id
                tracks_by_display[display_id].add(current["track_id"])
                accepted.append(
                    {
                        "from_track_id": int(best["track_id"]),
                        "to_track_id": int(current["track_id"]),
                        "display_track_id": int(display_id),
                        "gap": int(current["start"] - best["end"]),
                        "distance": tracklet_distance(best, current),
                        "visual_similarity": cosine_similarity(best.get("visual_embedding"), current.get("visual_embedding")),
                        **pitch_gate_payload(self._pitch_gate(best, current, current["start"] - best["end"])),
                        **reranking_selection_payload(selection),
                    }
                )

        for frame_tracks in tracks.get("players", []):
            for raw_track_id, track in frame_tracks.items():
                track["raw_track_id"] = int(raw_track_id)
                track["display_track_id"] = int(display_id_by_track.get(int(raw_track_id), int(raw_track_id)))
        annotate_baseline_candidate_ranks(candidates, accepted)
        self.diagnostics = {
            "enabled": True,
            "num_raw_tracklets": len(summaries),
            "num_display_tracklets": len(set(display_id_by_track.values())),
            "accepted_links": accepted,
            "candidates": candidates,
            "rejected_links": rejected,
            "rejection_counts": count_reasons(rejected),
            "settings": {
                "max_gap": self.max_gap,
                "max_distance": self.max_distance,
                "min_frames": self.min_frames,
                "team_gate_enabled": self.team_gate_enabled,
                "team_gate_min_confidence": self.team_gate_min_confidence,
                "appearance_gate_enabled": self.appearance_gate_enabled,
                "appearance_min_similarity": self.appearance_min_similarity,
                "pitch_gate": self.pitch_gate,
                "pitch_reranking": self.pitch_reranking,
                "calibration_source": self.calibration_source,
            },
            "pitch_gate": pitch_gate_diagnostics(accepted, rejected, self.pitch_gate, self.calibration_source),
            "pitch_reranking": pitch_reranking_diagnostics(candidates, accepted, self.pitch_reranking),
        }
        return display_id_by_track

    @staticmethod
    def ensure_display_ids(tracks):
        for frame_tracks in tracks.get("players", []):
            for raw_track_id, track in frame_tracks.items():
                track.setdefault("raw_track_id", int(raw_track_id))
                track.setdefault("display_track_id", int(raw_track_id))

    def _summaries(self, player_frames, frames=None):
        grouped = defaultdict(list)
        for frame_num, frame_tracks in enumerate(player_frames):
            for track_id, track in frame_tracks.items():
                grouped[int(track_id)].append((frame_num, track))
        summaries = {}
        for track_id, items in grouped.items():
            items.sort(key=lambda item: item[0])
            team_ids = [item[1].get("team") for item in items if item[1].get("team") is not None]
            team_id, team_votes = mode_count(team_ids)
            visual_values = []
            for frame_num, track in items:
                if track.get("visual_embedding") is not None:
                    visual_values.append(track["visual_embedding"])
                elif frames is not None and frame_num < len(frames):
                    visual = extract_from_frame(frames[frame_num], track.get("bbox"))
                    if visual is not None:
                        visual_values.append(visual)
            summaries[track_id] = {
                "track_id": track_id,
                "start": items[0][0],
                "end": items[-1][0],
                "num_frames": len(items),
                "frames": {frame_num for frame_num, _ in items},
                "first_position": items[0][1].get("position"),
                "last_position": items[-1][1].get("position"),
                "first_position_pitch": items[0][1].get("position_pitch"),
                "last_position_pitch": items[-1][1].get("position_pitch"),
                "team_id": team_id,
                "team_votes": team_votes,
                "mean_team_confidence": mean(
                    item[1].get("team_confidence", 0.0)
                    for item in items
                    if item[1].get("team") == team_id
                ),
                "visual_embedding": mean_embedding(visual_values),
            }
        return summaries

    @staticmethod
    def _cluster_conflict(current, previous, summaries, display_id_by_track, tracks_by_display):
        display_id = display_id_by_track[previous["track_id"]]
        for track_id in tracks_by_display.get(display_id, set()):
            if current["frames"].intersection(summaries[track_id]["frames"]):
                return True
        return False

    def _gate(self, current, previous, gap):
        """Reject links that look plausible geometrically but not semantically."""
        team_gate_pass = True
        if self.team_gate_enabled:
            current_team = current.get("team_id")
            previous_team = previous.get("team_id")
            current_conf = float(current.get("mean_team_confidence", 0.0) or 0.0)
            previous_conf = float(previous.get("mean_team_confidence", 0.0) or 0.0)
            if (
                current_team is not None
                and previous_team is not None
                and int(current_team) != int(previous_team)
                and current_conf >= self.team_gate_min_confidence
                and previous_conf >= self.team_gate_min_confidence
            ):
                team_gate_pass = False

        visual_similarity = cosine_similarity(previous.get("visual_embedding"), current.get("visual_embedding"))
        appearance_gate_pass = True
        if (
            self.appearance_gate_enabled
            and visual_similarity is not None
            and visual_similarity < self.appearance_min_similarity
        ):
            appearance_gate_pass = False

        pitch = self._pitch_gate(previous, current, gap)

        if not team_gate_pass:
            return {
                "pass": False,
                "reason": "team_gate",
                "team_gate_pass": False,
                "appearance_gate_pass": appearance_gate_pass,
                "visual_similarity": visual_similarity,
            }
        if not appearance_gate_pass:
            return {
                "pass": False,
                "reason": "appearance_gate",
                "team_gate_pass": True,
                "appearance_gate_pass": False,
                "visual_similarity": visual_similarity,
                **pitch,
            }
        if pitch.get("pitch_gate_blocked"):
            return {
                "pass": False,
                "reason": "pitch_speed_gate",
                "team_gate_pass": True,
                "appearance_gate_pass": True,
                "visual_similarity": visual_similarity,
                **pitch,
            }
        return {
            "pass": True,
            "team_gate_pass": True,
            "appearance_gate_pass": True,
            "visual_similarity": visual_similarity,
            **pitch,
        }

    def _pitch_gate(self, previous, current, gap):
        settings = self.pitch_gate
        source_ok = source_matches(
            self.calibration_source, settings["require_source_prefix"]
        )
        first = point2(current.get("first_position_pitch"))
        last = point2(previous.get("last_position_pitch"))
        endpoints_in_bounds = (
            in_pitch_bounds(last, settings) and in_pitch_bounds(first, settings)
        )
        usable = bool(
            (settings["enabled"] or self.pitch_reranking["enabled"])
            and source_ok
            and gap > 0
            and last is not None
            and first is not None
            and (endpoints_in_bounds or not settings["require_in_bounds"])
        )
        pitch_distance = euclidean(last, first) if usable else None
        duration = gap / settings["fps"] if usable and settings["fps"] > 0 else None
        speed = pitch_distance / duration if duration else None
        would_block = bool(settings["enabled"] and usable and speed > settings["max_speed_mps"])
        return {
            "pitch_gate_mode": settings["mode"],
            "pitch_gate_source_ok": source_ok,
            "pitch_gate_endpoints_in_bounds": endpoints_in_bounds,
            "pitch_gate_usable": usable,
            "pitch_distance_m": pitch_distance,
            "required_speed_mps": speed,
            "pitch_gate_would_block": would_block,
            "pitch_gate_blocked": bool(would_block and settings["mode"] == "apply"),
        }

    def _record_rejection(self, rejected, current, previous, reason, **payload):
        if len(rejected) >= self.max_rejection_records:
            return
        row = {
            "from_track_id": int(previous["track_id"]),
            "to_track_id": int(current["track_id"]),
            "reason": reason,
        }
        row.update({key: normalize_value(value) for key, value in payload.items()})
        rejected.append(row)


def mode_count(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    if not counts:
        return None, 0
    value, count = max(counts.items(), key=lambda item: item[1])
    return value, int(count)


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def count_reasons(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["reason"]] += 1
    return dict(sorted(counts.items()))


def tracklet_distance(previous, current):
    if previous.get("last_position") is None or current.get("first_position") is None:
        return None
    return float(distance(previous["last_position"], current["first_position"]))


def normalize_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def normalize_pitch_gate(value):
    value = dict(value or {})
    mode = str(value.get("mode", "audit")).lower()
    if mode not in {"audit", "apply"}:
        raise ValueError("linking.pitch_gate.mode must be audit or apply")
    fps = float(value.get("fps", 25.0))
    max_speed = float(value.get("max_speed_mps", 14.0))
    if fps <= 0:
        raise ValueError("linking.pitch_gate.fps must be positive")
    if max_speed <= 0:
        raise ValueError("linking.pitch_gate.max_speed_mps must be positive")
    return {
        "enabled": bool(value.get("enabled", False)),
        "mode": mode,
        "max_speed_mps": max_speed,
        "fps": fps,
        "pitch_length": float(value.get("pitch_length", 105.0)),
        "pitch_width": float(value.get("pitch_width", 68.0)),
        "bounds_tolerance": float(value.get("bounds_tolerance", 2.0)),
        "require_in_bounds": bool(value.get("require_in_bounds", True)),
        "require_source_prefix": str(value.get("require_source_prefix", "tvcalib:")),
    }


def normalize_pitch_reranking(value):
    value = dict(value or {})
    mode = str(value.get("mode", "audit")).lower()
    if mode not in {"audit", "apply"}:
        raise ValueError("linking.pitch_reranking.mode must be audit or apply")
    weight = float(value.get("weight", 0.10))
    baseline_scale = float(value.get("baseline_distance_scale", 160.0))
    pitch_scale = float(value.get("pitch_distance_scale", 10.0))
    if weight < 0:
        raise ValueError("linking.pitch_reranking.weight must be non-negative")
    if baseline_scale <= 0 or pitch_scale <= 0:
        raise ValueError("linking.pitch_reranking scales must be positive")
    return {
        "enabled": bool(value.get("enabled", False)),
        "mode": mode,
        "weight": weight,
        "baseline_distance_scale": baseline_scale,
        "pitch_distance_scale": pitch_scale,
        "require_all_candidates_usable": bool(
            value.get("require_all_candidates_usable", True)
        ),
    }


def select_pitch_reranked_candidate(eligible, settings):
    if not eligible:
        return {
            "baseline": None,
            "proposed": None,
            "selected": None,
            "group_usable": False,
            "changed": False,
        }
    baseline = min(
        eligible,
        key=lambda item: (
            item["baseline_score"],
            int(item["previous"]["track_id"]),
        ),
    )
    all_usable = all(item["pitch"].get("pitch_gate_usable") for item in eligible)
    any_usable = any(item["pitch"].get("pitch_gate_usable") for item in eligible)
    group_usable = bool(
        settings["enabled"]
        and (all_usable if settings["require_all_candidates_usable"] else any_usable)
    )
    for item in eligible:
        pitch_distance = item["pitch"].get("pitch_distance_m")
        pitch_cost = (
            settings["weight"] * float(pitch_distance) / settings["pitch_distance_scale"]
            if group_usable and pitch_distance is not None
            else 0.0
        )
        item["soft_score"] = (
            item["baseline_score"] / settings["baseline_distance_scale"] + pitch_cost
        )
        item["diagnostic"]["pitch_soft_score"] = item["soft_score"]
        item["diagnostic"]["pitch_reranking_group_usable"] = group_usable
    proposed = min(
        eligible,
        key=lambda item: (
            item["soft_score"],
            item["baseline_score"],
            int(item["previous"]["track_id"]),
        ),
    )
    changed = proposed is not baseline
    selected = proposed if settings["mode"] == "apply" and group_usable else baseline
    for item in eligible:
        item["diagnostic"]["pitch_reranking_proposed_winner"] = item is proposed
        item["diagnostic"]["pitch_reranking_selected"] = item is selected
    return {
        "baseline": baseline,
        "proposed": proposed,
        "selected": selected,
        "group_usable": group_usable,
        "changed": changed,
    }


def reranking_selection_payload(selection):
    baseline = selection.get("baseline")
    proposed = selection.get("proposed")
    return {
        "pitch_reranking_group_usable": selection.get("group_usable", False),
        "pitch_reranking_changed": selection.get("changed", False),
        "pitch_reranking_baseline_from_track_id": (
            int(baseline["previous"]["track_id"]) if baseline else None
        ),
        "pitch_reranking_proposed_from_track_id": (
            int(proposed["previous"]["track_id"]) if proposed else None
        ),
    }


def pitch_reranking_diagnostics(candidates, accepted, settings):
    changed_targets = {
        int(row["to_track_id"])
        for row in candidates
        if row.get("pitch_reranking_proposed_winner")
        and not row.get("baseline_winner")
    }
    applied = sum(
        row.get("pitch_reranking_changed") is True
        for row in accepted
    ) if settings["mode"] == "apply" else 0
    return {
        "enabled": settings["enabled"],
        "mode": settings["mode"],
        "proposed_changed_targets": len(changed_targets),
        "applied_changed_links": applied,
        "settings": settings,
    }


def point2(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def source_matches(source, prefix):
    return bool(source is not None and str(source).lower().startswith(str(prefix).lower()))


def in_pitch_bounds(point, settings):
    if point is None:
        return False
    tolerance = settings["bounds_tolerance"]
    return bool(
        -tolerance <= point[0] <= settings["pitch_length"] + tolerance
        and -tolerance <= point[1] <= settings["pitch_width"] + tolerance
    )


def euclidean(first, second):
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def pitch_gate_payload(row):
    return {
        key: normalize_value(row.get(key))
        for key in (
            "pitch_gate_mode",
            "pitch_gate_source_ok",
            "pitch_gate_endpoints_in_bounds",
            "pitch_gate_usable",
            "pitch_distance_m",
            "required_speed_mps",
            "pitch_gate_would_block",
            "pitch_gate_blocked",
        )
    }


def pitch_gate_diagnostics(accepted, rejected, settings, source):
    rows = list(accepted) + list(rejected)
    return {
        "enabled": settings["enabled"],
        "mode": settings["mode"],
        "calibration_source": source,
        "source_compatible": source_matches(source, settings["require_source_prefix"]),
        "evaluated_pairs": sum(row.get("pitch_gate_usable") is not None for row in rows),
        "usable_pairs": sum(row.get("pitch_gate_usable") is True for row in rows),
        "would_block_pairs": sum(row.get("pitch_gate_would_block") is True for row in rows),
        "blocked_pairs": sum(row.get("pitch_gate_blocked") is True for row in rows),
        "settings": settings,
    }


def annotate_baseline_candidate_ranks(candidates, accepted):
    accepted_pairs = {
        (int(row["from_track_id"]), int(row["to_track_id"])) for row in accepted
    }
    grouped = defaultdict(list)
    for row in candidates:
        grouped[int(row["to_track_id"])].append(row)
    for rows in grouped.values():
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["baseline_score"]),
                int(row["gap"]),
                int(row["from_track_id"]),
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["baseline_rank"] = rank
            row["baseline_winner"] = rank == 1
            row["accepted_link"] = (
                int(row["from_track_id"]), int(row["to_track_id"])
            ) in accepted_pairs
