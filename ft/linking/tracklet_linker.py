from collections import defaultdict

from ft.features.visual import cosine_similarity, extract_from_frame, mean_embedding
from ft.utils.geometry import distance


class TrackletLinker:
    """Link fragmented raw tracks into stable display_track_id values."""

    def __init__(
        self,
        max_gap=90,
        max_distance=160.0,
        min_frames=4,
        team_gate_enabled=True,
        team_gate_min_confidence=0.65,
        appearance_gate_enabled=True,
        appearance_min_similarity=0.72,
        max_rejection_records=5000,
    ):
        self.max_gap = int(max_gap)
        self.max_distance = float(max_distance)
        self.min_frames = int(min_frames)
        self.team_gate_enabled = bool(team_gate_enabled)
        self.team_gate_min_confidence = float(team_gate_min_confidence)
        self.appearance_gate_enabled = bool(appearance_gate_enabled)
        self.appearance_min_similarity = float(appearance_min_similarity)
        self.max_rejection_records = int(max_rejection_records)
        self.diagnostics = {}

    def apply(self, tracks, frames=None):
        summaries = self._summaries(tracks.get("players", []), frames=frames)
        display_id_by_track = {track_id: track_id for track_id in summaries}
        tracks_by_display = {track_id: {track_id} for track_id in summaries}
        ordered = sorted(summaries.values(), key=lambda row: (row["start"], row["track_id"]))
        accepted = []
        rejected = []

        for current in ordered:
            if current["num_frames"] < self.min_frames:
                continue
            best = None
            best_score = None
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
                    self._record_rejection(rejected, current, previous, "overlap", gap=gap)
                    continue
                dist = tracklet_distance(previous, current)
                if dist is None:
                    self._record_rejection(rejected, current, previous, "distance", gap=gap, distance=None)
                    continue
                if dist > self.max_distance:
                    self._record_rejection(rejected, current, previous, "distance", gap=gap, distance=dist)
                    continue
                gate = self._gate(current, previous)
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
                    )
                    continue
                score = dist + gap * 0.5
                if best_score is None or score < best_score:
                    best_score = score
                    best = previous
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
                    }
                )

        for frame_tracks in tracks.get("players", []):
            for raw_track_id, track in frame_tracks.items():
                track["raw_track_id"] = int(raw_track_id)
                track["display_track_id"] = int(display_id_by_track.get(int(raw_track_id), int(raw_track_id)))
        self.diagnostics = {
            "enabled": True,
            "num_raw_tracklets": len(summaries),
            "num_display_tracklets": len(set(display_id_by_track.values())),
            "accepted_links": accepted,
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
            },
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

    def _gate(self, current, previous):
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
            }
        return {
            "pass": True,
            "team_gate_pass": True,
            "appearance_gate_pass": True,
            "visual_similarity": visual_similarity,
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
