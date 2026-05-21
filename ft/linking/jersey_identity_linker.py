from collections import defaultdict

from ft.utils.geometry import distance


class JerseyIdentityLinker:
    """Merge display IDs when OCR gives a strong same-number continuation.

    The regular tracklet linker runs before OCR, so it cannot use jersey
    evidence. This post-OCR pass handles the narrower case where two
    non-overlapping display IDs have the same reliable team/jersey and are close
    enough in image space to be a plausible reappearance.
    """

    def __init__(
        self,
        max_gap=90,
        max_distance=140.0,
        min_frames=3,
        min_confidence=0.20,
        min_head_confidence=0.55,
        min_winner_margin=0.10,
        min_votes=5,
        team_gate_min_confidence=0.60,
        max_rejection_records=5000,
    ):
        self.max_gap = int(max_gap)
        self.max_distance = float(max_distance)
        self.min_frames = int(min_frames)
        self.min_confidence = float(min_confidence)
        self.min_head_confidence = float(min_head_confidence)
        self.min_winner_margin = float(min_winner_margin)
        self.min_votes = int(min_votes)
        self.team_gate_min_confidence = float(team_gate_min_confidence)
        self.max_rejection_records = int(max_rejection_records)

    def apply(self, tracks, rows=None):
        summaries = self._summaries(tracks.get("players", []))
        display_id_by_display = {display_id: display_id for display_id in summaries}
        displays_by_cluster = {display_id: {display_id} for display_id in summaries}
        ordered = sorted(summaries.values(), key=lambda item: (item["start"], item["display_id"]))
        accepted = []
        rejected = []

        for current in ordered:
            if not current["reliable_jersey"]:
                continue
            best = None
            best_score = None
            for previous in ordered:
                if previous["display_id"] == current["display_id"]:
                    break
                if not previous["reliable_jersey"]:
                    continue
                gap = current["start"] - previous["end"]
                if gap <= 0 or gap > self.max_gap:
                    self._record_rejection(rejected, current, previous, "gap", gap=gap)
                    continue
                if self._cluster_conflict(current, previous, summaries, display_id_by_display, displays_by_cluster):
                    self._record_rejection(rejected, current, previous, "overlap", gap=gap)
                    continue
                if not same_team_jersey(current, previous, self.team_gate_min_confidence):
                    self._record_rejection(rejected, current, previous, "team_or_jersey", gap=gap)
                    continue
                dist = tracklet_distance(previous, current)
                if dist is None or dist > self.max_distance:
                    self._record_rejection(rejected, current, previous, "distance", gap=gap, distance=dist)
                    continue
                # A close next-frame continuation should win over an older
                # same-number tracklet even when both have strong OCR.
                score = dist + gap * 0.5
                if best_score is None or score < best_score:
                    best_score = score
                    best = previous
            if best is None:
                continue
            canonical_id = display_id_by_display[best["display_id"]]
            display_id_by_display[current["display_id"]] = canonical_id
            displays_by_cluster[canonical_id].add(current["display_id"])
            accepted.append(
                {
                    "from_display_track_id": int(best["display_id"]),
                    "to_display_track_id": int(current["display_id"]),
                    "display_track_id": int(canonical_id),
                    "team_id": int(current["team_id"]),
                    "jersey_number": int(current["jersey_number"]),
                    "gap": int(current["start"] - best["end"]),
                    "distance": tracklet_distance(best, current),
                    "from_jersey_confidence": float(best["jersey_confidence"]),
                    "to_jersey_confidence": float(current["jersey_confidence"]),
                }
            )

        changed = apply_display_mapping(tracks, rows, display_id_by_display)
        return {
            "enabled": True,
            "num_display_tracklets_before": len(summaries),
            "num_display_tracklets_after": len(set(display_id_by_display.values())),
            "changed_rows": int(changed),
            "accepted_links": accepted,
            "rejected_links": rejected,
            "rejection_counts": count_reasons(rejected),
            "settings": {
                "max_gap": self.max_gap,
                "max_distance": self.max_distance,
                "min_frames": self.min_frames,
                "min_confidence": self.min_confidence,
                "min_head_confidence": self.min_head_confidence,
                "min_winner_margin": self.min_winner_margin,
                "min_votes": self.min_votes,
                "team_gate_min_confidence": self.team_gate_min_confidence,
            },
        }

    def _summaries(self, player_frames):
        grouped = defaultdict(list)
        for frame_num, frame_tracks in enumerate(player_frames):
            for raw_id, track in frame_tracks.items():
                display_id = int(track.get("display_track_id", raw_id))
                grouped[display_id].append((frame_num, raw_id, track))

        summaries = {}
        for display_id, items in grouped.items():
            items.sort(key=lambda item: item[0])
            jerseys = [item[2].get("jersey_number") for item in items if item[2].get("jersey_number") not in (None, "", -1)]
            teams = [item[2].get("team") for item in items if item[2].get("team") not in (None, "", 0)]
            jersey_number, jersey_count = mode_count(jerseys)
            team_id, team_count = mode_count(teams)
            confidences = [jersey_confidence(item[2]) for item in items if item[2].get("jersey_number") == jersey_number]
            head_confidences = [jersey_head_confidence(item[2]) for item in items if item[2].get("jersey_number") == jersey_number]
            margins = [jersey_winner_margin(item[2]) for item in items if item[2].get("jersey_number") == jersey_number]
            votes = [int(item[2].get("jersey_votes", 0) or 0) for item in items if item[2].get("jersey_number") == jersey_number]
            team_confidences = [float(item[2].get("team_confidence", 0.0) or 0.0) for item in items if item[2].get("team") == team_id]
            reliable = (
                len(items) >= self.min_frames
                and jersey_number is not None
                and team_id is not None
                and max(confidences or [0.0]) >= self.min_confidence
                and max(head_confidences or [0.0]) >= self.min_head_confidence
                and max(margins or [0.0]) >= self.min_winner_margin
                and max(votes or [0]) >= self.min_votes
            )
            summaries[display_id] = {
                "display_id": int(display_id),
                "start": int(items[0][0]),
                "end": int(items[-1][0]),
                "frames": {frame_num for frame_num, _raw_id, _track in items},
                "num_frames": int(len(items)),
                "first_position": items[0][2].get("position"),
                "last_position": items[-1][2].get("position"),
                "team_id": int(team_id) if team_id is not None else None,
                "team_votes": int(team_count),
                "mean_team_confidence": average(team_confidences),
                "jersey_number": int(jersey_number) if jersey_number is not None else None,
                "jersey_votes": int(max(votes or [0])),
                "jersey_track_votes": int(jersey_count),
                "jersey_confidence": float(max(confidences or [0.0])),
                "jersey_head_confidence": float(max(head_confidences or [0.0])),
                "jersey_winner_margin": float(max(margins or [0.0])),
                "reliable_jersey": bool(reliable),
            }
        return summaries

    @staticmethod
    def _cluster_conflict(current, previous, summaries, display_id_by_display, displays_by_cluster):
        canonical_id = display_id_by_display[previous["display_id"]]
        for display_id in displays_by_cluster.get(canonical_id, set()):
            if current["frames"].intersection(summaries[display_id]["frames"]):
                return True
        return False

    def _record_rejection(self, rejected, current, previous, reason, **payload):
        if len(rejected) >= self.max_rejection_records:
            return
        row = {
            "from_display_track_id": int(previous["display_id"]),
            "to_display_track_id": int(current["display_id"]),
            "reason": reason,
        }
        row.update(payload)
        rejected.append(row)


def apply_display_mapping(tracks, rows, display_id_by_display):
    changed = 0
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            old_display_id = int(track.get("display_track_id", raw_id))
            new_display_id = int(display_id_by_display.get(old_display_id, old_display_id))
            if new_display_id == old_display_id:
                continue
            track["jersey_link_previous_display_track_id"] = old_display_id
            track["display_track_id"] = new_display_id
            changed += 1
    if rows is not None:
        for row in rows:
            old_display_id = int(row.get("display_track_id", row.get("track_id", 0)))
            new_display_id = int(display_id_by_display.get(old_display_id, old_display_id))
            if new_display_id == old_display_id:
                continue
            row["jersey_link_previous_display_track_id"] = old_display_id
            row["display_track_id"] = new_display_id
    return changed


def same_team_jersey(current, previous, team_gate_min_confidence):
    if current.get("jersey_number") is None or previous.get("jersey_number") is None:
        return False
    if int(current["jersey_number"]) != int(previous["jersey_number"]):
        return False
    if current.get("team_id") is None or previous.get("team_id") is None:
        return False
    if int(current["team_id"]) == int(previous["team_id"]):
        return True
    current_conf = float(current.get("mean_team_confidence", 0.0) or 0.0)
    previous_conf = float(previous.get("mean_team_confidence", 0.0) or 0.0)
    return current_conf < team_gate_min_confidence or previous_conf < team_gate_min_confidence


def jersey_confidence(track):
    evidence = track.get("jersey_evidence") or {}
    return float(evidence.get("confidence", track.get("jersey_confidence", 0.0)) or 0.0)


def jersey_head_confidence(track):
    evidence = track.get("jersey_evidence") or {}
    return float(evidence.get("head_confidence", track.get("jersey_head_confidence", 0.0)) or 0.0)


def jersey_winner_margin(track):
    evidence = track.get("jersey_evidence") or {}
    return float(evidence.get("winner_margin", track.get("jersey_winner_margin", 0.0)) or 0.0)


def tracklet_distance(previous, current):
    if previous.get("last_position") is None or current.get("first_position") is None:
        return None
    return float(distance(previous["last_position"], current["first_position"]))


def mode_count(values):
    counts = defaultdict(int)
    for value in values:
        if value in (None, "", "None"):
            continue
        counts[value] += 1
    if not counts:
        return None, 0
    value, count = max(counts.items(), key=lambda item: item[1])
    return value, int(count)


def average(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def count_reasons(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["reason"]] += 1
    return dict(sorted(counts.items()))
