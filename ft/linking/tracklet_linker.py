from collections import defaultdict

from ft.utils.geometry import distance


class TrackletLinker:
    """Link fragmented raw tracks into stable display_track_id values."""

    def __init__(self, max_gap=90, max_distance=160.0, min_frames=4):
        self.max_gap = int(max_gap)
        self.max_distance = float(max_distance)
        self.min_frames = int(min_frames)

    def apply(self, tracks):
        summaries = self._summaries(tracks.get("players", []))
        display_id_by_track = {track_id: track_id for track_id in summaries}
        tracks_by_display = {track_id: {track_id} for track_id in summaries}
        ordered = sorted(summaries.values(), key=lambda row: (row["start"], row["track_id"]))

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
                    continue
                if self._cluster_conflict(current, previous, summaries, display_id_by_track, tracks_by_display):
                    continue
                dist = distance(previous["last_position"], current["first_position"])
                if dist > self.max_distance:
                    continue
                score = dist + gap * 0.5
                if best_score is None or score < best_score:
                    best_score = score
                    best = previous
            if best is not None:
                display_id = display_id_by_track[best["track_id"]]
                display_id_by_track[current["track_id"]] = display_id
                tracks_by_display[display_id].add(current["track_id"])

        for frame_tracks in tracks.get("players", []):
            for raw_track_id, track in frame_tracks.items():
                track["raw_track_id"] = int(raw_track_id)
                track["display_track_id"] = int(display_id_by_track.get(int(raw_track_id), int(raw_track_id)))
        return display_id_by_track

    @staticmethod
    def ensure_display_ids(tracks):
        for frame_tracks in tracks.get("players", []):
            for raw_track_id, track in frame_tracks.items():
                track.setdefault("raw_track_id", int(raw_track_id))
                track.setdefault("display_track_id", int(raw_track_id))

    def _summaries(self, frames):
        grouped = defaultdict(list)
        for frame_num, frame_tracks in enumerate(frames):
            for track_id, track in frame_tracks.items():
                grouped[int(track_id)].append((frame_num, track))
        summaries = {}
        for track_id, items in grouped.items():
            items.sort(key=lambda item: item[0])
            summaries[track_id] = {
                "track_id": track_id,
                "start": items[0][0],
                "end": items[-1][0],
                "num_frames": len(items),
                "frames": {frame_num for frame_num, _ in items},
                "first_position": items[0][1].get("position"),
                "last_position": items[-1][1].get("position"),
            }
        return summaries

    @staticmethod
    def _cluster_conflict(current, previous, summaries, display_id_by_track, tracks_by_display):
        display_id = display_id_by_track[previous["track_id"]]
        for track_id in tracks_by_display.get(display_id, set()):
            if current["frames"].intersection(summaries[track_id]["frames"]):
                return True
        return False

