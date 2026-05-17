from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """Conservative team assignment from torso color aggregated by tracklet."""

    def __init__(
        self,
        max_seed_frames=12,
        min_seed_colors=8,
        min_cluster_separation=30.0,
        min_classification_margin=12.0,
        min_tracklet_colors=3,
    ):
        self.max_seed_frames = int(max_seed_frames)
        self.min_seed_colors = int(min_seed_colors)
        self.min_cluster_separation = float(min_cluster_separation)
        self.min_classification_margin = float(min_classification_margin)
        self.min_tracklet_colors = int(min_tracklet_colors)
        self.kmeans = None
        self.team_colors = {}

    def fit_apply(self, frames, tracks):
        if not frames or not tracks.get("players"):
            return {}
        seed_colors = []
        seeded = 0
        for frame_num, player_tracks in enumerate(tracks["players"]):
            if not player_tracks:
                continue
            for track in player_tracks.values():
                color = self.player_color(frames[frame_num], track["bbox"])
                if color is not None:
                    seed_colors.append(color)
            seeded += 1
            if seeded >= self.max_seed_frames and len(seed_colors) >= self.min_seed_colors:
                break
        self._fit(seed_colors)
        assignments = self._assign_tracklets(frames, tracks)
        frame_assignments = self._assign_frames(frames, tracks)
        self._apply(assignments, frame_assignments, tracks)
        return assignments

    def _fit(self, colors):
        if len(colors) < self.min_seed_colors:
            return
        x = np.asarray(colors, dtype=np.float32)
        model = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x)
        centers = model.cluster_centers_
        if np.linalg.norm(centers[0] - centers[1]) < self.min_cluster_separation:
            return
        self.kmeans = model
        self.team_colors = {
            1: tuple(int(v) for v in centers[0]),
            2: tuple(int(v) for v in centers[1]),
        }

    def _assign_tracklets(self, frames, tracks):
        colors_by_tracklet = defaultdict(list)
        for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
            frame = frames[frame_num]
            for raw_id, track in frame_tracks.items():
                display_id = int(track.get("display_track_id", raw_id))
                color = self.player_color(frame, track["bbox"])
                if color is not None:
                    colors_by_tracklet[display_id].append(color)
        assignments = {}
        for display_id, colors in colors_by_tracklet.items():
            assignments[display_id] = self._classify_colors(colors)
        return assignments

    def _classify_colors(self, colors):
        if self.kmeans is None or len(colors) < self.min_tracklet_colors:
            return {"team": None, "confidence": 0.0, "num_colors": len(colors)}
        distances = self.kmeans.transform(np.asarray(colors, dtype=np.float32))
        votes = []
        margins = []
        for row in distances:
            order = np.argsort(row)
            margins.append(float(row[order[1]] - row[order[0]]))
            if margins[-1] >= self.min_classification_margin:
                votes.append(int(order[0]) + 1)
        if not votes:
            return {"team": None, "confidence": 0.0, "num_colors": len(colors)}
        counts = defaultdict(int)
        for vote in votes:
            counts[vote] += 1
        team, count = max(counts.items(), key=lambda item: item[1])
        confidence = count / max(1, len(colors))
        return {
            "team": int(team),
            "confidence": float(confidence),
            "num_colors": len(colors),
            "votes": dict(counts),
            "mean_margin": float(np.mean(margins)) if margins else 0.0,
        }

    def _assign_frames(self, frames, tracks):
        frame_assignments = []
        for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
            frame = frames[frame_num]
            frame_row = {}
            for raw_id, track in frame_tracks.items():
                color = self.player_color(frame, track["bbox"])
                frame_row[int(raw_id)] = self._classify_color(color)
            frame_assignments.append(frame_row)
        return frame_assignments

    def _classify_color(self, color):
        if self.kmeans is None or color is None:
            return {"team": None, "confidence": 0.0, "margin": 0.0, "distances": {}}
        distances = self.kmeans.transform(np.asarray([color], dtype=np.float32))[0]
        order = np.argsort(distances)
        margin = float(distances[order[1]] - distances[order[0]])
        if margin < self.min_classification_margin:
            team = None
            confidence = 0.0
        else:
            team = int(order[0]) + 1
            confidence = min(1.0, margin / max(1.0, self.min_classification_margin * 2.0))
        return {
            "team": team,
            "confidence": float(confidence),
            "margin": float(margin),
            "distances": {int(index) + 1: float(value) for index, value in enumerate(distances)},
        }

    def _apply(self, assignments, frame_assignments, tracks):
        for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
            for raw_id, track in frame_tracks.items():
                display_id = int(track.get("display_track_id", raw_id))
                assignment = assignments.get(display_id, {})
                frame_assignment = frame_assignments[frame_num].get(int(raw_id), {})
                team = assignment.get("team")
                track["team"] = team
                track["team_confidence"] = float(assignment.get("confidence", 0.0))
                track["team_color"] = self.team_colors.get(team, (160, 160, 160))
                track["team_evidence"] = assignment
                track["frame_team"] = frame_assignment.get("team")
                track["frame_team_confidence"] = float(frame_assignment.get("confidence", 0.0))
                track["frame_team_margin"] = float(frame_assignment.get("margin", 0.0))
                track["frame_team_evidence"] = frame_assignment

    @staticmethod
    def player_color(frame, bbox):
        crop = torso_crop(frame, bbox)
        if crop is None or crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Ignore green field pixels and very dark/bright noise.
        green = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 35)
        valid = (~green) & (hsv[:, :, 2] > 35) & (hsv[:, :, 2] < 245)
        pixels = lab[valid]
        if len(pixels) < 20:
            pixels = lab.reshape(-1, 3)
        return np.median(pixels, axis=0).astype(float).tolist()


def torso_crop(frame, bbox):
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(bbox[0])))
    y1 = max(0, min(height - 1, int(bbox[1])))
    x2 = max(0, min(width, int(bbox[2])))
    y2 = max(0, min(height, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    box_h = y2 - y1
    box_w = x2 - x1
    top = y1 + int(box_h * 0.15)
    bottom = y1 + int(box_h * 0.65)
    left = x1 + int(box_w * 0.15)
    right = x2 - int(box_w * 0.15)
    return frame[top:max(top + 1, bottom), left:max(left + 1, right)]
