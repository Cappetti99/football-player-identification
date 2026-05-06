from collections import defaultdict

import numpy as np


REFEREE_COLOR_RANGES = {
    "black": [
        {"v_max": 70, "s_max": 120},
    ],
    "yellow": [
        {"h_min": 22, "h_max": 38, "s_min": 70, "v_min": 120},
    ],
    "light_blue": [
        {"h_min": 88, "h_max": 112, "s_min": 35, "v_min": 105},
    ],
    "red": [
        {"h_min": 0, "h_max": 10, "s_min": 70, "v_min": 70},
        {"h_min": 170, "h_max": 179, "s_min": 70, "v_min": 70},
    ],
}


class RefereeAppearanceAssigner:
    """Soft colour diagnostics for referee-like clothing.

    Football laws require match officials to be distinguishable from players;
    they do not define a universal official colour list. This module therefore
    exports a soft cue, not a hard class override.
    """

    def __init__(
        self,
        min_color_fraction=0.22,
        min_tracklet_frames=2,
        reclassify_player_candidates=True,
        player_candidate_min_color_fraction=0.42,
        player_candidate_max_team_confidence=0.50,
        require_palette_color=True,
    ):
        self.min_color_fraction = float(min_color_fraction)
        self.min_tracklet_frames = int(min_tracklet_frames)
        self.reclassify_player_candidates = bool(reclassify_player_candidates)
        self.player_candidate_min_color_fraction = float(player_candidate_min_color_fraction)
        self.player_candidate_max_team_confidence = float(player_candidate_max_team_confidence)
        self.require_palette_color = bool(require_palette_color)

    def apply(self, frames, tracks):
        diagnostics = {
            "referees": self._apply_group(frames, tracks.get("referees", []), mark_referee=True),
            "players": self._apply_group(frames, tracks.get("players", []), mark_referee=False),
        }
        return diagnostics

    def _apply_group(self, frames, frame_tracks, mark_referee):
        by_tracklet = defaultdict(list)
        for frame_num, frame_items in enumerate(frame_tracks):
            frame = frames[frame_num]
            for raw_id, track in frame_items.items():
                display_id = int(track.get("display_track_id", raw_id))
                sample = classify_referee_palette(frame, track["bbox"])
                track["referee_color_evidence"] = sample
                track["referee_like_score"] = sample["score"]
                track["referee_like_color"] = sample["color"]
                by_tracklet[display_id].append(sample)

        assignments = {}
        for display_id, samples in sorted(by_tracklet.items()):
            summary = summarize_samples(samples)
            summary["is_referee_palette"] = (
                summary["score"] >= self.min_color_fraction
                and summary["num_samples"] >= self.min_tracklet_frames
            )
            assignments[str(display_id)] = summary

        for frame_items in frame_tracks:
            for raw_id, track in frame_items.items():
                display_id = int(track.get("display_track_id", raw_id))
                summary = assignments.get(str(display_id), {})
                track["referee_like_score"] = summary.get("score", track.get("referee_like_score", 0.0))
                track["referee_like_color"] = summary.get("color", track.get("referee_like_color"))
                track["referee_palette_match"] = bool(summary.get("is_referee_palette", False))
                if mark_referee:
                    track["role_detection"] = "referee"
                elif self.reclassify_player_candidates and self._is_player_referee_candidate(track, summary):
                    track["role_detection"] = "referee_candidate"
                    track["referee_candidate_reason"] = {
                        "score": summary.get("score", 0.0),
                        "color": summary.get("color"),
                        "team_id": track.get("team"),
                        "team_confidence": track.get("team_confidence", 0.0),
                    }
        return assignments

    def _is_player_referee_candidate(self, track, summary):
        score = float(summary.get("score", 0.0))
        if score < self.player_candidate_min_color_fraction:
            return False
        if self.require_palette_color and summary.get("color") not in REFEREE_COLOR_RANGES:
            return False
        if int(summary.get("num_samples", 0)) < self.min_tracklet_frames:
            return False
        team_confidence = float(track.get("team_confidence", 0.0) or 0.0)
        team_unknown = track.get("team") in (None, 0)
        return team_unknown or team_confidence <= self.player_candidate_max_team_confidence


def classify_referee_palette(frame, bbox):
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"color": None, "score": 0.0, "scores": {}, "shorts_black_score": 0.0}

    ch, cw = crop.shape[:2]
    upper = crop[int(ch * 0.12) : max(1, int(ch * 0.62)), int(cw * 0.12) : max(1, int(cw * 0.88))]
    lower = crop[int(ch * 0.58) : max(1, int(ch * 0.88)), int(cw * 0.18) : max(1, int(cw * 0.82))]
    upper_hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV) if upper.size else np.empty((0, 0, 3), dtype=np.uint8)
    lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV) if lower.size else np.empty((0, 0, 3), dtype=np.uint8)

    scores = {
        name: palette_fraction(upper_hsv, ranges)
        for name, ranges in REFEREE_COLOR_RANGES.items()
    }
    color, score = max(scores.items(), key=lambda item: item[1]) if scores else (None, 0.0)
    shorts_black_score = palette_fraction(lower_hsv, REFEREE_COLOR_RANGES["black"])
    return {
        "color": color,
        "score": float(score),
        "scores": {key: float(value) for key, value in sorted(scores.items())},
        "shorts_black_score": float(shorts_black_score),
    }


def palette_fraction(hsv, ranges):
    if hsv.size == 0:
        return 0.0
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    for rule in ranges:
        current = np.ones(hsv.shape[:2], dtype=bool)
        if "h_min" in rule:
            current &= h >= int(rule["h_min"])
        if "h_max" in rule:
            current &= h <= int(rule["h_max"])
        if "s_min" in rule:
            current &= s >= int(rule["s_min"])
        if "s_max" in rule:
            current &= s <= int(rule["s_max"])
        if "v_min" in rule:
            current &= v >= int(rule["v_min"])
        if "v_max" in rule:
            current &= v <= int(rule["v_max"])
        mask |= current
    return float(mask.mean()) if mask.size else 0.0


def summarize_samples(samples):
    valid = [sample for sample in samples if sample.get("color") is not None]
    if not valid:
        return {"color": None, "score": 0.0, "num_samples": 0, "shorts_black_score": 0.0}
    by_color = defaultdict(list)
    for sample in valid:
        by_color[sample["color"]].append(float(sample["score"]))
    color, values = max(by_color.items(), key=lambda item: (np.mean(item[1]), len(item[1])))
    return {
        "color": color,
        "score": float(np.mean(values)),
        "num_samples": len(valid),
        "shorts_black_score": float(np.mean([sample.get("shorts_black_score", 0.0) for sample in valid])),
    }
