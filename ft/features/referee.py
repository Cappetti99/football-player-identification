from collections import defaultdict

import numpy as np


REFEREE_COLOR_RANGES = {
    "black": [
        {"v_max": 70, "s_max": 120},
    ],
    "yellow": [
        {"h_min": 22, "h_max": 38, "s_min": 70, "v_min": 120},
    ],
    "fluorescent_yellow": [
        {"h_min": 20, "h_max": 42, "s_min": 55, "v_min": 135},
    ],
    "orange": [
        {"h_min": 6, "h_max": 24, "s_min": 70, "v_min": 105},
    ],
    "light_blue": [
        {"h_min": 88, "h_max": 112, "s_min": 35, "v_min": 105},
    ],
    "blue": [
        {"h_min": 100, "h_max": 128, "s_min": 45, "v_min": 65},
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
        color_ranges=None,
        trusted_color_min_fraction=0.50,
        trusted_color_override_team_confidence=True,
    ):
        self.min_color_fraction = float(min_color_fraction)
        self.min_tracklet_frames = int(min_tracklet_frames)
        self.reclassify_player_candidates = bool(reclassify_player_candidates)
        self.player_candidate_min_color_fraction = float(player_candidate_min_color_fraction)
        self.player_candidate_max_team_confidence = float(player_candidate_max_team_confidence)
        self.require_palette_color = bool(require_palette_color)
        self.color_ranges = normalize_color_ranges(color_ranges) or REFEREE_COLOR_RANGES
        self.trusted_color_ranges = bool(color_ranges)
        self.trusted_color_min_fraction = float(trusted_color_min_fraction)
        self.trusted_color_override_team_confidence = bool(trusted_color_override_team_confidence)

    def apply(self, frames, tracks):
        diagnostics = {
            "color_ranges": sorted(self.color_ranges),
            "trusted_color_ranges": self.trusted_color_ranges,
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
                sample = classify_referee_palette(frame, track["bbox"], self.color_ranges)
                track["referee_color_evidence"] = sample
                track["referee_like_score"] = sample["score"]
                track["referee_like_color"] = sample["color"]
                by_tracklet[display_id].append(sample)

        assignments = {}
        for display_id, samples in sorted(by_tracklet.items()):
            summary = summarize_samples(samples)
            # The decision is made after aggregating the whole display_track_id.
            # A single yellow/blue frame is too easy to get from boards, boots or
            # occlusion, so sample count is part of the criterion.
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
        if self.require_palette_color and summary.get("color") not in self.color_ranges:
            return False
        if int(summary.get("num_samples", 0)) < self.min_tracklet_frames:
            return False
        team_confidence = float(track.get("team_confidence", 0.0) or 0.0)
        team_unknown = track.get("team") in (None, 0)
        if team_unknown or team_confidence <= self.player_candidate_max_team_confidence:
            return True
        # Roster-provided official colours are stronger than the built-in generic
        # palette, so they may override a confident team colour when the score is
        # high enough.
        return (
            self.trusted_color_ranges
            and self.trusted_color_override_team_confidence
            and score >= self.trusted_color_min_fraction
        )


def classify_referee_palette(frame, bbox, color_ranges=None):
    import cv2

    color_ranges = normalize_color_ranges(color_ranges) or REFEREE_COLOR_RANGES
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
        for name, ranges in color_ranges.items()
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


def referee_color_ranges_from_roster(roster):
    ranges = {}
    for player in roster or []:
        role = str(player.get("role") or "").lower()
        metadata = player.get("metadata", {}) or {}
        color = (
            metadata.get("referee_color")
            or metadata.get("kit_color")
            or metadata.get("uniform_color")
            or metadata.get("shirt_color")
            or metadata.get("color")
        )
        if role not in {"referee", "official", "match_official", "referee_candidate"} and color is None:
            continue
        if role not in {"referee", "official", "match_official", "referee_candidate"}:
            continue
        if color is None:
            continue
        name = f"roster_{str(color).strip().lower().replace('#', '').replace(' ', '_')}"
        # The roster decides which official colours are trusted for this match.
        # This avoids hard-coding "yellow referee" across competitions.
        ranges.update(color_to_ranges(color, name=name))
    return ranges


def normalize_color_ranges(color_ranges):
    if not color_ranges:
        return {}
    normalized = {}
    for name, ranges in color_ranges.items():
        if isinstance(ranges, str):
            normalized.update(color_to_ranges(ranges, name=name))
        else:
            normalized[str(name)] = list(ranges)
    return normalized


def color_to_ranges(color, name=None, h_tolerance=12, s_min=45, v_min=95):
    if color is None:
        return {}
    if isinstance(color, str):
        value = color.strip().lower()
        if value in REFEREE_COLOR_RANGES:
            return {name or value: REFEREE_COLOR_RANGES[value]}
        rgb = parse_hex_color(value)
    elif isinstance(color, (list, tuple)) and len(color) == 3:
        rgb = tuple(int(v) for v in color)
    else:
        return {}
    if rgb is None:
        return {}

    import cv2

    r, g, b = rgb
    hsv = cv2.cvtColor(np.asarray([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    hue = int(hsv[0])
    h_min = hue - int(h_tolerance)
    h_max = hue + int(h_tolerance)
    rules = []
    if h_min < 0:
        rules.append({"h_min": 0, "h_max": h_max, "s_min": s_min, "v_min": v_min})
        rules.append({"h_min": 180 + h_min, "h_max": 179, "s_min": s_min, "v_min": v_min})
    elif h_max > 179:
        rules.append({"h_min": h_min, "h_max": 179, "s_min": s_min, "v_min": v_min})
        rules.append({"h_min": 0, "h_max": h_max - 180, "s_min": s_min, "v_min": v_min})
    else:
        rules.append({"h_min": h_min, "h_max": h_max, "s_min": s_min, "v_min": v_min})
    return {name or "custom": rules}


def parse_hex_color(value):
    text = str(value).strip().lower()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


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
