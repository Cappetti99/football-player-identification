from collections import defaultdict

import numpy as np

from ft.features.referee import color_to_ranges, normalize_color_ranges, palette_fraction


class GoalkeeperAppearanceAssigner:
    """Assign goalkeeper role from roster-provided kit colors.

    With a person/ball detector, goalkeeper is not a detector class. This module
    recovers the role as a semantic cue from roster kit colours.
    """

    def __init__(
        self,
        color_ranges_by_team=None,
        min_color_fraction=0.45,
        min_tracklet_frames=2,
        assign_team_from_color=True,
        team_correction_min_score=0.55,
    ):
        self.color_ranges_by_team = normalize_ranges_by_team(color_ranges_by_team)
        self.min_color_fraction = float(min_color_fraction)
        self.min_tracklet_frames = int(min_tracklet_frames)
        self.assign_team_from_color = bool(assign_team_from_color)
        self.team_correction_min_score = float(team_correction_min_score)

    def apply(self, frames, tracks):
        diagnostics = {
            "enabled": bool(self.color_ranges_by_team),
            "teams": sorted(self.color_ranges_by_team),
            "tracklets": {},
        }
        if not self.color_ranges_by_team:
            return diagnostics

        by_tracklet = defaultdict(list)
        for frame_num, frame_items in enumerate(tracks.get("players", [])):
            frame = frames[frame_num]
            for raw_id, track in frame_items.items():
                display_id = int(track.get("display_track_id", raw_id))
                sample = classify_goalkeeper_palette(frame, track["bbox"], self.color_ranges_by_team)
                track["goalkeeper_color_evidence"] = sample
                track["goalkeeper_like_score"] = sample["score"]
                track["goalkeeper_like_team"] = sample["team_id"]
                track["goalkeeper_like_color"] = sample["color"]
                by_tracklet[display_id].append(sample)

        summaries = {}
        for display_id, samples in sorted(by_tracklet.items()):
            summary = summarize_goalkeeper_samples(samples)
            # Goalkeeper colours are evaluated at tracklet level. This prevents
            # one bright frame from turning a regular player into a goalkeeper.
            summary["is_goalkeeper_palette"] = (
                summary["team_id"] is not None
                and summary["score"] >= self.min_color_fraction
                and summary["num_samples"] >= self.min_tracklet_frames
            )
            summaries[str(display_id)] = summary
        diagnostics["tracklets"] = summaries

        for frame_items in tracks.get("players", []):
            for raw_id, track in frame_items.items():
                display_id = int(track.get("display_track_id", raw_id))
                summary = summaries.get(str(display_id), {})
                if not summary.get("is_goalkeeper_palette"):
                    continue
                if str(track.get("role_detection") or "").lower() in {"referee", "referee_candidate"}:
                    continue
                track["role_detection"] = "goalkeeper"
                track["goalkeeper_palette_match"] = True
                track["goalkeeper_like_score"] = summary.get("score", 0.0)
                track["goalkeeper_like_team"] = summary.get("team_id")
                track["goalkeeper_like_color"] = summary.get("color")
                self._apply_team_from_color(track, summary)
        return diagnostics

    def _apply_team_from_color(self, track, summary):
        if not self.assign_team_from_color:
            return
        team_id = summary.get("team_id")
        score = float(summary.get("score", 0.0) or 0.0)
        if team_id is None or score < self.team_correction_min_score:
            return
        previous_team = track.get("team")
        if previous_team == team_id:
            return
        # This is intentionally gated by team_correction_min_score. It fixes
        # cases where the team classifier assigns the goalkeeper to the opponent,
        # without letting weak colour matches rewrite player teams.
        track["team"] = int(team_id)
        track["team_confidence"] = max(float(track.get("team_confidence", 0.0) or 0.0), score)
        track["team_evidence"] = {
            "source": "goalkeeper_roster_color",
            "previous_team": previous_team,
            "team": int(team_id),
            "confidence": score,
            "color": summary.get("color"),
        }


def goalkeeper_color_ranges_by_team_from_roster(roster):
    ranges_by_team = {}
    for player in roster or []:
        role = str(player.get("role") or "").lower()
        team_id = player.get("team_id")
        if role not in {"goalkeeper", "keeper", "gk"} or team_id is None:
            continue
        metadata = player.get("metadata", {}) or {}
        color = (
            metadata.get("goalkeeper_color")
            or metadata.get("kit_color")
            or metadata.get("uniform_color")
            or metadata.get("shirt_color")
            or metadata.get("color")
        )
        if color is None:
            continue
        team_id = int(team_id)
        name = f"team{team_id}_goalkeeper_{str(color).strip().lower().replace('#', '').replace(' ', '_')}"
        # Ranges are stored per team because the same colour can mean different
        # things across matches, and only the roster knows the intended keeper.
        ranges_by_team.setdefault(team_id, {}).update(color_to_ranges(color, name=name))
    return ranges_by_team


def classify_goalkeeper_palette(frame, bbox, color_ranges_by_team):
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"team_id": None, "color": None, "score": 0.0, "scores": {}}

    ch, cw = crop.shape[:2]
    torso = crop[int(ch * 0.10) : max(1, int(ch * 0.70)), int(cw * 0.10) : max(1, int(cw * 0.90))]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV) if torso.size else np.empty((0, 0, 3), dtype=np.uint8)
    scores = {}
    for team_id, ranges_by_name in color_ranges_by_team.items():
        for name, ranges in ranges_by_name.items():
            scores[(int(team_id), name)] = palette_fraction(hsv, ranges)
    if not scores:
        return {"team_id": None, "color": None, "score": 0.0, "scores": {}}
    (team_id, color), score = max(scores.items(), key=lambda item: item[1])
    return {
        "team_id": int(team_id),
        "color": color,
        "score": float(score),
        "scores": {f"{team}_{name}": float(value) for (team, name), value in sorted(scores.items())},
    }


def summarize_goalkeeper_samples(samples):
    valid = [sample for sample in samples if sample.get("team_id") is not None]
    if not valid:
        return {"team_id": None, "color": None, "score": 0.0, "num_samples": 0}
    by_team_color = defaultdict(list)
    for sample in valid:
        by_team_color[(int(sample["team_id"]), sample["color"])].append(float(sample["score"]))
    (team_id, color), values = max(by_team_color.items(), key=lambda item: (np.mean(item[1]), len(item[1])))
    return {
        "team_id": int(team_id),
        "color": color,
        "score": float(np.mean(values)),
        "num_samples": len(values),
    }


def normalize_ranges_by_team(color_ranges_by_team):
    normalized = {}
    for team_id, ranges in (color_ranges_by_team or {}).items():
        normalized[int(team_id)] = normalize_color_ranges(ranges)
    return normalized
