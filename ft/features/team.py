from collections import defaultdict

import numpy as np

from ft.features.referee import color_to_ranges, palette_fraction


TEAM_COLOR_RANGES = {
    "black": [
        {"v_max": 72, "s_max": 140},
    ],
    "white": [
        {"s_max": 55, "v_min": 150},
    ],
    "blue": [
        {"h_min": 98, "h_max": 132, "s_min": 45, "v_min": 45},
    ],
    "dark_blue": [
        {"h_min": 98, "h_max": 132, "s_min": 45, "v_min": 28, "v_max": 135},
    ],
    "red": [
        {"h_min": 0, "h_max": 10, "s_min": 70, "v_min": 70},
        {"h_min": 170, "h_max": 179, "s_min": 70, "v_min": 70},
    ],
}

TEAM_COLOR_ALIASES = {
    "black_blue": ["black", "blue"],
    "blackblue": ["black", "blue"],
    "black_and_blue": ["black", "blue"],
    "nero_blu": ["black", "blue"],
    "neroblu": ["black", "blue"],
    "nerazzurro": ["black", "blue"],
    "nerazzurri": ["black", "blue"],
    "lightblue": ["light_blue"],
    "light_blue": ["light_blue"],
}


class TeamAssigner:
    """Conservative team assignment from torso color aggregated by tracklet.

    Two labels are exported: a tracklet-level team majority and a per-frame
    frame_team. The second one is intentionally local and is later used to catch
    ID switches when a tracklet suddenly looks like the opponent.
    """

    def __init__(
        self,
        max_seed_frames=12,
        min_seed_colors=8,
        min_cluster_separation=30.0,
        min_classification_margin=12.0,
        min_tracklet_colors=3,
        color_ranges_by_team=None,
        roster_color_min_fraction=0.16,
        roster_color_min_margin=0.04,
        prefer_roster_palette=True,
        trusted_palette_min_fraction=None,
        trusted_palette_min_margin=None,
        trusted_palette_min_samples=None,
        kmeans_random_states=None,
    ):
        self.max_seed_frames = int(max_seed_frames)
        self.min_seed_colors = int(min_seed_colors)
        self.min_cluster_separation = float(min_cluster_separation)
        self.min_classification_margin = float(min_classification_margin)
        self.min_tracklet_colors = int(min_tracklet_colors)
        self.color_ranges_by_team = normalize_team_ranges_by_team(color_ranges_by_team)
        self.prefer_roster_palette = bool(prefer_roster_palette)
        self.roster_color_min_fraction = float(
            trusted_palette_min_fraction
            if trusted_palette_min_fraction is not None
            else roster_color_min_fraction
        )
        self.roster_color_min_margin = float(
            trusted_palette_min_margin
            if trusted_palette_min_margin is not None
            else roster_color_min_margin
        )
        self.trusted_palette_min_samples = (
            int(trusted_palette_min_samples) if trusted_palette_min_samples is not None else None
        )
        self.kmeans_random_states = normalize_random_states(kmeans_random_states)
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
        # Roster colours are trusted when they produce a clear palette match.
        # KMeans remains available as a fallback for videos/teams where the
        # roster does not describe the kit or the crop is too ambiguous.
        cluster_assignments = self._assign_tracklets(frames, tracks)
        roster_assignments = (
            self._assign_tracklets_by_roster_palette(frames, tracks)
            if self.prefer_roster_palette
            else {}
        )
        assignments = merge_assignments(roster_assignments, cluster_assignments)
        frame_assignments = self._assign_frames(frames, tracks)
        self._apply(assignments, frame_assignments, tracks)
        return assignments

    def _fit(self, colors):
        if len(colors) < self.min_seed_colors:
            return
        from sklearn.cluster import KMeans

        x = np.asarray(colors, dtype=np.float32)
        best_model = None
        best_separation = -1.0
        for random_state in self.kmeans_random_states:
            model = KMeans(n_clusters=2, random_state=random_state, n_init=10).fit(x)
            centers = model.cluster_centers_
            separation = float(np.linalg.norm(centers[0] - centers[1]))
            if separation > best_separation:
                best_model = model
                best_separation = separation

        if best_model is None or best_separation < self.min_cluster_separation:
            # If the first frames do not separate into two kit colours, it is
            # safer to leave team unknown than to build a noisy classifier.
            return
        self.kmeans = best_model
        self.team_colors = {
            1: tuple(int(v) for v in self.kmeans.cluster_centers_[0]),
            2: tuple(int(v) for v in self.kmeans.cluster_centers_[1]),
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

    def _assign_tracklets_by_roster_palette(self, frames, tracks):
        if not self.color_ranges_by_team:
            return {}
        samples_by_tracklet = defaultdict(list)
        for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
            frame = frames[frame_num]
            for raw_id, track in frame_tracks.items():
                display_id = int(track.get("display_track_id", raw_id))
                samples_by_tracklet[display_id].append(
                    classify_team_palette(
                        frame,
                        track["bbox"],
                        self.color_ranges_by_team,
                        self.roster_color_min_fraction,
                        self.roster_color_min_margin,
                    )
                )
        return {
            display_id: summarize_team_palette_samples(samples, self.min_tracklet_colors)
            for display_id, samples in samples_by_tracklet.items()
        }

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
                # Low-margin colours are ignored rather than forced into a team.
                # This matters for referees, goalkeepers and heavy occlusions.
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
                # Per-frame palette evidence is intentionally local. A sudden
                # team flip inside one display_track_id is later treated as an
                # identity-switch signal by the constraint stage.
                roster_assignment = classify_team_palette(
                    frame,
                    track["bbox"],
                    self.color_ranges_by_team,
                    self.roster_color_min_fraction,
                    self.roster_color_min_margin,
                )
                if roster_assignment.get("team") is not None:
                    frame_row[int(raw_id)] = roster_assignment
                    continue
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
            "source": "kmeans",
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
                # frame_team is not used as the main team label. It is a local
                # consistency check consumed later by identity constraints.
                track["frame_team_confidence"] = float(frame_assignment.get("confidence", 0.0))
                track["frame_team_margin"] = float(frame_assignment.get("margin", 0.0))
                track["frame_team_evidence"] = frame_assignment

    @staticmethod
    def player_color(frame, bbox):
        import cv2

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


def merge_assignments(primary, fallback):
    """Prefer roster-palette teams only when they produce a valid decision."""
    merged = dict(fallback or {})
    for display_id, assignment in (primary or {}).items():
        if assignment.get("team") is not None:
            merged[display_id] = assignment
        elif display_id not in merged:
            merged[display_id] = assignment
    return merged


def normalize_random_states(values):
    """Normalize configurable K-Means seeds into a deterministic integer list."""
    if values in (None, "", []):
        return [0, 42, 7, 13]
    if isinstance(values, int):
        return [int(values)]
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    states = [int(value) for value in values]
    return states or [0, 42, 7, 13]


def classify_team_palette(frame, bbox, color_ranges_by_team, min_fraction=0.16, min_margin=0.04):
    """Classify one detection against roster-provided team kit palettes."""
    import cv2

    if not color_ranges_by_team:
        return {"team": None, "confidence": 0.0, "source": "roster_color", "margin": 0.0, "scores": {}}
    crop = torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return {"team": None, "confidence": 0.0, "source": "roster_color", "margin": 0.0, "scores": {}}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scores = {}
    color_scores = {}
    for team_id, ranges_by_name in color_ranges_by_team.items():
        team_total = 0.0
        team_color_scores = {}
        for name, ranges in ranges_by_name.items():
            score = palette_fraction(hsv, ranges)
            team_color_scores[name] = float(score)
            team_total += float(score)
        # Multi-colour kits such as Inter's black/blue stripes are represented
        # as several colour ranges for the same team. Summing their fractions
        # lets either stripe contribute while capping the final evidence.
        scores[int(team_id)] = min(1.0, team_total)
        color_scores[int(team_id)] = team_color_scores
    if not scores:
        return {"team": None, "confidence": 0.0, "source": "roster_color", "margin": 0.0, "scores": {}}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    team, score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = float(score - runner_up)
    if score < float(min_fraction) or margin < float(min_margin):
        team = None
        confidence = 0.0
    else:
        confidence = min(1.0, score)
    return {
        "team": int(team) if team is not None else None,
        "confidence": float(confidence),
        "source": "roster_color",
        "margin": margin,
        "scores": {int(key): float(value) for key, value in sorted(scores.items())},
        "color_scores": color_scores,
    }


def summarize_team_palette_samples(samples, min_samples):
    """Aggregate per-frame roster-colour votes into one tracklet decision."""
    if len(samples) < int(min_samples):
        return {
            "team": None,
            "confidence": 0.0,
            "source": "roster_color",
            "num_colors": len(samples),
            "reason": "not_enough_samples",
        }
    votes = defaultdict(int)
    by_team_scores = defaultdict(list)
    margins = []
    for sample in samples:
        team = sample.get("team")
        if team is None:
            continue
        team = int(team)
        votes[team] += 1
        by_team_scores[team].append(float(sample.get("confidence", 0.0) or 0.0))
        margins.append(float(sample.get("margin", 0.0) or 0.0))
    if not votes:
        return {
            "team": None,
            "confidence": 0.0,
            "source": "roster_color",
            "num_colors": len(samples),
            "votes": {},
        }
    team, count = max(votes.items(), key=lambda item: item[1])
    return {
        "team": int(team),
        "confidence": float(count / max(1, len(samples))),
        "source": "roster_color",
        "num_colors": len(samples),
        "votes": {int(key): int(value) for key, value in sorted(votes.items())},
        "mean_score": float(np.mean(by_team_scores[team])) if by_team_scores[team] else 0.0,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
    }


def team_color_ranges_by_team_from_roster(roster):
    """Extract outfield kit colours from player metadata, excluding GK/referee."""
    ranges_by_team = {}
    ignored_roles = {"goalkeeper", "keeper", "gk", "referee", "official", "match_official"}
    for player in roster or []:
        team_id = player.get("team_id")
        if team_id is None:
            continue
        role = str(player.get("role") or "").lower()
        if role in ignored_roles:
            continue
        metadata = player.get("metadata", {}) or {}
        color = (
            metadata.get("team_kit_color")
            or metadata.get("team_colors")
            or metadata.get("kit_colors")
            or metadata.get("kit_color")
            or metadata.get("uniform_color")
            or metadata.get("shirt_color")
            or metadata.get("color")
        )
        team_id = int(team_id)
        colors = []
        if color is not None:
            colors.append(color)
        kit_hint = metadata.get("kit_hint") or {}
        if isinstance(kit_hint, dict):
            shirt = kit_hint.get("shirt") or kit_hint.get("shirt_color") or kit_hint.get("primary")
            if shirt:
                colors.append(shirt)
        for item in colors:
            ranges_by_team.setdefault(team_id, {}).update(team_color_to_ranges(item, team_id=team_id))
    return ranges_by_team


def normalize_team_ranges_by_team(color_ranges_by_team):
    normalized = {}
    for team_id, ranges in (color_ranges_by_team or {}).items():
        team_id = int(team_id)
        normalized[team_id] = {}
        for name, value in (ranges or {}).items():
            if is_hsv_rule_list(value):
                normalized[team_id][str(name)] = list(value)
            elif isinstance(value, str) or isinstance(value, (list, tuple)):
                normalized[team_id].update(team_color_to_ranges(value, team_id=team_id, name=str(name)))
            else:
                normalized[team_id][str(name)] = list(value)
    return {team: ranges for team, ranges in normalized.items() if ranges}


def is_hsv_rule_list(value):
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def team_color_to_ranges(color, team_id=None, name=None):
    """Normalize a roster colour value into HSV rules used by palette_fraction."""
    ranges = {}
    for token in expand_team_color_tokens(color):
        range_name = name or f"team{team_id}_kit_{token}" if team_id is not None else f"kit_{token}"
        if token in TEAM_COLOR_RANGES:
            ranges[range_name] = TEAM_COLOR_RANGES[token]
            continue
        ranges.update(color_to_ranges(token, name=range_name))
    return ranges


def expand_team_color_tokens(color):
    """Accept simple names, aliases, lists and composite strings like black_blue."""
    if color is None:
        return []
    if isinstance(color, (list, tuple)):
        tokens = []
        for item in color:
            tokens.extend(expand_team_color_tokens(item))
        return tokens
    text = str(color).strip().lower()
    if not text:
        return []
    normalized = (
        text.replace("-", "_")
        .replace("/", "_")
        .replace("+", "_")
        .replace("&", "_")
        .replace(" ", "_")
    )
    if normalized in TEAM_COLOR_ALIASES:
        return TEAM_COLOR_ALIASES[normalized]
    if normalized in TEAM_COLOR_RANGES:
        return [normalized]
    if normalized.startswith("#") or len(normalized) == 6:
        return [normalized]
    parts = [part for part in normalized.split("_") if part and part not in {"and", "e"}]
    if len(parts) > 1:
        return parts
    return [normalized]
