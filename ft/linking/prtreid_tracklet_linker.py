from collections import Counter, defaultdict

from ft.features.visual import cosine_similarity, mean_embedding
from ft.utils.geometry import distance


class PRTReIDTrackletLinker:
    """Conservatively add display-tracklet merges using PRTReID prototypes."""

    def __init__(
        self,
        min_samples=3,
        min_prototype_consistency=0.90,
        require_team_match=True,
        min_team_confidence=0.55,
        same_scene=None,
        cross_scene=None,
        max_diagnostic_records=5000,
    ):
        self.min_samples = int(min_samples)
        self.min_prototype_consistency = float(min_prototype_consistency)
        self.require_team_match = bool(require_team_match)
        self.min_team_confidence = float(min_team_confidence)
        self.same_scene = normalized_policy(
            same_scene,
            defaults={
                "enabled": True,
                "max_gap": 180,
                "max_distance": 220.0,
                "min_similarity": 0.99,
                "min_margin": 0.01,
                "mutual_nearest": True,
            },
        )
        self.cross_scene = normalized_policy(
            cross_scene,
            defaults={
                "enabled": False,
                "max_segment_gap": 1,
                "max_gap": 60,
                "min_similarity": 0.995,
                "min_margin": 0.02,
                "mutual_nearest": True,
            },
        )
        self.max_diagnostic_records = int(max_diagnostic_records)
        self.diagnostics = {}

    def apply(self, tracks):
        summaries = summarize_display_tracklets(tracks)
        candidates = []
        rejected = []
        ordered = sorted(summaries.values(), key=lambda row: (row["start"], row["display_track_id"]))
        for current_index, current in enumerate(ordered):
            for previous in ordered[:current_index]:
                candidate, reason = self._candidate(previous, current)
                if candidate is not None:
                    candidates.append(candidate)
                elif reason is not None:
                    append_limited(rejected, reason, self.max_diagnostic_records)

        annotate_candidate_ranks(candidates)
        accepted = []
        union = OverlapSafeUnion(summaries)
        for candidate in sorted(candidates, key=candidate_sort_key):
            policy = self.same_scene if candidate["link_type"] == "same_scene" else self.cross_scene
            reason = candidate_rejection(candidate, policy)
            if reason:
                row = dict(candidate)
                row["reason"] = reason
                append_limited(rejected, row, self.max_diagnostic_records)
                continue
            if not union.union(candidate["from_display_track_id"], candidate["to_display_track_id"]):
                row = dict(candidate)
                row["reason"] = "cluster_overlap"
                append_limited(rejected, row, self.max_diagnostic_records)
                continue
            row = dict(candidate)
            row["status"] = "accepted"
            accepted.append(row)

        display_map = union.canonical_map()
        apply_display_map(tracks, display_map)
        rejection_counts = Counter(row.get("reason", "unknown") for row in rejected)
        self.diagnostics = {
            "enabled": True,
            "status": "ok",
            "num_input_display_tracklets": len(summaries),
            "num_output_display_tracklets": len(set(display_map.values())),
            "accepted_links": accepted,
            "candidates": candidates[: self.max_diagnostic_records],
            "rejected_links": rejected,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "settings": {
                "min_samples": self.min_samples,
                "min_prototype_consistency": self.min_prototype_consistency,
                "require_team_match": self.require_team_match,
                "min_team_confidence": self.min_team_confidence,
                "same_scene": self.same_scene,
                "cross_scene": self.cross_scene,
            },
        }
        return display_map

    def _candidate(self, previous, current):
        base = pair_payload(previous, current)
        if previous["frames"].intersection(current["frames"]):
            base["reason"] = "overlap"
            return None, base
        gap = int(current["start"] - previous["end"])
        if gap <= 0:
            base.update({"reason": "gap", "gap": gap})
            return None, base
        if not prototype_is_eligible(previous, self.min_samples, self.min_prototype_consistency):
            base["reason"] = "previous_prototype"
            return None, base
        if not prototype_is_eligible(current, self.min_samples, self.min_prototype_consistency):
            base["reason"] = "current_prototype"
            return None, base
        if not team_is_compatible(previous, current, self.require_team_match, self.min_team_confidence):
            base["reason"] = "team_gate"
            return None, base

        previous_segment = previous.get("scene_segment_id")
        current_segment = current.get("scene_segment_id")
        segment_gap = None
        if previous_segment is not None and current_segment is not None:
            segment_gap = int(current_segment) - int(previous_segment)
        same_segment = segment_gap in (None, 0)
        if same_segment:
            policy = self.same_scene
            link_type = "same_scene"
            if gap > int(policy["max_gap"]):
                base.update({"reason": "same_scene_gap", "gap": gap})
                return None, base
            spatial_distance = tracklet_distance(previous, current)
            if spatial_distance is None or spatial_distance > float(policy["max_distance"]):
                base.update({"reason": "same_scene_distance", "gap": gap, "distance": spatial_distance})
                return None, base
        else:
            policy = self.cross_scene
            link_type = "cross_scene"
            if segment_gap <= 0 or segment_gap > int(policy["max_segment_gap"]):
                base.update({"reason": "cross_scene_segment_gap", "segment_gap": segment_gap})
                return None, base
            if gap > int(policy["max_gap"]):
                base.update({"reason": "cross_scene_gap", "gap": gap, "segment_gap": segment_gap})
                return None, base
            spatial_distance = None

        similarity = cosine_similarity(previous.get("visual_embedding"), current.get("visual_embedding"))
        if similarity is None:
            base["reason"] = "missing_similarity"
            return None, base
        base.update(
            {
                "link_type": link_type,
                "gap": gap,
                "segment_gap": segment_gap,
                "distance": spatial_distance,
                "visual_similarity": float(similarity),
                "policy_enabled": bool(policy["enabled"]),
                "from_sample_count": int(previous["reid_sample_count"]),
                "to_sample_count": int(current["reid_sample_count"]),
                "from_prototype_consistency": previous.get("reid_prototype_consistency"),
                "to_prototype_consistency": current.get("reid_prototype_consistency"),
            }
        )
        return base, None


def summarize_display_tracklets(tracks):
    grouped = defaultdict(list)
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            grouped[display_id].append((int(frame_num), int(raw_id), track))
    summaries = {}
    for display_id, items in grouped.items():
        items.sort(key=lambda item: (item[0], item[1]))
        team_values = [item[2].get("team") for item in items if item[2].get("team") is not None]
        team_id = mode(team_values)
        embeddings = [
            item[2].get("prtreid_tracklet_embedding")
            for item in items
            if item[2].get("prtreid_tracklet_embedding") is not None
        ]
        sample_counts = [int(item[2].get("prtreid_tracklet_sample_count") or 0) for item in items]
        consistencies = [
            item[2].get("prtreid_tracklet_consistency")
            for item in items
            if item[2].get("prtreid_tracklet_consistency") is not None
        ]
        segments = [item[2].get("scene_segment_id") for item in items if item[2].get("scene_segment_id") is not None]
        summaries[display_id] = {
            "display_track_id": int(display_id),
            "start": int(items[0][0]),
            "end": int(items[-1][0]),
            "frames": {item[0] for item in items},
            "raw_track_ids": sorted({item[1] for item in items}),
            "first_position": items[0][2].get("position"),
            "last_position": items[-1][2].get("position"),
            "team_id": team_id,
            "team_confidence": mean(
                item[2].get("team_confidence") for item in items if item[2].get("team") == team_id
            ),
            "scene_segment_id": mode(segments),
            "visual_embedding": mean_embedding(embeddings),
            "reid_sample_count": max(sample_counts, default=0),
            "reid_prototype_consistency": mean(consistencies) if consistencies else None,
        }
    return summaries


def annotate_candidate_ranks(candidates):
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for row in candidates:
        key = row["link_type"]
        incoming[(key, row["to_display_track_id"])].append(row)
        outgoing[(key, row["from_display_track_id"])].append(row)
    for rows in incoming.values():
        annotate_rank_group(rows, "incoming")
    for rows in outgoing.values():
        annotate_rank_group(rows, "outgoing")
    for row in candidates:
        row["mutual_nearest"] = row.get("incoming_rank") == 1 and row.get("outgoing_rank") == 1
        row["similarity_margin"] = min(row.get("incoming_margin", 1.0), row.get("outgoing_margin", 1.0))


def annotate_rank_group(rows, prefix):
    ordered = sorted(rows, key=lambda row: (-row["visual_similarity"], row["gap"], row["from_display_track_id"], row["to_display_track_id"]))
    second = ordered[1]["visual_similarity"] if len(ordered) > 1 else None
    for rank, row in enumerate(ordered, start=1):
        row[f"{prefix}_rank"] = rank
        row[f"{prefix}_margin"] = float(row["visual_similarity"] - second) if rank == 1 and second is not None else (1.0 if rank == 1 else 0.0)


def candidate_rejection(candidate, policy):
    if not policy.get("enabled", False):
        return "policy_disabled"
    if candidate["visual_similarity"] < float(policy["min_similarity"]):
        return "similarity"
    if candidate.get("similarity_margin", 0.0) < float(policy["min_margin"]):
        return "margin"
    if policy.get("mutual_nearest", True) and not candidate.get("mutual_nearest", False):
        return "mutual_nearest"
    return None


class OverlapSafeUnion:
    def __init__(self, summaries):
        self.parent = {key: key for key in summaries}
        self.frames = {key: set(value["frames"]) for key, value in summaries.items()}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.frames[left_root].intersection(self.frames[right_root]):
            return False
        canonical = min(left_root, right_root)
        other = max(left_root, right_root)
        self.parent[other] = canonical
        self.frames[canonical].update(self.frames[other])
        return True

    def canonical_map(self):
        return {value: self.find(value) for value in self.parent}


def apply_display_map(tracks, display_map):
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            previous = int(track.get("display_track_id", raw_id))
            current = int(display_map.get(previous, previous))
            if current != previous:
                track["prtreid_link_previous_display_track_id"] = previous
            track["display_track_id"] = current


def prototype_is_eligible(summary, min_samples, min_consistency):
    consistency = summary.get("reid_prototype_consistency")
    return (
        summary.get("visual_embedding") is not None
        and int(summary.get("reid_sample_count") or 0) >= int(min_samples)
        and consistency is not None
        and float(consistency) >= float(min_consistency)
    )


def team_is_compatible(previous, current, require_match, min_confidence):
    if not require_match:
        return True
    if previous.get("team_id") is None or current.get("team_id") is None:
        return False
    return (
        int(previous["team_id"]) == int(current["team_id"])
        and float(previous.get("team_confidence") or 0.0) >= float(min_confidence)
        and float(current.get("team_confidence") or 0.0) >= float(min_confidence)
    )


def pair_payload(previous, current):
    return {
        "from_display_track_id": int(previous["display_track_id"]),
        "to_display_track_id": int(current["display_track_id"]),
        "from_start": int(previous["start"]),
        "from_end": int(previous["end"]),
        "to_start": int(current["start"]),
        "to_end": int(current["end"]),
        "from_scene_segment_id": previous.get("scene_segment_id"),
        "to_scene_segment_id": current.get("scene_segment_id"),
        "team_id": previous.get("team_id"),
    }


def tracklet_distance(previous, current):
    if previous.get("last_position") is None or current.get("first_position") is None:
        return None
    return float(distance(previous["last_position"], current["first_position"]))


def candidate_sort_key(row):
    return (
        0 if row["link_type"] == "same_scene" else 1,
        -float(row["visual_similarity"]),
        -float(row.get("similarity_margin", 0.0)),
        int(row["gap"]),
    )


def normalized_policy(value, defaults):
    output = dict(defaults)
    output.update(value or {})
    return output


def append_limited(rows, row, limit):
    if len(rows) < int(limit):
        rows.append(row)


def mode(values):
    values = list(values)
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0
