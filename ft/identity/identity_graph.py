from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from ft.features.visual import cosine_similarity


EMPTY = {"", "None", "unknown", None}


@dataclass
class TrackletNode:
    """Identity-propagation node built from one identity/display tracklet."""

    display_track_id: int
    player_id: str | None = None
    player_name: str | None = None
    identity_confidence: float = 0.0
    team_id: int | None = None
    mean_team_confidence: float = 0.0
    jersey_number: int | None = None
    jersey_votes: int = 0
    jersey_confidence: float = 0.0
    jersey_head_confidence: float = 0.0
    jersey_winner_margin: float = 0.0
    jersey_distribution: list = field(default_factory=list)
    raw_jersey_distribution: list = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0
    num_frames: int = 0
    mean_pitch_position: list | None = None
    first_position: list | None = None
    last_position: list | None = None
    visual_embedding: list | None = None
    mean_crop_quality: float = 0.0
    role_detection: str | None = None
    semantic_group_id: int | None = None
    propagation_depth: int = 0

    @property
    def is_assigned(self):
        return self.player_id not in EMPTY

    @property
    def is_referee(self):
        role = str(self.role_detection or "").lower()
        return role in {"referee", "referee_candidate"} or self.semantic_group_id == 5

    @property
    def is_goalkeeper(self):
        role = str(self.role_detection or "").lower()
        return role in {"goalkeeper", "keeper", "gk"} or self.semantic_group_id in {3, 4}


@dataclass
class CompatibilityEdge:
    """Compatibility evidence for propagating one source identity to a target."""

    source_id: int
    target_id: int
    team_score: float = 0.0
    jersey_score: float = 0.0
    temporal_score: float = 0.0
    spatial_score: float = 0.0
    appearance_score: float = 0.0
    team_match: bool = False
    jersey_match: bool = False
    temporal_gap: int = 0
    spatial_distance: float | None = None
    visual_similarity: float | None = None
    has_temporal_overlap: bool = False
    cut_bridge: bool = False
    cut_frame: int | None = None
    rejection_reason: str | None = None

    @property
    def composite_score(self):
        if self.cut_bridge:
            return max(
                0.0,
                (
                    0.45 * self.team_score
                    + 0.45 * self.jersey_score
                    + 0.10 * self.temporal_score
                ),
            )
        return max(
            0.0,
            (
                0.40 * self.team_score
                + 0.30 * self.jersey_score
                + 0.15 * self.temporal_score
                + 0.10 * self.spatial_score
                + 0.05 * self.appearance_score
            ),
        )

    def diagnostics(self):
        return {
            "source_display_id": int(self.source_id),
            "target_display_id": int(self.target_id),
            "composite_score": float(self.composite_score),
            "team_score": float(self.team_score),
            "jersey_score": float(self.jersey_score),
            "temporal_score": float(self.temporal_score),
            "spatial_score": float(self.spatial_score),
            "appearance_score": float(self.appearance_score),
            "team_match": bool(self.team_match),
            "jersey_match": bool(self.jersey_match),
            "temporal_gap": int(self.temporal_gap),
            "spatial_distance": self.spatial_distance,
            "visual_similarity": self.visual_similarity,
            "cut_bridge": bool(self.cut_bridge),
            "cut_frame": self.cut_frame,
            "rejection_reason": self.rejection_reason,
        }


class IdentityGraphBuilder:
    """Build viable source-target edges for post-Hungarian propagation."""

    def __init__(
        self,
        max_temporal_gap=300,
        max_spatial_distance=25.0,
        min_team_confidence=0.50,
        min_appearance_similarity=0.50,
        min_composite_score=0.40,
        require_team_match=True,
        require_jersey_match=False,
        block_goalkeeper_mismatch=True,
        require_jersey_or_strong_appearance=True,
        strong_appearance_similarity=0.72,
        allow_temporal_overlap=False,
        temporal_overlap_score=0.10,
        scene_cut_frames=None,
        cut_bridge_enabled=False,
        cut_bridge_max_gap=5,
        cut_bridge_min_jersey_confidence=0.20,
        cut_bridge_min_jersey_votes=3,
    ):
        self.max_temporal_gap = int(max_temporal_gap)
        self.max_spatial_distance = float(max_spatial_distance)
        self.min_team_confidence = float(min_team_confidence)
        self.min_appearance_similarity = float(min_appearance_similarity)
        self.min_composite_score = float(min_composite_score)
        self.require_team_match = bool(require_team_match)
        self.require_jersey_match = bool(require_jersey_match)
        self.block_goalkeeper_mismatch = bool(block_goalkeeper_mismatch)
        self.require_jersey_or_strong_appearance = bool(require_jersey_or_strong_appearance)
        self.strong_appearance_similarity = float(strong_appearance_similarity)
        self.allow_temporal_overlap = bool(allow_temporal_overlap)
        self.temporal_overlap_score = float(temporal_overlap_score)
        self.scene_cut_frames = sorted({int(frame) for frame in (scene_cut_frames or []) if int(frame) > 0})
        self.cut_bridge_enabled = bool(cut_bridge_enabled)
        self.cut_bridge_max_gap = int(cut_bridge_max_gap)
        self.cut_bridge_min_jersey_confidence = float(cut_bridge_min_jersey_confidence)
        self.cut_bridge_min_jersey_votes = int(cut_bridge_min_jersey_votes)
        self._diagnostics = {}

    def build(self, nodes, allow_propagated_sources=False):
        """Return viable compatibility edges from assigned sources to unknown targets."""
        rejected = Counter()
        rejected_examples = []
        accepted_examples = []
        candidate_pairs = 0
        sources = [
            node
            for node in nodes
            if node.is_assigned
            and not node.is_referee
            and (allow_propagated_sources or node.propagation_depth == 0)
        ]
        targets = [node for node in nodes if not node.is_assigned and not node.is_referee]
        if not sources or not targets:
            self._diagnostics = {
                "source_nodes": len(sources),
                "target_nodes": len(targets),
                "candidate_pairs": 0,
                "accepted_edges": 0,
                "rejected": {},
                "rejected_examples": [],
            }
            return []

        source_by_bucket = defaultdict(list)
        bucket_size = 50
        for source in sources:
            for frame in (source.start_frame, source.end_frame):
                source_by_bucket[int(frame) // bucket_size].append(source)

        edges = []
        for target in targets:
            target_bucket = int(target.start_frame) // bucket_size
            bucket_window = self.max_temporal_gap // bucket_size + 2
            candidate_sources = []
            seen = set()
            for bucket in range(max(0, target_bucket - bucket_window), target_bucket + bucket_window + 1):
                for source in source_by_bucket.get(bucket, []):
                    if source.display_track_id in seen:
                        continue
                    seen.add(source.display_track_id)
                    candidate_sources.append(source)
            for source in candidate_sources:
                candidate_pairs += 1
                edge = self.build_edge(source, target)
                if edge is None:
                    rejected["unknown"] += 1
                    continue
                if edge.rejection_reason:
                    rejected[edge.rejection_reason] += 1
                    if len(rejected_examples) < 50:
                        rejected_examples.append(edge.diagnostics())
                    continue
                if edge.composite_score < self.min_composite_score:
                    edge.rejection_reason = "below_min_composite_score"
                    rejected[edge.rejection_reason] += 1
                    if len(rejected_examples) < 50:
                        rejected_examples.append(edge.diagnostics())
                    continue
                if edge is not None and edge.composite_score >= self.min_composite_score:
                    edges.append(edge)
                    if len(accepted_examples) < 50:
                        accepted_examples.append(edge.diagnostics())
        self._diagnostics = {
            "source_nodes": len(sources),
            "target_nodes": len(targets),
            "candidate_pairs": int(candidate_pairs),
            "accepted_edges": len(edges),
            "rejected": {key: int(value) for key, value in rejected.most_common()},
            "rejected_examples": rejected_examples,
            "accepted_examples": accepted_examples,
        }
        return edges

    def diagnostics(self):
        """Return diagnostics from the most recent build call."""
        return dict(self._diagnostics)

    def build_edge(self, source, target):
        """Score one source-target pair and return None when it is impossible."""
        edge = CompatibilityEdge(source_id=source.display_track_id, target_id=target.display_track_id)

        edge.has_temporal_overlap = overlaps(source.start_frame, source.end_frame, target.start_frame, target.end_frame)
        if edge.has_temporal_overlap:
            if not self.allow_temporal_overlap:
                edge.rejection_reason = "temporal_overlap"
                return edge
            edge.temporal_gap = 0
            # Overlap is allowed only because the propagation stage can write
            # non-conflicting frames. It remains weak temporal evidence.
            edge.temporal_score = self.temporal_overlap_score
        else:
            gap = temporal_gap(source, target)
            edge.temporal_gap = gap
            if gap > self.max_temporal_gap:
                edge.rejection_reason = "temporal_gap"
                return edge
            edge.temporal_score = 1.0 - min(1.0, gap / max(1, self.max_temporal_gap))
            cut_frame = self._cut_frame_between(source, target, gap)
            if cut_frame is not None:
                edge.cut_bridge = True
                edge.cut_frame = int(cut_frame)
                edge.temporal_score = 1.0

        if self.block_goalkeeper_mismatch and source.is_goalkeeper != target.is_goalkeeper:
            edge.rejection_reason = "goalkeeper_role_mismatch"
            return edge
        if not self._score_team(edge, source, target):
            return edge
        if not self._score_jersey(edge, source, target):
            return edge
        if self.require_jersey_match and not edge.jersey_match:
            edge.rejection_reason = "missing_required_jersey_match"
            return edge
        if edge.cut_bridge:
            if not edge.team_match:
                edge.rejection_reason = "cut_bridge_missing_team_match"
                return edge
            if not edge.jersey_match:
                edge.rejection_reason = "cut_bridge_missing_jersey_match"
                return edge
            if float(target.jersey_confidence or 0.0) < self.cut_bridge_min_jersey_confidence:
                edge.rejection_reason = "cut_bridge_low_jersey_confidence"
                return edge
            if int(target.jersey_votes or 0) < self.cut_bridge_min_jersey_votes:
                edge.rejection_reason = "cut_bridge_low_jersey_votes"
                return edge
            edge.spatial_score = 0.0
            edge.appearance_score = 0.0
            return edge
        if not self._score_spatial(edge, source, target):
            return edge
        if not self._score_appearance(edge, source, target):
            return edge

        if self.require_jersey_or_strong_appearance and not edge.jersey_match:
            strong_appearance = edge.visual_similarity is not None and edge.visual_similarity >= self.strong_appearance_similarity
            if not strong_appearance:
                edge.rejection_reason = "missing_jersey_or_strong_appearance"
                return edge
        return edge

    def _cut_frame_between(self, source, target, gap):
        if not self.cut_bridge_enabled or not self.scene_cut_frames:
            return None
        if int(gap) > self.cut_bridge_max_gap:
            return None
        if source.end_frame < target.start_frame:
            lower = int(source.end_frame) + 1
            upper = int(target.start_frame)
        elif target.end_frame < source.start_frame:
            lower = int(target.end_frame) + 1
            upper = int(source.start_frame)
        else:
            return None
        for frame in self.scene_cut_frames:
            if lower <= frame <= upper:
                return int(frame)
        return None

    def _score_team(self, edge, source, target):
        source_team = to_int(source.team_id)
        target_team = to_int(target.team_id)
        if source_team is None or target_team is None:
            edge.team_score = 0.15
            if self.require_team_match:
                edge.rejection_reason = "missing_team"
                return False
            return True
        if source_team != target_team:
            edge.rejection_reason = "team_mismatch"
            return False
        edge.team_match = True
        confidence = (float(source.mean_team_confidence or 0.0) + float(target.mean_team_confidence or 0.0)) / 2.0
        edge.team_score = min(1.0, max(self.min_team_confidence, confidence))
        return True

    def _score_jersey(self, edge, source, target):
        expected = to_int(source.jersey_number) or player_jersey_from_id(source.player_id)
        observed = to_int(target.jersey_number)
        if expected is None:
            edge.jersey_score = 0.0
            return True
        if observed is None:
            edge.jersey_score = 0.0
            return True
        if observed == expected:
            edge.jersey_match = True
            edge.jersey_score = min(
                1.0,
                0.50 * float(target.jersey_confidence or 0.0)
                + 0.30 * min(1.0, int(target.jersey_votes or 0) / 10.0)
                + 0.20 * float(target.jersey_head_confidence or 0.0),
            )
            return True
        if int(target.jersey_votes or 0) >= 3 and float(target.jersey_confidence or 0.0) >= 0.25:
            edge.rejection_reason = "confirmed_jersey_mismatch"
            return False
        edge.jersey_score = -0.10
        return True

    def _score_spatial(self, edge, source, target):
        pos_a, pos_b = transition_positions(source, target)
        if pos_a is None or pos_b is None:
            edge.spatial_score = 0.30
            return True
        distance = float(np.linalg.norm(np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float)))
        edge.spatial_distance = distance
        max_plausible = self.max_spatial_distance + max(0, edge.temporal_gap) * 0.15
        if distance > max_plausible:
            edge.rejection_reason = "spatial_distance"
            return False
        edge.spatial_score = max(0.0, 1.0 - distance / max(1.0, self.max_spatial_distance))
        return True

    def _score_appearance(self, edge, source, target):
        similarity = cosine_similarity(source.visual_embedding, target.visual_embedding)
        if similarity is None:
            edge.appearance_score = 0.25
            return True
        edge.visual_similarity = similarity
        if similarity < self.min_appearance_similarity and not edge.jersey_match:
            edge.rejection_reason = "appearance_similarity"
            return False
        edge.appearance_score = max(0.0, similarity)
        return True


def player_jersey_from_id(player_id):
    """Extract jersey number from roster ids such as team1_09."""
    if player_id in EMPTY:
        return None
    parts = str(player_id).rsplit("_", 1)
    if len(parts) != 2:
        return None
    try:
        number = int(parts[1])
    except ValueError:
        return None
    return number if 1 <= number <= 99 else None


def overlaps(a_start, a_end, b_start, b_end):
    return int(a_start) <= int(b_end) and int(b_start) <= int(a_end)


def temporal_gap(source, target):
    if source.end_frame < target.start_frame:
        return int(target.start_frame - source.end_frame)
    return int(source.start_frame - target.end_frame)


def transition_positions(source, target):
    if source.end_frame < target.start_frame:
        return source.last_position or source.mean_pitch_position, target.first_position or target.mean_pitch_position
    return target.last_position or target.mean_pitch_position, source.first_position or source.mean_pitch_position


def to_int(value):
    if value in EMPTY:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
