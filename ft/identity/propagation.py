from collections import Counter, defaultdict

from ft.identity.identity_graph import IdentityGraphBuilder, TrackletNode, to_int


EMPTY = {"", "None", "unknown", None}


class IdentityPropagationEngine:
    """Propagate strong post-Hungarian identities across compatible tracklets."""

    def __init__(
        self,
        roster,
        min_composite_score=0.40,
        min_score_margin=0.08,
        max_hops=1,
        allow_propagated_sources=False,
        min_source_confidence=0.55,
        conflict_buffer=0,
        allow_partial_conflict_frames=False,
        min_partial_frames=20,
        min_partial_fraction=0.25,
        propagate_goalkeepers=True,
        **graph_kwargs,
    ):
        self.roster = list(roster or [])
        self.min_composite_score = float(min_composite_score)
        self.min_score_margin = float(min_score_margin)
        self.max_hops = max(1, int(max_hops or 1))
        self.allow_propagated_sources = bool(allow_propagated_sources)
        self.min_source_confidence = float(min_source_confidence)
        self.conflict_buffer = max(0, int(conflict_buffer or 0))
        self.allow_partial_conflict_frames = bool(allow_partial_conflict_frames)
        self.min_partial_frames = max(1, int(min_partial_frames or 1))
        self.min_partial_fraction = float(min_partial_fraction)
        self.propagate_goalkeepers = bool(propagate_goalkeepers)
        self.roster_by_player_id = {str(player["player_id"]): player for player in self.roster}
        self.graph_builder = IdentityGraphBuilder(
            min_composite_score=self.min_composite_score,
            **graph_kwargs,
        )

    def apply(self, tracks, summaries, assignments):
        """Apply propagation to tracks and return audit diagnostics."""
        nodes = build_nodes(tracks, summaries, assignments)
        diagnostics = {
            "enabled": True,
            "status": "ok",
            "total_nodes": len(nodes),
            "assigned_nodes": sum(1 for node in nodes if node.is_assigned),
            "unknown_nodes": sum(1 for node in nodes if not node.is_assigned),
            "total_propagated": 0,
            "hops": {},
            "graph": {},
            "propagations": [],
            "rejected_propagations": [],
        }
        if not self.roster_by_player_id:
            diagnostics["status"] = "missing_roster"
            return diagnostics
        if not nodes:
            diagnostics["status"] = "no_nodes"
            return diagnostics

        propagated_assignments = {}
        for hop in range(self.max_hops):
            edges = self.graph_builder.build(
                nodes,
                allow_propagated_sources=self.allow_propagated_sources,
            )
            diagnostics["graph"][f"hop_{hop + 1}"] = self.graph_builder.diagnostics()
            count = self._propagate_hop(
                tracks,
                nodes,
                edges,
                propagated_assignments,
                diagnostics,
                propagation_depth=hop + 1,
            )
            diagnostics["hops"][f"hop_{hop + 1}"] = int(count)
            diagnostics["total_propagated"] += int(count)
            if count == 0:
                break
        return diagnostics

    def _propagate_hop(self, tracks, nodes, edges, propagated_assignments, diagnostics, propagation_depth):
        edges_by_target = defaultdict(list)
        for edge in edges:
            edges_by_target[int(edge.target_id)].append(edge)
        node_by_id = {int(node.display_track_id): node for node in nodes}
        applied = 0

        for target_id, target_edges in sorted(edges_by_target.items()):
            target = node_by_id.get(target_id)
            if target is None or target.is_assigned:
                continue
            target_edges.sort(key=lambda edge: edge.composite_score, reverse=True)
            best = target_edges[0]
            second_score = target_edges[1].composite_score if len(target_edges) > 1 else 0.0
            if best.composite_score - second_score < self.min_score_margin:
                diagnostics["rejected_propagations"].append(
                    {
                        "reason": "ambiguous_best_edge",
                        "target_display_id": int(target_id),
                        "best_score": float(best.composite_score),
                        "second_score": float(second_score),
                        "min_score_margin": float(self.min_score_margin),
                    }
                )
                continue

            source = node_by_id.get(int(best.source_id))
            if source is None or not source.is_assigned:
                continue
            if float(source.identity_confidence or 0.0) < self.min_source_confidence:
                diagnostics["rejected_propagations"].append(
                    {
                        "reason": "source_confidence_too_low",
                        "source_display_id": int(best.source_id),
                        "target_display_id": int(target_id),
                        "source_confidence": float(source.identity_confidence or 0.0),
                        "min_source_confidence": float(self.min_source_confidence),
                    }
                )
                continue

            player = self.roster_by_player_id.get(str(source.player_id))
            if player is None:
                diagnostics["rejected_propagations"].append(
                    {
                        "reason": "source_not_in_roster",
                        "source_display_id": int(best.source_id),
                        "target_display_id": int(target_id),
                        "player_id": source.player_id,
                    }
                )
                continue
            if not self.propagate_goalkeepers and is_goalkeeper_player(player):
                diagnostics["rejected_propagations"].append(
                    {
                        "reason": "goalkeeper_source_propagation_disabled",
                        "source_display_id": int(best.source_id),
                        "target_display_id": int(target_id),
                        "player_id": player["player_id"],
                    }
                )
                continue

            valid_frames, conflicts, target_frames = valid_frames_for_assignment(
                tracks,
                target_id,
                player,
                buffer=self.conflict_buffer,
            )
            partial_fraction = len(valid_frames) / max(1, len(target_frames))
            if conflicts and not self.allow_partial_conflict_frames:
                diagnostics["rejected_propagations"].append(
                    {
                        "reason": "frame_conflict",
                        "source_display_id": int(best.source_id),
                        "target_display_id": int(target_id),
                        "player_id": player["player_id"],
                        "target_frames": int(len(target_frames)),
                        "valid_frames": int(len(valid_frames)),
                        "conflict_frames": int(len({item["frame"] for item in conflicts})),
                        "conflict_sample": conflicts[:20],
                    }
                )
                continue
            if len(valid_frames) < self.min_partial_frames or partial_fraction < self.min_partial_fraction:
                diagnostics["rejected_propagations"].append(
                    {
                        "reason": "insufficient_non_conflicting_frames",
                        "source_display_id": int(best.source_id),
                        "target_display_id": int(target_id),
                        "player_id": player["player_id"],
                        "target_frames": int(len(target_frames)),
                        "valid_frames": int(len(valid_frames)),
                        "min_partial_frames": int(self.min_partial_frames),
                        "partial_fraction": float(partial_fraction),
                        "min_partial_fraction": float(self.min_partial_fraction),
                        "conflict_frames": int(len({item["frame"] for item in conflicts})),
                        "conflict_sample": conflicts[:20],
                    }
                )
                continue

            assignment = build_assignment(player, source, target, best, propagation_depth)
            apply_assignment_to_tracks(tracks, target_id, assignment, valid_frames=valid_frames)
            apply_assignment_to_node(target, assignment)
            propagated_assignments[target_id] = assignment
            diagnostics["propagations"].append(
                {
                    **best.diagnostics(),
                    "player_id": player["player_id"],
                    "player_name": player.get("name", player["player_id"]),
                    "propagation_depth": int(propagation_depth),
                    "source_identity_confidence": float(source.identity_confidence or 0.0),
                    "target_num_frames": int(target.num_frames or 0),
                    "applied_frames": int(len(valid_frames)),
                    "target_frames": int(len(target_frames)),
                    "conflict_frames": int(len({item["frame"] for item in conflicts})),
                    "partial_fraction": float(partial_fraction),
                    "second_best_score": float(second_score),
                    "assignment_confidence": float(assignment["confidence"]),
                }
            )
            applied += 1
        return applied


def build_nodes(tracks, summaries, assignments):
    """Build graph nodes from Hungarian summaries plus current track state."""
    positions = first_last_positions_by_display(tracks)
    nodes = []
    for summary in summaries:
        track_id = int(summary["track_id"])
        assignment = assignments.get(track_id, {})
        player_id = assignment.get("player_id")
        if player_id in EMPTY:
            player_id = None
        display_id = int(summary.get("display_track_id") or track_id)
        first_position, last_position = positions.get(display_id, (None, None))
        nodes.append(
            TrackletNode(
                display_track_id=track_id,
                player_id=player_id,
                player_name=assignment.get("player_name"),
                identity_confidence=float(assignment.get("confidence") or 0.0),
                team_id=to_int(summary.get("team_id")),
                mean_team_confidence=float(summary.get("mean_team_confidence") or 0.0),
                jersey_number=to_int(summary.get("jersey_number")),
                jersey_votes=int(summary.get("jersey_votes") or 0),
                jersey_confidence=float(summary.get("jersey_confidence") or 0.0),
                jersey_head_confidence=float(summary.get("jersey_head_confidence") or 0.0),
                jersey_winner_margin=float(summary.get("jersey_winner_margin") or 0.0),
                jersey_distribution=summary.get("jersey_distribution") or [],
                raw_jersey_distribution=summary.get("raw_jersey_distribution") or [],
                start_frame=int(summary.get("start_frame") or 0),
                end_frame=int(summary.get("end_frame") or 0),
                num_frames=int(summary.get("num_frames") or 0),
                mean_pitch_position=summary.get("mean_pitch_position"),
                first_position=first_position,
                last_position=last_position,
                visual_embedding=summary.get("visual_embedding"),
                mean_crop_quality=float(summary.get("mean_crop_quality") or 0.0),
                role_detection=summary.get("role_detection"),
                semantic_group_id=to_int(summary.get("semantic_group_id")),
            )
        )
    return nodes


def first_last_positions_by_display(tracks):
    positions = {}
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id))
            position = track.get("position_pitch")
            if position is None:
                continue
            first_position, _last_position = positions.get(display_id, (None, None))
            positions[display_id] = (first_position or position, position)
    return positions


def valid_frames_for_assignment(tracks, target_display_id, player, buffer=0):
    """Return non-conflicting target frames plus conflict diagnostics."""
    player_id = str(player["player_id"])
    team_id = to_int(player.get("team_id"))
    jersey = to_int(player.get("jersey_number"))
    target_frames = []
    frame_rows = []
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        target_present = False
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id))
            if display_id == int(target_display_id):
                target_present = True
                continue
            frame_rows.append((frame_num, raw_id, track))
        if target_present:
            target_frames.append(frame_num)
    if buffer and len(target_frames) > buffer * 2:
        active_frames = set(target_frames[buffer:-buffer])
    else:
        active_frames = set(target_frames)

    conflicts = []
    conflict_frames = set()
    for frame_num, raw_id, track in frame_rows:
        if frame_num not in active_frames:
            continue
        if track.get("player_id") == player_id:
            conflict_frames.add(frame_num)
            conflicts.append(
                {
                    "frame": int(frame_num),
                    "display_track_id": int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id)),
                    "reason": "duplicate_player_id",
                    "player_id": player_id,
                }
            )
        if team_id is not None and jersey is not None:
            track_team = to_int(track.get("team"))
            track_jersey = to_int(track.get("jersey_number"))
            if track_team == team_id and track_jersey == jersey:
                conflict_frames.add(frame_num)
                conflicts.append(
                    {
                        "frame": int(frame_num),
                        "display_track_id": int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id)),
                        "reason": "duplicate_team_jersey",
                        "team_id": int(team_id),
                        "jersey_number": int(jersey),
                    }
                )
    valid_frames = active_frames - conflict_frames
    return valid_frames, conflicts, active_frames


def frame_conflict_for_assignment(tracks, target_display_id, player, buffer=0):
    """Return frame-level conflicts if a player/team+jersey is already occupied."""
    _valid_frames, conflicts, _target_frames = valid_frames_for_assignment(
        tracks,
        target_display_id,
        player,
        buffer=buffer,
    )
    return conflicts


def build_assignment(player, source, target, edge, propagation_depth):
    """Create a propagated assignment record to store on every target frame."""
    confidence = edge.composite_score * min(1.0, float(source.identity_confidence or 0.0)) * 0.85
    return {
        "player_id": player["player_id"],
        "player_name": player.get("name", player["player_id"]),
        "team_id": target.team_id if target.team_id is not None else player.get("team_id"),
        "jersey_number": player.get("jersey_number") or target.jersey_number,
        "confidence": float(confidence),
        "evidence": {
            "status": "propagated",
            "propagation_source_display_id": int(source.display_track_id),
            "propagation_depth": int(propagation_depth),
            "composite_score": float(edge.composite_score),
            "temporal_gap": int(edge.temporal_gap),
            "spatial_distance": edge.spatial_distance,
            "visual_similarity": edge.visual_similarity,
            "cut_bridge": bool(edge.cut_bridge),
            "cut_frame": edge.cut_frame,
            "jersey_match": bool(edge.jersey_match),
            "team_match": bool(edge.team_match),
            "source_identity_confidence": float(source.identity_confidence or 0.0),
            "source_player_id": source.player_id,
        },
    }


def apply_assignment_to_tracks(tracks, target_display_id, assignment, valid_frames=None):
    valid_frames = set(valid_frames) if valid_frames is not None else None
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        if valid_frames is not None and frame_num not in valid_frames:
            continue
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("identity_tracklet_id") or track.get("display_track_id", raw_id))
            if display_id != int(target_display_id):
                continue
            if track.get("player_id") not in EMPTY:
                continue
            track["player_id"] = assignment["player_id"]
            track["player_name"] = assignment["player_name"]
            track["jersey_number"] = assignment["jersey_number"]
            track["identity_confidence"] = assignment["confidence"]
            track["identity_evidence"] = assignment["evidence"]
            track["jersey_evidence"] = {
                "status": "propagated_from_identity",
                "confidence": assignment["confidence"],
                "votes": 0,
                "source_player_id": assignment["player_id"],
            }


def apply_assignment_to_node(node, assignment):
    node.player_id = assignment["player_id"]
    node.player_name = assignment["player_name"]
    node.identity_confidence = assignment["confidence"]
    node.jersey_number = assignment["jersey_number"]
    node.propagation_depth = int(assignment["evidence"].get("propagation_depth") or 1)


def propagation_rows(diagnostics):
    """Flatten propagation diagnostics for CSV export."""
    rows = []
    for item in diagnostics.get("propagations", []):
        rows.append(
            {
                "source_display_id": item.get("source_display_id"),
                "target_display_id": item.get("target_display_id"),
                "player_id": item.get("player_id"),
                "player_name": item.get("player_name"),
                "assignment_confidence": item.get("assignment_confidence"),
                "composite_score": item.get("composite_score"),
                "team_score": item.get("team_score"),
                "jersey_score": item.get("jersey_score"),
                "temporal_score": item.get("temporal_score"),
                "spatial_score": item.get("spatial_score"),
                "appearance_score": item.get("appearance_score"),
                "temporal_gap": item.get("temporal_gap"),
                "spatial_distance": item.get("spatial_distance"),
                "visual_similarity": item.get("visual_similarity"),
                "cut_bridge": item.get("cut_bridge"),
                "cut_frame": item.get("cut_frame"),
                "jersey_match": item.get("jersey_match"),
                "team_match": item.get("team_match"),
                "propagation_depth": item.get("propagation_depth"),
                "target_num_frames": item.get("target_num_frames"),
            }
        )
    return rows


def summarize_propagated_players(diagnostics):
    counts = Counter(item.get("player_id") for item in diagnostics.get("propagations", []))
    return [{"player_id": player_id, "propagations": int(count)} for player_id, count in counts.most_common()]


def is_goalkeeper_player(player):
    """Return whether a roster entry describes a goalkeeper identity."""
    role = str((player or {}).get("role") or "").lower()
    return role in {"goalkeeper", "keeper", "gk"}
