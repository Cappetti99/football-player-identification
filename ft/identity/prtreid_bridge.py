from collections import Counter, defaultdict

from ft.features.visual import cosine_similarity, mean_embedding


class PRTReIDIdentityBridge:
    """Propose or apply conservative identity links from reliable-jersey anchors."""

    def __init__(
        self,
        apply=False,
        min_samples=3,
        min_prototype_consistency=0.90,
        min_source_confidence=0.75,
        require_team_match=True,
        min_team_confidence=0.55,
        max_segment_gap=1,
        max_gap=60,
        min_similarity=1.0,
        min_margin=1.0,
        mutual_nearest=True,
        max_diagnostic_records=5000,
        **_unused,
    ):
        self.apply_enabled = bool(apply)
        self.min_samples = int(min_samples)
        self.min_prototype_consistency = float(min_prototype_consistency)
        self.min_source_confidence = float(min_source_confidence)
        self.require_team_match = bool(require_team_match)
        self.min_team_confidence = float(min_team_confidence)
        self.max_segment_gap = int(max_segment_gap)
        self.max_gap = int(max_gap)
        self.min_similarity = float(min_similarity)
        self.min_margin = float(min_margin)
        self.mutual_nearest = bool(mutual_nearest)
        self.max_diagnostic_records = int(max_diagnostic_records)

    def apply(self, tracks, features=None):
        summaries = summarize_identity_tracklets(tracks, features=features)
        anchors = [row for row in summaries.values() if is_anchor(row, self.min_source_confidence)]
        targets = [row for row in summaries.values() if row["all_rows_unknown"]]
        candidates = []
        rejected = []
        for anchor in anchors:
            for target in targets:
                candidate, reason = self._candidate(anchor, target)
                if candidate:
                    candidates.append(candidate)
                elif reason and len(rejected) < self.max_diagnostic_records:
                    rejected.append(reason)
        annotate_ranks(candidates)
        proposed = []
        for candidate in sorted(candidates, key=candidate_sort_key):
            reason = proposal_rejection(candidate, self)
            row = dict(candidate)
            if reason:
                row["reason"] = reason
                if len(rejected) < self.max_diagnostic_records:
                    rejected.append(row)
                continue
            row["status"] = "proposed"
            proposed.append(row)

        applied = []
        applied_targets = set()
        applied_rows = 0
        if self.apply_enabled:
            for proposal in proposed:
                target_id = int(proposal["target_display_track_id"])
                if target_id in applied_targets:
                    continue
                applied_targets.add(target_id)
                rows_changed = apply_proposal(tracks, proposal)
                applied_rows += rows_changed
                row = dict(proposal)
                row["status"] = "applied"
                row["applied_rows"] = int(rows_changed)
                applied.append(row)

        return {
            "enabled": True,
            "apply": self.apply_enabled,
            "status": "ok",
            "anchors": len(anchors),
            "unknown_targets": len(targets),
            "candidates": candidates[: self.max_diagnostic_records],
            "proposed_links": proposed[: self.max_diagnostic_records],
            "applied_links": applied,
            "applied_rows": int(applied_rows),
            "rejected_links": rejected,
            "rejection_counts": dict(sorted(Counter(row["reason"] for row in rejected).items())),
            "settings": self.settings(),
        }

    def _candidate(self, anchor, target):
        base = pair_payload(anchor, target)
        if anchor["frames"] & target["frames"]:
            base["reason"] = "overlap"
            return None, base
        if not prototype_eligible(anchor, self) or not prototype_eligible(target, self):
            base["reason"] = "prototype"
            return None, base
        if self.require_team_match:
            if anchor["team_id"] is None or target["team_id"] is None or anchor["team_id"] != target["team_id"]:
                base["reason"] = "team"
                return None, base
            if min(anchor["team_confidence"], target["team_confidence"]) < self.min_team_confidence:
                base["reason"] = "team_confidence"
                return None, base
        segment_gap = segment_distance(anchor, target)
        if segment_gap is None or segment_gap <= 0 or segment_gap > self.max_segment_gap:
            base.update({"reason": "segment_gap", "segment_gap": segment_gap})
            return None, base
        gap = temporal_gap(anchor, target)
        if gap <= 0 or gap > self.max_gap:
            base.update({"reason": "gap", "gap": gap, "segment_gap": segment_gap})
            return None, base
        similarity = cosine_similarity(anchor["visual_embedding"], target["visual_embedding"])
        if similarity is None:
            base["reason"] = "missing_similarity"
            return None, base
        base.update({
            "gap": int(gap),
            "segment_gap": int(segment_gap),
            "visual_similarity": float(similarity),
            "source_player_id": anchor["player_id"],
            "source_player_name": anchor["player_name"],
            "source_identity_confidence": anchor["identity_confidence"],
        })
        return base, None

    def settings(self):
        return {
            "min_samples": self.min_samples,
            "min_prototype_consistency": self.min_prototype_consistency,
            "min_source_confidence": self.min_source_confidence,
            "require_team_match": self.require_team_match,
            "min_team_confidence": self.min_team_confidence,
            "max_segment_gap": self.max_segment_gap,
            "max_gap": self.max_gap,
            "min_similarity": self.min_similarity,
            "min_margin": self.min_margin,
            "mutual_nearest": self.mutual_nearest,
        }


def summarize_identity_tracklets(tracks, features=None):
    features_by_display = {
        int(row["display_track_id"]): row
        for row in (features or [])
        if row.get("display_track_id") is not None
    }
    grouped = defaultdict(list)
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        for raw_id, track in frame_tracks.items():
            grouped[int(track.get("display_track_id", raw_id))].append((frame_num, track))
    output = {}
    for display_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        tracks_only = [track for _frame, track in items]
        player_id = mode([track.get("player_id") for track in tracks_only if track.get("player_id") not in (None, "")]) or "unknown"
        player_ids = {str(track.get("player_id") or "unknown") for track in tracks_only}
        feature = features_by_display.get(int(display_id), {})
        output[display_id] = {
            "display_track_id": int(display_id),
            "start": int(items[0][0]),
            "end": int(items[-1][0]),
            "frames": {int(frame) for frame, _track in items},
            "player_id": str(player_id),
            "all_rows_assigned": len(player_ids) == 1 and "unknown" not in player_ids,
            "all_rows_unknown": player_ids == {"unknown"},
            "player_name": mode([track.get("player_name") for track in tracks_only]) or str(player_id),
            "identity_status": mode([track.get("identity_status") for track in tracks_only]),
            "identity_confidence": mean(track.get("identity_confidence") for track in tracks_only),
            "assignment_gate_reason": mode([gate_reason(track) for track in tracks_only if gate_reason(track)]),
            "team_id": mode([track.get("team") for track in tracks_only if track.get("team") is not None]),
            "team_confidence": mean(track.get("team_confidence") for track in tracks_only),
            "scene_segment_id": mode([track.get("scene_segment_id") for track in tracks_only if track.get("scene_segment_id") is not None]),
            "visual_embedding": feature.get("visual_embedding"),
            "reid_sample_count": int(feature.get("sample_count") or 0),
            "reid_prototype_consistency": feature.get("prototype_consistency"),
        }
    return output


def is_anchor(row, min_confidence):
    return (
        row["player_id"] != "unknown"
        and row["all_rows_assigned"]
        and row["identity_status"] == "assigned"
        and row["assignment_gate_reason"] == "reliable_jersey"
        and row["identity_confidence"] >= float(min_confidence)
    )


def prototype_eligible(row, bridge):
    consistency = row.get("reid_prototype_consistency")
    return (
        row.get("visual_embedding") is not None
        and row.get("reid_sample_count", 0) >= bridge.min_samples
        and consistency is not None
        and consistency >= bridge.min_prototype_consistency
    )


def gate_reason(track):
    evidence = track.get("identity_evidence") or {}
    gate = evidence.get("assignment_gate") or {} if isinstance(evidence, dict) else {}
    return gate.get("reason") if isinstance(gate, dict) else None


def annotate_ranks(candidates):
    by_anchor = defaultdict(list)
    by_target = defaultdict(list)
    for row in candidates:
        by_anchor[row["anchor_display_track_id"]].append(row)
        by_target[row["target_display_track_id"]].append(row)
    for rows, prefix in [(rows, "outgoing") for rows in by_anchor.values()]:
        rank_rows(rows, prefix)
    for rows, prefix in [(rows, "incoming") for rows in by_target.values()]:
        rank_rows(rows, prefix)
    for row in candidates:
        row["mutual_nearest"] = row.get("outgoing_rank") == 1 and row.get("incoming_rank") == 1
        row["similarity_margin"] = min(row.get("outgoing_margin", 1.0), row.get("incoming_margin", 1.0))


def rank_rows(rows, prefix):
    ordered = sorted(rows, key=lambda row: (-row["visual_similarity"], row["gap"]))
    second = ordered[1]["visual_similarity"] if len(ordered) > 1 else None
    for rank, row in enumerate(ordered, 1):
        row[f"{prefix}_rank"] = rank
        row[f"{prefix}_margin"] = float(row["visual_similarity"] - second) if rank == 1 and second is not None else (1.0 if rank == 1 else 0.0)


def proposal_rejection(row, bridge):
    if row["visual_similarity"] < bridge.min_similarity:
        return "similarity"
    if row.get("similarity_margin", 0.0) < bridge.min_margin:
        return "margin"
    if bridge.mutual_nearest and not row.get("mutual_nearest", False):
        return "mutual_nearest"
    return None


def apply_proposal(tracks, proposal):
    target_id = int(proposal["target_display_track_id"])
    changed = 0
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            if int(track.get("display_track_id", raw_id)) != target_id or track.get("player_id") not in (None, "", "unknown"):
                continue
            track["player_id"] = proposal["source_player_id"]
            track["player_name"] = proposal["source_player_name"]
            track["identity_confidence"] = proposal["source_identity_confidence"]
            track["identity_status"] = "prtreid_bridge"
            track["identity_sources"] = {"prtreid_identity_bridge": True}
            track["identity_risk_flags"] = ["prtreid_identity_bridge"]
            track["identity_evidence"] = {"status": "prtreid_bridge", "bridge": proposal}
            changed += 1
    return changed


def pair_payload(anchor, target):
    return {
        "anchor_display_track_id": int(anchor["display_track_id"]),
        "target_display_track_id": int(target["display_track_id"]),
        "team_id": anchor.get("team_id"),
    }


def temporal_gap(left, right):
    if left["end"] < right["start"]:
        return int(right["start"] - left["end"])
    if right["end"] < left["start"]:
        return int(left["start"] - right["end"])
    return 0


def segment_distance(left, right):
    if left.get("scene_segment_id") is None or right.get("scene_segment_id") is None:
        return None
    return abs(int(left["scene_segment_id"]) - int(right["scene_segment_id"]))


def candidate_sort_key(row):
    return (-row["visual_similarity"], -row.get("similarity_margin", 0.0), row["gap"])


def mode(values):
    values = [value for value in values if value not in (None, "")]
    return Counter(values).most_common(1)[0][0] if values else None


def mean(values):
    values = [float(value) for value in values if value not in (None, "")]
    return sum(values) / len(values) if values else 0.0
