"""Pure adapters from existing stage outputs to :class:`Evidence` rows.

Every function here is additive and read-only: it receives the exact structure a
stage already returns today and produces evidence rows from it. No stage is
modified, no field is renamed, and the legacy artifacts are untouched. This is
what makes step 1 of the migration risk-free -- the run must stay bit-identical.

The field names inside ``payload`` deliberately mirror the legacy keys so that
the decision layer can be ported in step 2 without a second translation.
"""

from __future__ import annotations

from ft.core.evidence import (
    Evidence,
    EvidenceKind,
    SubjectType,
    evidence_value,
    subject_id,
)


# Producer names are stable identifiers written into the artifact; the decision
# policy selects sources by these strings, so they must not drift.
PRODUCER_JERSEY_PRIMARY = "jersey_ocr_primary"
PRODUCER_JERSEY_SECONDARY = "jersey_ocr_secondary_audit"
PRODUCER_JERSEY_REGION_CTC = "jersey_region_ctc"
PRODUCER_TEAM_COLOR = "team_color"
PRODUCER_REFEREE_PALETTE = "referee_palette"
PRODUCER_TRACKLET_LINKER = "tracklet_linker"


# display_track_id restarts per track group: the players group and the referees
# group both contain an id 13, and they are different physical subjects. Every
# subject id is therefore namespaced by its group.
GROUP_PLAYERS = "players"
GROUP_REFEREES = "referees"


def display_subject(track_group, display_id, segment_index=None):
    """Subject id for a display track, optionally split by jersey segment.

    Segment semantics mirror ``ft.pipeline.identity_tracklet_id``; the group
    prefix is what keeps ``players:13`` and ``referees:13`` apart.
    ``tests/test_evidence_emitters.py`` pins both properties.
    """
    parts = [str(track_group), str(int(display_id))]
    if segment_index is not None:
        parts.append(str(int(segment_index)))
    return subject_id(":".join(parts))


def _display_id_from_key(track_id, assignment):
    """Resolve the display id the way ``assignment_display_id`` does."""
    if assignment and assignment.get("display_track_id") is not None:
        return int(assignment["display_track_id"])
    if isinstance(track_id, tuple):
        return int(track_id[0])
    return int(str(track_id).split(":")[0])


# jersey_number and confidence become the Evidence value/score; everything else
# is copied wholesale rather than through a fixed key list, so the payload is
# lossless by construction. A whitelist would silently drop fields added
# upstream (the roster filter, for one) and the round-trip through evidence
# would then alter the artifacts.
_JERSEY_PROMOTED_KEYS = ("jersey_number", "confidence")


def _jersey_payload(assignment):
    """Copy the legacy assignment verbatim, minus the promoted columns."""
    return {
        key: value
        for key, value in assignment.items()
        if key not in _JERSEY_PROMOTED_KEYS
    }


def jersey_assignment_evidence(
    assignments,
    config_hash,
    produced_by=PRODUCER_JERSEY_PRIMARY,
    model_sha256=None,
):
    """Evidence rows from a ``{track_id: voted}`` jersey assignment mapping."""
    rows = []
    for track_id, assignment in (assignments or {}).items():
        if not assignment:
            continue
        display_id = _display_id_from_key(track_id, assignment)
        segment_index = assignment.get("segment_index")
        payload = _jersey_payload(assignment)
        # Normalized, not str(track_id): a tuple key would otherwise serialize
        # as "(12, 0)" and never parse back.
        payload["legacy_key"] = subject_id(track_id)
        rows.append(
            Evidence(
                subject_type=SubjectType.IDENTITY_TRACKLET,
                subject_id=display_subject(GROUP_PLAYERS, display_id, segment_index),
                kind=EvidenceKind.JERSEY_NUMBER,
                value=evidence_value(assignment.get("jersey_number")),
                score=_float_or_none(assignment.get("confidence")),
                frame_start=_int_or_none(assignment.get("segment_start_frame")),
                frame_end=_int_or_none(assignment.get("segment_end_frame")),
                produced_by=produced_by,
                model_sha256=model_sha256,
                config_hash=config_hash,
                payload=payload,
            )
        )
    return rows


def jersey_assignments_from_evidence(rows):
    """Rebuild the legacy ``{track_id: voted}`` mapping from evidence rows.

    Only used to prove sufficiency: if the reconstruction feeds ``_apply_jersey``
    and yields byte-identical rows, then no jersey information lives solely in
    the mutated rows, and step 2 can port the decision layer safely.
    """
    assignments = {}
    for row in rows or ():
        if row.kind != EvidenceKind.JERSEY_NUMBER:
            continue
        payload = dict(row.payload or {})
        legacy_key = payload.pop("legacy_key", None)
        if legacy_key is None:
            continue
        assignment = {key: value for key, value in payload.items()}
        assignment["jersey_number"] = int(row.value) if row.value is not None else None
        assignment["confidence"] = row.score
        assignments[_legacy_key(legacy_key)] = assignment
    return assignments


def _legacy_key(text):
    if ":" in str(text):
        display, segment = str(text).split(":", 1)
        return (int(display), int(segment))
    return int(text)


def jersey_region_ctc_evidence(diagnostics, config_hash):
    """Track- and crop-level evidence from the audit-only CTC recognizer."""
    if not diagnostics or not diagnostics.get("enabled"):
        return []
    configuration = diagnostics.get("configuration") or {}
    ctc_sha = configuration.get("ctc_checkpoint_sha256")
    rows = []
    for key, proposal in (diagnostics.get("standalone_assignments") or {}).items():
        frames = proposal.get("frames") or []
        # The CTC auditor groups by (display_track_id, scene_segment_id), a
        # different segmentation from the jersey segment_index used by the
        # primary OCR. Downstream it is already consumed by plain display id,
        # so that stays the subject and the scene segment goes in the payload.
        rows.append(
            Evidence(
                subject_type=SubjectType.DISPLAY_TRACK,
                subject_id=display_subject(
                    GROUP_PLAYERS, proposal.get("display_track_id", str(key).split("#", 1)[0])
                ),
                kind=EvidenceKind.JERSEY_NUMBER,
                value=evidence_value(proposal.get("jersey_number")),
                score=_float_or_none(proposal.get("confidence")),
                frame_start=_int_or_none(min(frames)) if frames else None,
                frame_end=_int_or_none(max(frames)) if frames else None,
                produced_by=PRODUCER_JERSEY_REGION_CTC,
                model_sha256=ctc_sha,
                config_hash=config_hash,
                payload={
                    "display_track_id": proposal.get("display_track_id"),
                    "scene_segment_id": _scene_segment_from_key(key),
                    "winner_margin": proposal.get("winner_margin"),
                    "recognized_frames": proposal.get("recognized_frames"),
                    "top5": proposal.get("top5"),
                    "applied": proposal.get("applied", False),
                    "legacy_key": str(key),
                },
            )
        )
    rows.extend(_jersey_region_ctc_crop_evidence(diagnostics, config_hash, ctc_sha))
    return rows


def _jersey_region_ctc_crop_evidence(diagnostics, config_hash, ctc_sha):
    """Crop-level rows: this is where the per-crop provenance already lives."""
    rows = []
    for crop in diagnostics.get("crops") or []:
        crop_path = crop.get("crop_path")
        if not crop_path:
            continue
        rows.append(
            Evidence(
                subject_type=SubjectType.CROP,
                subject_id=subject_id(crop_path),
                kind=EvidenceKind.JERSEY_NUMBER,
                value=evidence_value(crop.get("ctc_top1")),
                score=_float_or_none(crop.get("ctc_top1_log_probability")),
                frame_start=_int_or_none(crop.get("frame")),
                frame_end=_int_or_none(crop.get("frame")),
                produced_by=PRODUCER_JERSEY_REGION_CTC,
                model_sha256=ctc_sha,
                config_hash=config_hash,
                payload={
                    key: crop[key]
                    for key in (
                        "display_track_id",
                        "crop_sha256",
                        "crop_bytes",
                        "crop_quality",
                        "selection_score",
                        "selection_reason",
                        "selection_rank",
                        "detector_confidence",
                        "detector_checkpoint_sha256",
                        "region_xyxyn",
                        "region_width",
                        "region_height",
                        "box_padding",
                        "ctc_top5",
                    )
                    if key in crop
                },
            )
        )
    return rows


def team_evidence(team_assignments, config_hash):
    """Evidence rows from ``TeamAssigner.fit_apply`` output."""
    rows = []
    for display_id, assignment in (team_assignments or {}).items():
        if assignment is None:
            continue
        rows.append(
            Evidence(
                subject_type=SubjectType.DISPLAY_TRACK,
                subject_id=display_subject(GROUP_PLAYERS, display_id),
                kind=EvidenceKind.TEAM,
                value=evidence_value(assignment.get("team")),
                score=_float_or_none(assignment.get("confidence")),
                produced_by=PRODUCER_TEAM_COLOR,
                config_hash=config_hash,
                payload={
                    key: assignment[key]
                    for key in ("source", "margin", "scores", "distances", "num_colors")
                    if key in assignment
                },
            )
        )
    return rows


def referee_role_evidence(referee_diagnostics, config_hash):
    """Role evidence from the referee colour pass, for both track groups."""
    if not referee_diagnostics or referee_diagnostics.get("enabled") is False:
        return []
    rows = []
    for group in (GROUP_REFEREES, GROUP_PLAYERS):
        for display_id, summary in (referee_diagnostics.get(group) or {}).items():
            # Only the referees group carries a decision: _apply_group sets
            # role_detection="referee" unconditionally there. For players the
            # stage decides per track via _is_player_referee_candidate, using a
            # higher colour threshold plus team-confidence conditions that are
            # not in these diagnostics. Emitting "referee" from
            # is_referee_palette alone would invent a decision the stage never
            # made -- on SNGS-025 that mislabelled 13 of 20 players. So players
            # abstain, and the raw palette signal travels in the payload.
            value = "referee" if group == GROUP_REFEREES else None
            rows.append(
                Evidence(
                    subject_type=SubjectType.DISPLAY_TRACK,
                    subject_id=display_subject(group, display_id),
                    kind=EvidenceKind.ROLE,
                    value=value,
                    score=_float_or_none(summary.get("score")),
                    produced_by=PRODUCER_REFEREE_PALETTE,
                    config_hash=config_hash,
                    payload={
                        "track_group": group,
                        "color": summary.get("color"),
                        "num_samples": summary.get("num_samples"),
                        "is_referee_palette": bool(summary.get("is_referee_palette", False)),
                    },
                )
            )
    return rows


def linking_evidence(linking_diagnostics, config_hash):
    """One row per accepted tracklet link, keyed by the merged display track."""
    if not linking_diagnostics or not linking_diagnostics.get("enabled"):
        return []
    rows = []
    for link in linking_diagnostics.get("accepted_links") or []:
        rows.append(
            Evidence(
                subject_type=SubjectType.DISPLAY_TRACK,
                subject_id=display_subject(GROUP_PLAYERS, link["display_track_id"]),
                kind=EvidenceKind.LINK,
                value=subject_id(link["to_track_id"]),
                score=_float_or_none(link.get("visual_similarity")),
                produced_by=PRODUCER_TRACKLET_LINKER,
                config_hash=config_hash,
                payload={
                    "from_track_id": link.get("from_track_id"),
                    "to_track_id": link.get("to_track_id"),
                    "gap": link.get("gap"),
                    "distance": link.get("distance"),
                },
            )
        )
    return rows


def _scene_segment_from_key(key):
    """Recover the scene segment from a ``display_track_id#scene_segment`` key."""
    parts = str(key).split("#", 1)
    return parts[1] if len(parts) == 2 else None


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
