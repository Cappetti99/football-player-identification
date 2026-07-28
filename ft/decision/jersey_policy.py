"""Select which jersey evidence becomes an assignment.

This is the only place allowed to arbitrate between competing recognizers. The
producers write evidence; this module reads it and produces the legacy
``{track_id: assignment}`` mapping that ``_apply_jersey`` consumes, so the rest
of the pipeline is unaffected.

Promoting a recognizer is a configuration change:

```yaml
decision_policy:
  jersey_number:
    sources: [jersey_region_ctc, jersey_ocr_primary]
```

The default keeps a single source and reproduces today's behaviour exactly.

Subject granularity caveat: the primary OCR emits on ``players:<id>`` or, when
``jersey_ocr.segment_frames > 0``, on ``players:<id>:<segment>``; the region CTC
always emits on ``players:<id>``. With segments enabled the two never meet, so
a CTC source cannot win on a segmented tracklet. That is reported in the
diagnostics rather than papered over with an invented reconciliation rule.
"""

from __future__ import annotations

from ft.core.emitters import PRODUCER_JERSEY_PRIMARY
from ft.core.evidence import EvidenceKind, SubjectType


DEFAULT_JERSEY_POLICY = {
    "sources": [PRODUCER_JERSEY_PRIMARY],
    # "fallback": if a source abstains on a subject, try the next one. The only
    # transition this can produce against a single-source baseline is
    # unknown -> number; a source earlier in the list is never overridden.
    "on_abstain": "fallback",
}

# Subjects a jersey decision can attach to. Crop-level rows are per-frame
# evidence and are never promoted directly.
DECISION_SUBJECTS = (SubjectType.IDENTITY_TRACKLET, SubjectType.DISPLAY_TRACK)


def resolve_jersey_assignments(evidence, policy=None):
    """Return ``(assignments, diagnostics)`` chosen from evidence by policy.

    ``assignments`` is keyed exactly like the legacy jersey mapping, so it can
    be handed to ``_apply_jersey`` unchanged.
    """
    policy = normalize_policy(policy)
    sources = policy["sources"]
    fallback = policy["on_abstain"] == "fallback"

    by_subject = {}
    for row in evidence or ():
        if row.kind != EvidenceKind.JERSEY_NUMBER:
            continue
        if row.subject_type not in DECISION_SUBJECTS:
            continue
        if row.produced_by not in sources:
            continue
        by_subject.setdefault(row.subject_id, {})[row.produced_by] = row

    assignments = {}
    decisions = []
    for subject_id in sorted(by_subject):
        candidates = by_subject[subject_id]
        winner, reason = pick_winner(candidates, sources, fallback)
        decisions.append(
            {
                "subject_id": subject_id,
                "chosen_source": winner.produced_by if winner else None,
                "jersey_number": winner.value if winner else None,
                "reason": reason,
                "available_sources": sorted(candidates),
            }
        )
        if winner is None or winner.abstained:
            continue
        key, assignment = legacy_assignment(winner)
        if key is None:
            continue
        assignments[key] = assignment

    return assignments, {
        "policy": policy,
        "subjects": len(by_subject),
        "assigned": len(assignments),
        "decisions": decisions,
        "per_source": source_counts(decisions),
        "unreachable_sources": sorted(set(sources) - observed_sources(by_subject)),
    }


def pick_winner(candidates, sources, fallback):
    """First source in policy order that decided; honours the abstain rule."""
    first_present = None
    for source in sources:
        row = candidates.get(source)
        if row is None:
            continue
        if first_present is None:
            first_present = row
        if not row.abstained:
            reason = "first_source" if row is first_present else "fallback_after_abstain"
            return row, reason
        if not fallback:
            return row, "abstain_no_fallback"
    return first_present, "all_sources_abstained" if first_present else "no_source"


def legacy_assignment(row):
    """Rebuild the legacy assignment dict from one evidence row.

    Producers that already carry the legacy payload (the OCR recognizers) are
    reproduced verbatim. The region CTC has a different payload, so its fields
    are mapped explicitly and the provenance of that mapping is recorded.
    """
    payload = dict(row.payload or {})
    legacy_key = payload.pop("legacy_key", None)
    number = int(row.value)

    if row.produced_by == PRODUCER_JERSEY_PRIMARY or "votes" in payload:
        assignment = dict(payload)
        assignment["jersey_number"] = number
        assignment["confidence"] = row.score
        assignment["decision_source"] = row.produced_by
        return parse_legacy_key(legacy_key), assignment

    # Region CTC: recognized_frames is the vote count and top5 the candidate
    # list. No roster mass or head confidence exists upstream, so those stay
    # absent instead of being fabricated.
    top5 = payload.get("top5") or []
    assignment = {
        "jersey_number": number,
        "confidence": row.score,
        "winner_margin": payload.get("winner_margin"),
        "votes": int(payload.get("recognized_frames") or 0),
        "display_track_id": payload.get("display_track_id"),
        "segment_index": None,
        "candidates": [{"jersey_number": int(value)} for value in top5],
        "decision_source": row.produced_by,
        "decision_mapping": "recognized_frames->votes, top5->candidates",
    }
    display_id = payload.get("display_track_id")
    key = int(display_id) if display_id is not None else parse_legacy_key(legacy_key)
    return key, assignment


def parse_legacy_key(text):
    """Legacy keys are ints, or ``(display, segment)`` tuples for segments."""
    if text is None:
        return None
    text = str(text)
    if ":" in text:
        display, segment = text.split(":", 1)
        return (int(display), int(segment))
    try:
        return int(text)
    except ValueError:
        return None


def normalize_policy(policy):
    merged = {**DEFAULT_JERSEY_POLICY, **(policy or {})}
    sources = list(merged.get("sources") or [])
    if not sources:
        raise ValueError("decision_policy.jersey_number.sources cannot be empty")
    if len(set(sources)) != len(sources):
        raise ValueError(f"duplicate sources in jersey policy: {sources}")
    on_abstain = str(merged.get("on_abstain", "fallback"))
    if on_abstain not in {"fallback", "abstain"}:
        raise ValueError(f"unknown on_abstain policy: {on_abstain}")
    return {"sources": sources, "on_abstain": on_abstain}


def source_counts(decisions):
    counts = {}
    for decision in decisions:
        source = decision["chosen_source"]
        if source is None or decision["jersey_number"] is None:
            continue
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def observed_sources(by_subject):
    return {source for candidates in by_subject.values() for source in candidates}
