"""Immutable evidence records produced by feature stages.

An ``Evidence`` row states that a producer observed something about a subject.
It is deliberately *not* a decision: several producers may emit contradictory
evidence of the same ``kind`` for the same subject, and the decision layer is
the only component allowed to pick a winner.

Stages append evidence; nothing ever rewrites it. That is what makes an
``audit`` run the default behaviour instead of a mode that has to be enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


class SubjectType:
    """What an evidence row talks about."""

    CROP = "crop"
    RAW_TRACK = "raw_track"
    DISPLAY_TRACK = "display_track"
    IDENTITY_TRACKLET = "identity_tracklet"

    ALL = (CROP, RAW_TRACK, DISPLAY_TRACK, IDENTITY_TRACKLET)


class EvidenceKind:
    """Semantic channels. One kind, many possible producers."""

    JERSEY_NUMBER = "jersey_number"
    TEAM = "team"
    ROLE = "role"
    LINK = "link"
    PITCH_POSITION = "pitch_position"
    APPEARANCE_EMBEDDING = "appearance_embedding"

    ALL = (
        JERSEY_NUMBER,
        TEAM,
        ROLE,
        LINK,
        PITCH_POSITION,
        APPEARANCE_EMBEDDING,
    )


# Columns written to the tabular artifact, in order. ``payload`` is serialized
# as canonical JSON so the file stays flat and diffable.
COLUMNS = (
    "subject_type",
    "subject_id",
    "kind",
    "value",
    "score",
    "frame_start",
    "frame_end",
    "produced_by",
    "model_sha256",
    "config_hash",
    "payload",
)


@dataclass(frozen=True)
class Evidence:
    """One observation about one subject, by one producer.

    ``value`` is always a string (or ``None``) so that heterogeneous kinds share
    a single column. ``value=None`` with a populated ``score`` means the producer
    ran and explicitly abstained; a missing row means it never ran.
    """

    subject_type: str
    subject_id: str
    kind: str
    value: "str | None"
    score: "float | None"
    produced_by: str
    config_hash: str
    frame_start: "int | None" = None
    frame_end: "int | None" = None
    model_sha256: "str | None" = None
    payload: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.subject_type not in SubjectType.ALL:
            raise ValueError(f"unknown subject_type: {self.subject_type}")
        if self.kind not in EvidenceKind.ALL:
            raise ValueError(f"unknown kind: {self.kind}")
        if not isinstance(self.subject_id, str):
            # Subject ids are string keys on purpose: display_track_id is an int
            # in some artifacts and a str in others, and that mismatch has to
            # stop at this boundary.
            raise TypeError("subject_id must be a str")
        if self.value is not None and not isinstance(self.value, str):
            raise TypeError("value must be a str or None")

    @property
    def abstained(self):
        return self.value is None

    def to_dict(self):
        return asdict(self)


def subject_id(value):
    """Normalize any upstream identifier into a stable subject id."""
    if value is None:
        raise ValueError("subject_id cannot be None")
    if isinstance(value, tuple):
        return ":".join(str(int(part)) for part in value)
    if isinstance(value, bool):
        raise TypeError("subject_id cannot be a bool")
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if not text:
        raise ValueError("subject_id cannot be empty")
    return text


def evidence_value(value):
    """Normalize a decided value, mapping the legacy empty markers to ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "None", "unknown", "-1"}:
            return None
        return text
    if isinstance(value, int) and value == -1:
        return None
    return str(value)
