"""Core data contracts shared by every pipeline stage."""

from ft.core.evidence import Evidence, EvidenceKind, SubjectType, subject_id
from ft.core.evidence_store import EvidenceStore

__all__ = ["Evidence", "EvidenceKind", "SubjectType", "EvidenceStore", "subject_id"]
