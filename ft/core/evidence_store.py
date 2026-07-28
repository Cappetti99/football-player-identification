"""Append-only collector for :class:`~ft.core.evidence.Evidence` rows.

One store per run. Stages add to it; nothing removes from it. The store is a
pure in-memory structure with no pipeline dependencies so that a stage can be
exercised in a unit test without a video, a model or a GPU.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ft.core.evidence import COLUMNS, Evidence
from ft.utils.run_diagnostics import canonical_json


class EvidenceStore:
    """Ordered, append-only collection of evidence rows."""

    def __init__(self, config_hash=None):
        self.config_hash = config_hash
        self._rows = []

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    @property
    def rows(self):
        return tuple(self._rows)

    def add(self, evidence):
        if not isinstance(evidence, Evidence):
            raise TypeError("EvidenceStore accepts Evidence instances only")
        self._rows.append(evidence)
        return evidence

    def extend(self, evidences):
        added = 0
        for evidence in evidences or ():
            self.add(evidence)
            added += 1
        return added

    def query(self, kind=None, produced_by=None, subject_type=None, subject_id=None):
        """Filter rows. All criteria are ANDed; ``None`` means "any"."""
        return [
            row
            for row in self._rows
            if (kind is None or row.kind == kind)
            and (produced_by is None or row.produced_by == produced_by)
            and (subject_type is None or row.subject_type == subject_type)
            and (subject_id is None or row.subject_id == subject_id)
        ]

    def producers(self, kind=None):
        return sorted({row.produced_by for row in self.query(kind=kind)})

    def by_subject(self, kind=None, produced_by=None):
        """Group rows by ``subject_id``, preserving insertion order per subject."""
        grouped = {}
        for row in self.query(kind=kind, produced_by=produced_by):
            grouped.setdefault(row.subject_id, []).append(row)
        return grouped

    def summary(self):
        """Counts per kind/producer, used as the additive-run smoke gate."""
        per_kind = Counter(row.kind for row in self._rows)
        per_producer = Counter(f"{row.kind}/{row.produced_by}" for row in self._rows)
        return {
            "total": len(self._rows),
            "config_hash": self.config_hash,
            "per_kind": dict(sorted(per_kind.items())),
            "per_kind_producer": dict(sorted(per_producer.items())),
            "abstentions": sum(1 for row in self._rows if row.abstained),
        }

    def to_records(self):
        """Flat dicts ready for CSV/Parquet; ``payload`` becomes canonical JSON."""
        records = []
        for row in self._rows:
            record = row.to_dict()
            record["payload"] = canonical_json(record.get("payload") or {})
            records.append({column: record[column] for column in COLUMNS})
        return records

    def write(self, path):
        """Write Parquet when pyarrow is available, JSONL otherwise.

        Returns the path actually written, which the manifest records: the
        artifact format must never be inferred from the environment later on.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        if path.suffix == ".parquet":
            try:
                import pandas as pd

                pd.DataFrame(records, columns=list(COLUMNS)).to_parquet(path, index=False)
                return path
            except (ImportError, ValueError):
                # pyarrow/fastparquet missing in this env: fall back without
                # failing the run, since evidence is an additive artifact.
                path = path.with_suffix(".jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
        return path

    def manifest(self, artifact_path=None):
        payload = self.summary()
        payload["artifact"] = str(artifact_path) if artifact_path else None
        payload["models"] = dict(
            sorted(
                {
                    f"{row.kind}/{row.produced_by}": row.model_sha256
                    for row in self._rows
                    if row.model_sha256
                }.items()
            )
        )
        return payload


def read_evidence(path):
    """Read back an evidence artifact as flat records (JSONL or Parquet)."""
    import json

    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
