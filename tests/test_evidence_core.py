import json

import pytest

from ft.core.evidence import (
    COLUMNS,
    Evidence,
    EvidenceKind,
    SubjectType,
    evidence_value,
    subject_id,
)
from ft.core.evidence_store import EvidenceStore, read_evidence


def evidence(**overrides):
    base = {
        "subject_type": SubjectType.IDENTITY_TRACKLET,
        "subject_id": "12",
        "kind": EvidenceKind.JERSEY_NUMBER,
        "value": "10",
        "score": 0.8,
        "produced_by": "unit_test",
        "config_hash": "abc",
    }
    base.update(overrides)
    return Evidence(**base)


def test_rejects_unknown_subject_type_and_kind():
    with pytest.raises(ValueError):
        evidence(subject_type="nonsense")
    with pytest.raises(ValueError):
        evidence(kind="nonsense")


def test_subject_id_must_be_a_string():
    # display_track_id is an int in some legacy artifacts and a str in others;
    # the mismatch has to stop at this boundary, not leak into joins.
    with pytest.raises(TypeError):
        evidence(subject_id=12)


def test_value_must_be_string_or_none():
    with pytest.raises(TypeError):
        evidence(value=10)


def test_abstention_is_distinct_from_absence():
    abstained = evidence(value=None, score=0.3)
    assert abstained.abstained
    assert not evidence().abstained


@pytest.mark.parametrize(
    "raw,expected",
    [(12, "12"), ("12", "12"), ((12, 3), "12:3"), ("  a1 ", "a1")],
)
def test_subject_id_normalization(raw, expected):
    assert subject_id(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_subject_id_rejects_empty(raw):
    with pytest.raises((ValueError, TypeError)):
        subject_id(raw)


@pytest.mark.parametrize("raw", ["", "None", "unknown", "-1", -1, None])
def test_legacy_empty_markers_become_abstentions(raw):
    assert evidence_value(raw) is None


def test_store_is_append_only_and_ordered():
    store = EvidenceStore(config_hash="abc")
    first = store.add(evidence(subject_id="1"))
    second = store.add(evidence(subject_id="2"))
    assert store.rows == (first, second)
    assert len(store) == 2
    with pytest.raises(TypeError):
        store.add({"kind": "jersey_number"})


def test_query_and_grouping_separate_competing_producers():
    store = EvidenceStore(config_hash="abc")
    store.add(evidence(subject_id="7", produced_by="sar", value="9"))
    store.add(evidence(subject_id="7", produced_by="ctc", value="5"))
    store.add(evidence(subject_id="8", produced_by="ctc", value="5"))

    assert store.producers(kind=EvidenceKind.JERSEY_NUMBER) == ["ctc", "sar"]
    assert len(store.query(produced_by="ctc")) == 2
    # Two contradictory readings for the same subject coexist by design: the
    # decision layer picks, the store does not.
    assert len(store.by_subject(kind=EvidenceKind.JERSEY_NUMBER)["7"]) == 2


def test_summary_counts_producers_and_abstentions():
    store = EvidenceStore(config_hash="abc")
    store.add(evidence(subject_id="1", produced_by="sar"))
    store.add(evidence(subject_id="2", produced_by="sar", value=None))
    summary = store.summary()
    assert summary["total"] == 2
    assert summary["abstentions"] == 1
    assert summary["per_kind"] == {"jersey_number": 2}
    assert summary["per_kind_producer"] == {"jersey_number/sar": 2}


def test_write_jsonl_roundtrip(tmp_path):
    store = EvidenceStore(config_hash="abc")
    store.add(evidence(payload={"votes": 4, "candidates": [{"jersey_number": 10}]}))
    path = store.write(tmp_path / "evidence.jsonl")

    records = read_evidence(path)
    assert len(records) == 1
    # JSONL rows are canonical JSON (sorted keys); only the column set is a
    # contract, the on-disk key order is not.
    assert set(records[0]) == set(COLUMNS)
    assert json.loads(records[0]["payload"])["votes"] == 4


def test_manifest_records_artifact_and_model_hashes(tmp_path):
    store = EvidenceStore(config_hash="abc")
    store.add(evidence(produced_by="ctc", model_sha256="deadbeef"))
    path = store.write(tmp_path / "evidence.jsonl")

    manifest = store.manifest(path)
    assert manifest["config_hash"] == "abc"
    assert manifest["artifact"] == str(path)
    assert manifest["models"] == {"jersey_number/ctc": "deadbeef"}


def test_export_writes_exactly_two_new_files(tmp_path):
    from ft.pipeline import export_evidence_artifacts

    store = EvidenceStore(config_hash="abc")
    store.add(evidence())
    export_evidence_artifacts(store, tmp_path, "SNGS-025")

    written = {p.name for p in (tmp_path / "metadata").iterdir()}
    # Step 1 is allowed to add these two files and nothing else.
    assert written in (
        {"SNGS-025_evidence.parquet", "SNGS-025_evidence_manifest.json"},
        {"SNGS-025_evidence.jsonl", "SNGS-025_evidence_manifest.json"},
    )
