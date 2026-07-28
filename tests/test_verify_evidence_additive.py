import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "verify_evidence_additive",
    Path(__file__).resolve().parents[1] / "scripts" / "verify_evidence_additive.py",
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload))


def diagnostics(started, seconds, hits, misses, extra=None):
    payload = {
        "started_at": started,
        "total_stage_seconds": seconds,
        "stages": [{"stage": "jersey_ocr", "status": "ok", "seconds": seconds}],
        "jersey_ocr": {"cache": {"enabled": True, "hits": hits, "misses": misses}},
    }
    payload.update(extra or {})
    return payload


def test_timings_and_cache_counters_do_not_fail_the_gate(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "run_diagnostics.json", diagnostics("2026-01-01T00:00:00", 1907.4, 0, 86))
    write(candidate, "run_diagnostics.json", diagnostics("2026-07-25T20:03:18", 416.1, 86, 0))

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == []
    # Reported, not hidden: the file did change on disk.
    assert result["changed_volatile_fields_only"] == ["run_diagnostics.json"]


def test_a_real_content_change_still_fails(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "run_diagnostics.json", diagnostics("t0", 1.0, 0, 1, {"jersey_number": 10}))
    write(candidate, "run_diagnostics.json", diagnostics("t1", 2.0, 1, 0, {"jersey_number": 5}))

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == ["run_diagnostics.json"]
    assert result["changed_volatile_fields_only"] == []


def test_non_json_artifacts_are_compared_byte_for_byte(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    control.mkdir(parents=True)
    candidate.mkdir(parents=True)
    (control / "tracklets.csv").write_text("frame,jersey\n1,10\n")
    (candidate / "tracklets.csv").write_text("frame,jersey\n1,5\n")

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == ["tracklets.csv"]


def test_new_files_must_be_declared(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "a.json", {"x": 1})
    write(candidate, "a.json", {"x": 1})
    write(candidate, "SNGS-025_evidence_manifest.json", {"total": 0})
    write(candidate, "surprise.json", {"x": 1})

    result = gate.check_artifacts(
        control, candidate, allowed_new={"SNGS-025_evidence_manifest.json"}
    )
    assert result["unexpected_new_files"] == ["surprise.json"]


@pytest.mark.parametrize("key", ["seconds", "started_at", "hits", "artifacts_bytes"])
def test_volatile_keys_stripped_at_any_depth(key):
    payload = {"a": {"b": [{key: 1, "keep": 2}]}}
    assert gate.strip_volatile(payload) == {"a": {"b": [{"keep": 2}]}}


def manifest(config, started="t0"):
    return {"started_at": started, "config": config, "source_provenance": {"config_sha256": "h"}}


def test_config_surface_changes_get_their_own_bucket(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "run_manifest.json", manifest({"a": 1}))
    write(candidate, "run_manifest.json", manifest({"a": 1, "decision_policy": {"x": 1}}, "t1"))

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == []
    assert result["changed_config_surface_only"] == ["run_manifest.json"]


def test_content_change_outside_the_config_still_fails(tmp_path):
    """A config key may be added, but an existing value must not move."""
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "run_manifest.json", {**manifest({"a": 1}), "jersey_number": 10})
    write(candidate, "run_manifest.json", {**manifest({"a": 1, "new_key": 2}), "jersey_number": 5})

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == ["run_manifest.json"]
    assert result["changed_config_surface_only"] == []
    assert result["changed_added_keys_only"] == {}


def test_config_sha256_alone_is_a_config_surface_change(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "source_provenance.json", {"config_sha256": "aaa", "sources": {"x": 1}})
    write(candidate, "source_provenance.json", {"config_sha256": "bbb", "sources": {"x": 1}})

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed_config_surface_only"] == ["source_provenance.json"]


def test_a_purely_added_key_gets_its_own_bucket(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "jersey_ocr.json", {"assigned_tracklets": {"12": {"jersey_number": 10}}})
    write(candidate, "jersey_ocr.json", {
        "assigned_tracklets": {"12": {"jersey_number": 10}},
        "decision_policy": {"sources": ["jersey_ocr_primary"]},
    })

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == []
    assert result["changed_added_keys_only"] == {"jersey_ocr.json": ["/decision_policy"]}


def test_a_removed_key_is_not_additive(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "jersey_ocr.json", {"a": 1, "raw_jersey_distribution": [1]})
    write(candidate, "jersey_ocr.json", {"a": 1})

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == ["jersey_ocr.json"]


def test_a_changed_value_next_to_an_added_key_is_not_additive(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "jersey_ocr.json", {"jersey_number": 10})
    write(candidate, "jersey_ocr.json", {"jersey_number": 5, "decision_policy": {}})

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == ["jersey_ocr.json"]


def test_list_length_change_is_a_content_change(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "a.json", {"candidates": [{"jersey_number": 10}]})
    write(candidate, "a.json", {"candidates": [{"jersey_number": 10}, {"jersey_number": 5}]})

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == ["a.json"]


def test_added_key_inside_a_list_element_is_additive(tmp_path):
    control, candidate = tmp_path / "c", tmp_path / "k"
    write(control, "a.json", {"stages": [{"stage": "ocr"}]})
    write(candidate, "a.json", {"stages": [{"stage": "ocr", "note": "x"}]})

    result = gate.check_artifacts(control, candidate, allowed_new=set())
    assert result["changed"] == []
    assert result["changed_added_keys_only"] == {"a.json": ["/stages[0]/note"]}
