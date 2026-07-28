#!/usr/bin/env python3
"""Gate for step 1 of the evidence migration: the run must stay unchanged.

Compares the metadata directories of a control run (evidence code absent) and a
candidate run (evidence emitted). Every legacy artifact must match, and the
candidate may add only the two evidence files.

JSON artifacts are compared after stripping VOLATILE_KEYS -- wall clock, stage
timings, byte counts and OCR cache counters -- because an A/A run of identical
code showed those differ on their own. Files that match only after that
stripping are reported separately in ``changed_volatile_fields_only``, never
folded into the pass silently. Non-JSON artifacts are compared byte for byte.

Usage:

    python scripts/verify_evidence_additive.py \
        --control  evaluation_outputs/smoke_control/metadata \
        --candidate evaluation_outputs/smoke_evidence/metadata \
        --video-id SNGS-025
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ft.core.evidence import COLUMNS  # noqa: E402
from ft.core.evidence_store import read_evidence  # noqa: E402
from ft.utils.run_diagnostics import sha256_file  # noqa: E402


# Fields that legitimately differ between two runs of identical code: wall
# clock, per-stage timings, on-disk byte counts and OCR cache counters. An A/A
# run (same code twice) showed these are the only sources of noise, so they are
# stripped before hashing JSON artifacts. Everything else must match exactly.
VOLATILE_KEYS = frozenset(
    {
        "started_at",
        "finished_at",
        "timestamp",
        "seconds",
        "total_stage_seconds",
        "duration",
        "elapsed",
        "artifacts_bytes",
        "artifacts_bytes_before",
        "artifacts_bytes_after",
        "artifacts_bytes_delta",
        "hits",
        "misses",
        "writes",
    }
)


def strip_volatile(value):
    """Recursively drop volatile keys so two runs stay comparable."""
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


# The resolved configuration and its hash change by construction whenever a
# config key is added, which every migration step does. They are tracked in
# their own bucket instead of counting as content regressions -- but only the
# caller, with --allow-config-change, can turn that bucket into a pass.
CONFIG_KEYS = frozenset({"config", "config_sha256"})


def strip_config(value):
    if isinstance(value, dict):
        return {
            key: strip_config(item)
            for key, item in value.items()
            if key not in CONFIG_KEYS
        }
    if isinstance(value, list):
        return [strip_config(item) for item in value]
    return value


def _json_digest(payload, drop_config):
    if drop_config:
        payload = strip_config(payload)
    canonical = json.dumps(
        strip_volatile(payload), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_digest(path):
    """Hashes of a JSON artifact: volatile stripped, and also config stripped."""
    if path.suffix != ".json":
        return None, None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, None
    return _json_digest(payload, False), _json_digest(payload, True)


def load_json(path):
    if path.suffix != ".json":
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def is_additive(control, candidate):
    """True when candidate only *adds* keys to control.

    An inserted diagnostics key carries the same guarantee as an added file:
    nothing the control produced was altered or lost. Lists are ordered data,
    so a differing length or element counts as a content change, not an
    addition.
    """
    if isinstance(control, dict) and isinstance(candidate, dict):
        return all(
            key in candidate and is_additive(value, candidate[key])
            for key, value in control.items()
        )
    if isinstance(control, list) and isinstance(candidate, list):
        return len(control) == len(candidate) and all(
            is_additive(a, b) for a, b in zip(control, candidate)
        )
    return control == candidate


def added_keys(control, candidate, path=""):
    """Paths present only in the candidate, for the report."""
    if isinstance(control, dict) and isinstance(candidate, dict):
        for key in sorted(candidate):
            if key not in control:
                yield f"{path}/{key}"
            else:
                yield from added_keys(control[key], candidate[key], f"{path}/{key}")
    elif isinstance(control, list) and isinstance(candidate, list) and len(control) == len(candidate):
        for index, (a, b) in enumerate(zip(control, candidate)):
            yield from added_keys(a, b, f"{path}[{index}]")


def digests(directory):
    """Map filename -> (raw sha256, volatile-stripped sha256, +config-stripped)."""
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            without_volatile, without_config = normalized_digest(path)
            result[path.name] = (sha256_file(path), without_volatile, without_config)
    return result


def check_artifacts(control, candidate, allowed_new):
    control_digests = digests(control)
    candidate_digests = digests(candidate)

    added = sorted(set(candidate_digests) - set(control_digests))
    removed = sorted(set(control_digests) - set(candidate_digests))

    changed = []
    volatile_only = []
    config_only = []
    added_key_only = {}
    for name in sorted(set(control_digests) & set(candidate_digests)):
        control_raw, control_norm, control_no_config = control_digests[name]
        candidate_raw, candidate_norm, candidate_no_config = candidate_digests[name]
        if control_raw == candidate_raw:
            continue
        if control_norm is not None and control_norm == candidate_norm:
            # Byte-different but semantically identical: timings, timestamps or
            # cache counters only. Reported, never hidden.
            volatile_only.append(name)
            continue
        if control_no_config is not None and control_no_config == candidate_no_config:
            config_only.append(name)
            continue
        control_payload = load_json(Path(control) / name)
        candidate_payload = load_json(Path(candidate) / name)
        if control_payload is not None and candidate_payload is not None:
            a = strip_config(strip_volatile(control_payload))
            b = strip_config(strip_volatile(candidate_payload))
            if is_additive(a, b):
                added_key_only[name] = sorted(added_keys(a, b))
                continue
        changed.append(name)

    unexpected = sorted(set(added) - set(allowed_new))
    return {
        "control_files": len(control_digests),
        "candidate_files": len(candidate_digests),
        "added": added,
        "removed": removed,
        "changed": changed,
        "changed_volatile_fields_only": volatile_only,
        "changed_config_surface_only": config_only,
        "changed_added_keys_only": added_key_only,
        "unexpected_new_files": unexpected,
    }


def check_evidence(candidate, video_id, manifest):
    """Structural checks on the new artifact itself."""
    path = None
    for suffix in (".parquet", ".jsonl"):
        guess = Path(candidate) / f"{video_id}_evidence{suffix}"
        if guess.is_file():
            path = guess
            break
    if path is None:
        return {"status": "missing_evidence_artifact"}

    records = read_evidence(path)
    return {
        "status": "ok",
        "artifact": path.name,
        "rows": len(records),
        "columns_ok": all(set(record) == set(COLUMNS) for record in records),
        "manifest_total_matches": manifest.get("total") == len(records),
        "per_kind_producer": manifest.get("per_kind_producer", {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, help="metadata dir of the control run")
    parser.add_argument("--candidate", required=True, help="metadata dir of the evidence run")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--json", dest="json_out", help="optional path for the report")
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help="treat differences confined to the resolved config and its hash as a pass "
        "(use when the change intentionally adds a configuration key)",
    )
    parser.add_argument(
        "--allow-added-keys",
        action="store_true",
        help="treat artifacts that only gained new keys as a pass; every key the "
        "control produced must still be present and unchanged",
    )
    args = parser.parse_args()

    allowed_new = {
        f"{args.video_id}_evidence.parquet",
        f"{args.video_id}_evidence.jsonl",
        f"{args.video_id}_evidence_manifest.json",
    }

    manifest_path = Path(args.candidate) / f"{args.video_id}_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    artifacts = check_artifacts(args.control, args.candidate, allowed_new)
    evidence = check_evidence(args.candidate, args.video_id, manifest)

    config_surface = artifacts["changed_config_surface_only"]
    key_additions = artifacts["changed_added_keys_only"]
    passed = (
        not artifacts["changed"]
        and not artifacts["removed"]
        and not artifacts["unexpected_new_files"]
        and (args.allow_config_change or not config_surface)
        and (args.allow_added_keys or not key_additions)
        and evidence.get("status") == "ok"
        and evidence.get("columns_ok")
        and evidence.get("manifest_total_matches")
    )
    report = {
        "passed": passed,
        "allow_config_change": bool(args.allow_config_change),
        "allow_added_keys": bool(args.allow_added_keys),
        "artifacts": artifacts,
        "evidence": evidence,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
