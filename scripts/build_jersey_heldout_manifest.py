#!/usr/bin/env python3
"""Carve a fresh held-out block of GSR sequences for a jersey experiment.

Every earlier block (9 development, 11 frozen, the 12-sequence detection pilot,
the prefix-consolidation validation sets) has been opened, and GSR test is
frozen. Measuring a new decision policy therefore needs sequences that no prior
experiment has touched.

Two properties make this manifest usable as evidence rather than as a guess:

* exclusions are **derived**, not typed. Every file passed to --exclude-scan is
  searched for sequence ids, and each excluded sequence is reported with the
  file that disqualified it. Nothing is excluded "from memory".
* selection is deterministic and declared **before** any number is looked at:
  seed, pool, eligibility rule and the resulting list are written to the
  manifest, which hashes itself.

Eligibility is computed from ground-truth composition only -- how many player
tracks in a sequence carry a jersey number. That is dataset stratification, not
model-driven selection, so it does not leak.

Typical use, two phases:

    # 1. inspect the pool, select nothing
    python scripts/build_jersey_heldout_manifest.py --dry-run \
        --gsr-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
        --split train \
        --exclude-scan handoff.md docs evaluation evaluation_outputs

    # 2. commit to a block
    python scripts/build_jersey_heldout_manifest.py \
        --gsr-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
        --split train --size 15 --seed 20260726 \
        --exclude-scan handoff.md docs evaluation evaluation_outputs \
        --output evaluation/jersey_heldout_manifests/jersey_policy_heldout_v1.json
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SEQUENCE_RE = re.compile(r"SNGS-\d{3,}")
TEXT_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".tex", ".py"}
# Scanning these would be pointless and slow.
SKIP_DIR_NAMES = {".git", "__pycache__", ".ft_cache", "img1", "crops"}


def load_run_eval():
    """Reuse the evaluator's GT parsing so the counts match the measurement."""
    path = REPO_ROOT / "evaluation" / "gsr_jersey_ocr" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("gsr_run_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_scan_files(roots):
    for root in roots:
        path = Path(root)
        if path.is_file():
            yield path
            continue
        if not path.is_dir():
            continue
        for child in path.rglob("*"):
            if not child.is_file() or child.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in SKIP_DIR_NAMES for part in child.parts):
                continue
            yield child


def scan_exclusions(roots):
    """Map sequence id -> sorted list of files that mention it."""
    found = {}
    scanned = 0
    for path in iter_scan_files(roots):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        for sequence in set(SEQUENCE_RE.findall(text)):
            found.setdefault(sequence, set()).add(str(path))
    return {key: sorted(value) for key, value in sorted(found.items())}, scanned


def sequence_jersey_stats(label_path, run_eval, allowed_roles):
    """Count tracks and jersey-bearing tracks from one Labels-GameState.json."""
    payload = run_eval.read_json(label_path)
    categories = {
        category.get("id"): category.get("name")
        for category in payload.get("categories") or []
    }
    tracks = {}
    for annotation in payload.get("annotations") or []:
        if not isinstance(annotation, dict):
            continue
        role = run_eval.resolve_role(annotation, categories)
        if role not in allowed_roles:
            continue
        track_id = annotation.get("track_id") or annotation.get("person_id")
        if track_id in (None, ""):
            continue
        jersey = run_eval.jersey_number_from_attributes(annotation.get("attributes") or {})
        entry = tracks.setdefault(str(track_id), {"annotations": 0, "jersey": None})
        entry["annotations"] += 1
        if jersey is not None and entry["jersey"] is None:
            entry["jersey"] = int(jersey)
    with_jersey = [t for t in tracks.values() if t["jersey"] is not None]
    return {
        "tracks": len(tracks),
        "tracks_with_jersey": len(with_jersey),
        "annotations_with_jersey": sum(t["annotations"] for t in with_jersey),
        "distinct_numbers": len({t["jersey"] for t in with_jersey}),
    }


def build_pool(gsr_dir, split, run_eval, allowed_roles, excluded):
    pool, skipped = [], []
    for label_path in sorted(Path(gsr_dir).rglob("Labels-GameState.json")):
        sequence = label_path.parent.name
        if run_eval.infer_split(label_path).lower() != split:
            continue
        if sequence in excluded:
            skipped.append(
                {"sequence": sequence, "reason": "already_used", "evidence": excluded[sequence]}
            )
            continue
        stats = sequence_jersey_stats(label_path, run_eval, allowed_roles)
        pool.append({"sequence": sequence, **stats})
    return pool, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gsr-dir", default="/media/data-lie/cappetti/dataset/SoccerNet-GSR")
    parser.add_argument("--split", default="train", choices=["train", "val", "valid"])
    parser.add_argument("--exclude-scan", nargs="*", default=[], help="files/dirs searched for sequence ids")
    parser.add_argument("--exclude", nargs="*", default=[], help="extra sequence ids to exclude explicitly")
    parser.add_argument("--roles", nargs="+", default=["player", "goalkeeper"])
    parser.add_argument("--min-jersey-tracks", type=int, default=8)
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--dry-run", action="store_true", help="report the pool, select nothing")
    parser.add_argument("--output")
    args = parser.parse_args()

    split = "val" if args.split == "valid" else args.split
    if split == "test":  # pragma: no cover - argparse already forbids it
        raise SystemExit("GSR test is frozen and cannot host a new held-out")

    run_eval = load_run_eval()
    allowed_roles = set(args.roles)

    excluded, scanned_files = scan_exclusions(args.exclude_scan)
    for sequence in args.exclude:
        excluded.setdefault(sequence, []).append("--exclude")

    pool, skipped = build_pool(args.gsr_dir, split, run_eval, allowed_roles, excluded)
    eligible = [s for s in pool if s["tracks_with_jersey"] >= args.min_jersey_tracks]
    too_small = [s for s in pool if s["tracks_with_jersey"] < args.min_jersey_tracks]

    print(f"scanned files for exclusions : {scanned_files}")
    print(f"sequence ids found in record : {len(excluded)}")
    print(f"split                        : {split}")
    print(f"excluded (already used)      : {len(skipped)}")
    print(f"unused sequences             : {len(pool)}")
    print(f"  eligible (>= {args.min_jersey_tracks} jersey tracks): {len(eligible)}")
    print(f"  too small                  : {len(too_small)}")
    if eligible:
        counts = [s["tracks_with_jersey"] for s in eligible]
        print(f"  jersey tracks per sequence : min={min(counts)} max={max(counts)} total={sum(counts)}")

    if args.dry_run:
        for entry in eligible[:40]:
            print(f"    {entry['sequence']}  jersey_tracks={entry['tracks_with_jersey']}")
        print("\ndry run: nothing selected, no manifest written")
        return 0

    if len(eligible) < args.size:
        raise SystemExit(
            f"only {len(eligible)} eligible sequences, {args.size} requested. "
            "Lower --size or --min-jersey-tracks rather than reusing an opened block."
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(sorted(s["sequence"] for s in eligible), args.size))
    by_sequence = {s["sequence"]: s for s in eligible}

    manifest = {
        "name": Path(args.output).stem if args.output else "jersey_heldout",
        "purpose": "independent held-out for a jersey decision-policy experiment",
        "split": split,
        "seed": args.seed,
        "size": args.size,
        "selection": "random.Random(seed).sample over the sorted eligible pool",
        "eligibility": {
            "roles": sorted(allowed_roles),
            "min_jersey_tracks": args.min_jersey_tracks,
            "rule": "ground-truth composition only; no model output involved",
        },
        "sequences": selected,
        "sequence_stats": {name: by_sequence[name] for name in selected},
        "totals": {
            "sequences": len(selected),
            "tracks_with_jersey": sum(by_sequence[n]["tracks_with_jersey"] for n in selected),
            "tracks": sum(by_sequence[n]["tracks"] for n in selected),
        },
        "pool": {
            "eligible": len(eligible),
            "too_small": len(too_small),
            "excluded_already_used": len(skipped),
        },
        "exclusions": {
            "scanned_roots": list(args.exclude_scan),
            "scanned_files": scanned_files,
            "sequences": {entry["sequence"]: entry["evidence"] for entry in skipped},
        },
        "rules": [
            "declared before any metric was computed",
            "never reuse for threshold or checkpoint selection",
            "one evaluation only; a second question needs a new block",
        ],
    }

    body = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    manifest["manifest_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    body = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)

    print("\nselected:", " ".join(selected))
    print("tracks with jersey:", manifest["totals"]["tracks_with_jersey"])
    print("manifest sha256:", manifest["manifest_sha256"])

    if args.output:
        path = Path(args.output)
        if path.exists():
            raise SystemExit(
                f"{path} already exists. A held-out is declared once; "
                "write a new version instead of overwriting it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
        print("written:", path)
    else:
        print("\n(no --output given, manifest not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
