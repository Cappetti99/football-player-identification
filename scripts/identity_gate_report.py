#!/usr/bin/env python3
"""Human-readable per-tracklet explanation of the identity assignment gate.

Cross-references the three artifacts that today's manual debugging needed
separately (identity_assignments.json, jersey_ocr.json, candidate_scores.csv)
into one report: final status, which gate criterion passed/failed and why,
the underlying jersey OCR evidence (including full_body_sufficient), and the
best-cost roster candidate considered. No new pipeline run needed.

Run from ~/FT:
    python3 scripts/identity_gate_report.py --artifacts-dir artifacts/costume-video/<run_dir>
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def decode_assignment_key(key):
    """assignments dict keys are encoded as str(-(display_track_id*100000+1))."""
    try:
        value = int(key)
    except ValueError:
        return None
    if value >= 0:
        return value
    return (-value - 1) // 100000


def load_json(path):
    return json.loads(path.read_text())


def load_best_candidates(path):
    best_by_track = {}
    if not path.exists():
        return best_by_track
    with open(path, newline="", encoding="utf-8") as handle:
        rows_by_track = defaultdict(list)
        for row in csv.DictReader(handle):
            try:
                track_id = int(row["track_id"])
            except (KeyError, ValueError):
                continue
            rows_by_track[track_id].append(row)
    for track_id, rows in rows_by_track.items():
        rows.sort(key=lambda row: float(row.get("cost", 1.0) or 1.0))
        best_by_track[track_id] = rows[0]
    return best_by_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()

    root = Path(args.artifacts_dir)
    metadata_dir = root / "metadata"

    video_id = args.video_id
    if video_id is None:
        candidates = sorted(metadata_dir.glob("*_identity_assignments.json"))
        if not candidates:
            raise SystemExit(f"no *_identity_assignments.json found in {metadata_dir}")
        video_id = candidates[0].name[: -len("_identity_assignments.json")]

    identity_data = load_json(metadata_dir / f"{video_id}_identity_assignments.json")
    assignments = identity_data.get("assignments") or {}

    jersey_path = metadata_dir / f"{video_id}_jersey_ocr.json"
    jersey_tracks = load_json(jersey_path).get("tracklets") or {} if jersey_path.exists() else {}

    best_by_track = load_best_candidates(metadata_dir / f"{video_id}_candidate_scores.csv")

    rows = []
    for key, entry in assignments.items():
        track_id = decode_assignment_key(key)
        if track_id is None:
            continue
        rows.append((track_id, key, entry))
    rows.sort(key=lambda item: item[0])

    status_counts = defaultdict(int)
    reason_counts = defaultdict(int)

    print(f"Identity gate report: {video_id}")
    print(f"tracks: {len(rows)}")
    print()

    for track_id, key, entry in rows:
        status = entry.get("identity_status")
        status_counts[status] += 1
        evidence = entry.get("evidence") or {}
        gate = evidence.get("assignment_gate") or {}
        reason = gate.get("reason")
        reason_counts[reason] += 1

        jersey_track = jersey_tracks.get(str(track_id)) or {}
        voted = jersey_track.get("voted") or {}
        decision_status = (jersey_track.get("decision") or {}).get("status")
        full_body_sufficient = jersey_track.get("full_body_sufficient")

        best = best_by_track.get(track_id)

        print(f"--- track {track_id} (assignments key={key}) ---")
        print(
            f"  status={status}  player={entry.get('player_name')}"
            f"  confidence={round(float(entry.get('confidence') or 0), 3)}"
        )
        print(f"  gate: pass={gate.get('pass')} reason={reason}")
        print(
            "    reliable_jersey={}  goalkeeper_singleton={}  strong_combined={}".format(
                gate.get("reliable_jersey"), gate.get("goalkeeper_singleton"), gate.get("strong_combined")
            )
        )
        print(
            "    team_match={}  team_confidence={}  visual_similarity={}"
            "  position_prior_distance={}  tracklet_frames={}".format(
                gate.get("team_match"),
                round(float(gate.get("team_confidence") or 0), 3),
                gate.get("visual_similarity"),
                gate.get("position_prior_distance"),
                gate.get("tracklet_frames"),
            )
        )
        print(
            "  jersey OCR: number={} votes={} confidence={} winner_margin={}"
            " full_body_sufficient={} decision_status={}".format(
                voted.get("jersey_number"),
                voted.get("votes"),
                round(float(voted.get("confidence") or 0), 3),
                round(float(voted.get("winner_margin") or 0), 3),
                full_body_sufficient,
                decision_status,
            )
        )
        if best:
            print(
                "  best candidate: {} (team={}, jersey={}) cost={}".format(
                    best.get("player_name"),
                    best.get("player_team_id"),
                    best.get("player_jersey_number"),
                    round(float(best.get("cost") or 1.0), 3),
                )
            )
        risk_flags = entry.get("identity_risk_flags")
        if risk_flags:
            print(f"  risk_flags: {risk_flags}")
        print()

    print("=== summary ===")
    print("status counts:", dict(status_counts))
    print("gate reasons:", dict(reason_counts))


if __name__ == "__main__":
    main()
