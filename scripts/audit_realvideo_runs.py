#!/usr/bin/env python3
"""Summarize real-video identity runs for quick regression checks."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EMPTY = {"", "None", "unknown", None}


def nonempty(value):
    return value not in EMPTY


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run, video_id, artifacts_root, weak_link_confidence):
    root = artifacts_root / run / "metadata"
    tracklets_path = root / f"{video_id}_tracklets.csv"
    if not tracklets_path.exists():
        return {"run": run, "missing": str(tracklets_path)}

    with tracklets_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    players = [row for row in rows if row.get("track_group", "players") == "players"]
    refs = [row for row in rows if row.get("track_group") == "referees"]

    constraints = load_json(root / f"{video_id}_constraints.json")
    ocr = load_json(root / f"{video_id}_jersey_ocr.json")
    linking = load_json(root / f"{video_id}_jersey_identity_linking.json")
    propagation = load_json(root / f"{video_id}_identity_propagation.json")
    segment_candidates = load_segment_candidates(root / f"{video_id}_segment_jersey_candidates.csv")
    accepted_links = linking.get("accepted_links") or []
    propagations = propagation.get("propagations") or []
    weak_links = [
        link
        for link in accepted_links
        if min(
            float(link.get("from_jersey_confidence", 0.0) or 0.0),
            float(link.get("to_jersey_confidence", 0.0) or 0.0),
        )
        < weak_link_confidence
    ]

    return {
        "run": run,
        "rows": len(rows),
        "player_rows": len(players),
        "referee_group_rows": len(refs),
        "unique_display_ids": len({row.get("display_track_id") for row in players}),
        "unique_identity_tracklet_ids": len(
            {
                row.get("identity_tracklet_id") or row.get("display_track_id")
                for row in players
            }
        ),
        "assigned_rows": sum(1 for row in players if nonempty(row.get("player_id"))),
        "candidate_rows": sum(1 for row in players if nonempty(row.get("candidate_player_id"))),
        "propagated_rows": sum(1 for row in players if is_propagated_identity(row)),
        "segment_candidate_applied_rows": sum(
            1 for row in players if nonempty(row.get("segment_candidate_player_id"))
        ),
        "jersey_rows": sum(1 for row in players if nonempty(row.get("jersey_number"))),
        "gk_rows": sum(1 for row in players if row.get("role_detection") == "goalkeeper"),
        "top_jerseys": Counter(row.get("jersey_number") or "unknown" for row in players).most_common(15),
        "constraints": {
            key: constraints.get(key)
            for key in (
                "duplicate_player_frame_count",
                "remaining_duplicate_team_jersey_count",
                "remaining_duplicate_player_id_count",
                "goalkeeper_only_jersey_count",
            )
        },
        "number_roi": ocr.get("number_roi"),
        "segment_frames": ocr.get("segment_frames"),
        "mmocr": ocr.get("mmocr"),
        "roster_filter": (ocr.get("roster_filter") or {}).get("mode"),
        "segment_candidate_rows": len(segment_candidates),
        "segment_candidate_top_jerseys": Counter(
            row.get("jersey_number") or "unknown" for row in segment_candidates
        ).most_common(15),
        "accepted_jersey_links": len(accepted_links),
        "weak_jersey_links": len(weak_links),
        "weak_jersey_link_samples": weak_links[:10],
        "identity_propagation": {
            "enabled": propagation.get("enabled"),
            "status": propagation.get("status"),
            "total_propagated": propagation.get("total_propagated"),
            "accepted": len(propagations),
            "applied_frames": sum(int(item.get("applied_frames", 0) or 0) for item in propagations),
            "rejection_counts": Counter(
                item.get("reason", "unknown") for item in propagation.get("rejected_propagations", [])
            ).most_common(10),
            "samples": propagations[:10],
        },
    }


def load_segment_candidates(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_propagated_identity(row):
    evidence = row.get("identity_evidence")
    if not evidence:
        return False
    try:
        data = json.loads(evidence)
    except Exception:
        return False
    return data.get("status") == "propagated" or data.get("source") == "identity_propagation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/costume-video",
        type=Path,
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--weak-link-confidence", type=float, default=0.35)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summaries = [
        summarize_run(run, args.video_id, args.artifacts_root, args.weak_link_confidence)
        for run in args.run
    ]
    text = json.dumps(summaries, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
