#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

from ft.identity.gate_audit import build_identity_gate_audit


csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Audit identity assignment gates once per tracklet.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--roster-path", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--near-miss-window", type=float, default=0.10)
    args = parser.parse_args()
    metadata = Path(args.artifacts_root) / args.run / "metadata"
    manifest = read_json(metadata / f"{args.video_id}_run_manifest.json")
    summaries = normalize_rows(read_csv(metadata / f"{args.video_id}_tracklet_summaries.csv"))
    scores = normalize_rows(read_csv(metadata / f"{args.video_id}_candidate_scores.csv"))
    assignment_payload = read_json(metadata / f"{args.video_id}_identity_assignments.json")
    assignments = {int(key): value for key, value in assignment_payload.get("assignments", {}).items()}
    roster = read_json(Path(args.roster_path))
    report = build_identity_gate_audit(
        summaries,
        scores,
        assignments,
        roster,
        identity_config=manifest.get("config", {}).get("identity", {}),
        near_miss_window=args.near_miss_window,
    )
    json_path = metadata / f"{args.video_id}_identity_gate_audit.json"
    csv_path = metadata / f"{args.video_id}_identity_gate_audit.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report["tracklets"], csv_path)
    print(json.dumps(report["summary"], indent=2))
    print(f"json={json_path}\ncsv={csv_path}")


def normalize_rows(rows):
    for row in rows:
        for key in ("assignment_gate", "raw_jersey_distribution", "jersey_distribution", "crop_paths", "visual_embedding"):
            value = row.get(key)
            if isinstance(value, str) and value[:1] in {"[", "{"}:
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


if __name__ == "__main__":
    main()
