#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from build_prtreid_pair_audit import (
    group_rows_by_display,
    read_csv,
    representative_rows,
    write_csv,
    write_html,
    write_pair_sheet,
)


def main():
    parser = argparse.ArgumentParser(description="Build a visual audit for every PRTReID identity-bridge candidate.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--output-dir")
    parser.add_argument("--frames-per-tracklet", type=int, default=4)
    args = parser.parse_args()
    metadata = Path(args.artifacts_root) / args.run / "metadata"
    diagnostics = json.loads((metadata / f"{args.video_id}_prtreid_identity_bridge.json").read_text())
    rows = read_csv(metadata / f"{args.video_id}_tracklets.csv")
    rows_by_display = group_rows_by_display(rows)
    output = Path(args.output_dir or Path("evaluation_outputs/prtreid_identity_bridge_audit") / args.run)
    sheets = output / "pairs"
    sheets.mkdir(parents=True, exist_ok=True)
    manifest = []
    candidates = unique_candidates(diagnostics.get("candidates", []))
    for index, candidate in enumerate(candidates, 1):
        anchor = int(candidate["anchor_display_track_id"])
        target = int(candidate["target_display_track_id"])
        pair_id = f"{args.video_id}_{index:04d}"
        sheet = sheets / f"{pair_id}.jpg"
        sheet_candidate = dict(candidate)
        sheet_candidate["from_display_track_id"] = anchor
        sheet_candidate["to_display_track_id"] = target
        sheet_candidate["link_type"] = "cross_scene_identity_bridge"
        write_pair_sheet(
            representative_rows(rows_by_display.get(anchor, []), args.frames_per_tracklet),
            representative_rows(rows_by_display.get(target, []), args.frames_per_tracklet),
            sheet,
            sheet_candidate,
        )
        manifest.append({
            "pair_id": pair_id,
            "video_id": args.video_id,
            "run": args.run,
            "anchor_display_track_id": anchor,
            "target_display_track_id": target,
            "source_player_id": candidate.get("source_player_id"),
            "source_player_name": candidate.get("source_player_name"),
            "link_type": "cross_scene_identity_bridge",
            "visual_similarity": candidate.get("visual_similarity"),
            "similarity_margin": candidate.get("similarity_margin"),
            "mutual_nearest": candidate.get("mutual_nearest"),
            "gap": candidate.get("gap"),
            "sheet_path": str(sheet),
            "label": "uncertain",
            "notes": "",
        })
    write_csv(manifest, output / "labels.csv")
    write_html(manifest, output / "index.html")
    print(f"bridge_candidates={len(manifest)} output={output}")


def unique_candidates(rows):
    unique = {}
    for row in rows:
        key = (int(row["anchor_display_track_id"]), int(row["target_display_track_id"]))
        unique[key] = row
    return sorted(unique.values(), key=lambda row: (-float(row.get("visual_similarity") or 0.0), -float(row.get("similarity_margin") or 0.0)))


if __name__ == "__main__":
    main()
