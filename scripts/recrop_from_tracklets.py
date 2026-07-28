#!/usr/bin/env python3
"""Regenerate missing crop JPGs for a run whose crops/ directory was deleted
to reclaim disk space, but whose metadata/*_tracklets.csv and
metadata/*_run_manifest.json still exist.

Re-derives each crop from the original video using the exact bbox and frame
number recorded at export time, and writes it back to the same crop_path the
tracklets.csv already references -- so nothing downstream (benchmark
manifests, other scripts) needs to change.

Zero re-detection, re-tracking, or re-identification: this only re-does the
cheap bbox-crop step.
"""
import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="e.g. artifacts/costume-video/Int-Ata_prtreid_conservative_final_1200f")
    parser.add_argument("--tracklets-csv", default=None, help="defaults to metadata/<video_id>_tracklets.csv inside --run-dir")
    parser.add_argument("--video-path", default=None, help="overrides the video_path recorded in the run manifest")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metadata_dir = run_dir / "metadata"
    manifests = list(metadata_dir.glob("*_run_manifest.json"))
    if not manifests:
        raise SystemExit(f"no *_run_manifest.json found under {metadata_dir}")
    manifest = json.loads(manifests[0].read_text())
    video_id = manifest["video_id"]

    tracklets_csv = Path(args.tracklets_csv) if args.tracklets_csv else metadata_dir / f"{video_id}_tracklets.csv"
    if not tracklets_csv.exists():
        raise SystemExit(f"tracklets csv not found: {tracklets_csv}")

    video_path = args.video_path or manifest["config"]["video_path"]
    if not Path(video_path).exists():
        raise SystemExit(f"video not found: {video_path} (pass --video-path to override)")

    with tracklets_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    needed = {}
    for row in rows:
        crop_path = row.get("crop_path")
        if not crop_path or crop_path == "None":
            continue
        target = Path(crop_path)
        if args.skip_existing and target.exists():
            continue
        bbox = ast.literal_eval(row["bbox"])
        frame_num = int(row["frame"])
        needed.setdefault(frame_num, []).append((target, bbox))

    if not needed:
        print("nothing to do: all referenced crops already exist on disk")
        return

    print(f"video={video_path} tracklets_csv={tracklets_csv} frames_needed={len(needed)} crops_needed={sum(len(v) for v in needed.values())}")
    if args.dry_run:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    max_frame = max(needed)
    written = 0
    skipped_empty = 0
    frame_num = -1
    while frame_num < max_frame:
        ok, frame = cap.read()
        frame_num += 1
        if not ok:
            print(f"WARNING: video ended early at frame {frame_num}, {max_frame - frame_num} frames unreachable", file=sys.stderr)
            break
        targets = needed.get(frame_num)
        if not targets:
            continue
        for target, bbox in targets:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                skipped_empty += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(target), crop):
                written += 1
        if frame_num % 100 == 0:
            print(f"progress: frame={frame_num}/{max_frame} written={written}", flush=True)

    cap.release()
    print(f"done: written={written} skipped_empty={skipped_empty}")


if __name__ == "__main__":
    main()
