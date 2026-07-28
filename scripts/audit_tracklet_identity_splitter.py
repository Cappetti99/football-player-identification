#!/usr/bin/env python3
"""Audit-only GTA-style "Splitter": for each display tracklet in an existing
run, re-extract PRTReID embeddings from already-saved crops and run DBSCAN
to check whether the tracklet's samples form more than one visual cluster --
a candidate sign that ByteTrack silently mixed two different physical
players under one raw/display track_id, without ever changing track_id
(so a position-jump check alone would not catch it).

Motivation: `prtreid_tracklet_consistency` (a coarse mean-similarity-to-
prototype scalar, already computed by the live pipeline when
prtreid_linking/prtreid_identity_bridge is enabled) flagged display_track_id
4 on Int-Ata as the single lowest-consistency tracklet (0.8143) -- the same
raw/display id independently found earlier today to carry 6 conflicting
ground-truth jersey numbers across scene segments. This script checks the
stronger, paper-accurate signal (real per-sample clustering, not just a
mean-deviation scalar) using the same crops, no new pipeline run.

This is purely diagnostic: it writes nothing back into any track/identity
state. Nothing here decides or applies a split.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.prtreid import PRTReIDFeatureExtractor  # noqa: E402
from ft.features.visual import prtreid_extractor_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="e.g. artifacts/costume-video/Int-Ata_prtreid_conservative_final_1200f")
    parser.add_argument("--max-samples-per-tracklet", type=int, default=0, help="0 = no subsampling, use every available crop")
    parser.add_argument("--min-samples-for-clustering", type=int, default=6)
    parser.add_argument("--dbscan-eps", type=float, default=0.15)
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--only-display-track-id", type=int, default=None, help="restrict to one tracklet for a focused check")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metadata_dir = run_dir / "metadata"
    manifest = json.loads(next(metadata_dir.glob("*_run_manifest.json")).read_text())
    video_id = manifest["video_id"]
    prtreid_cfg = manifest["config"].get("prtreid", {})
    if not prtreid_cfg.get("enabled"):
        raise SystemExit(f"prtreid was not enabled for this run: {run_dir}")

    tracklets_csv = metadata_dir / f"{video_id}_tracklets.csv"
    with tracklets_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_display = defaultdict(list)
    for row in rows:
        crop_path = row.get("crop_path")
        if not crop_path or not Path(crop_path).is_file():
            continue
        by_display[row["display_track_id"]].append(row)

    if args.only_display_track_id is not None:
        target = str(args.only_display_track_id)
        by_display = {target: by_display[target]} if target in by_display else {}

    extractor = PRTReIDFeatureExtractor(**prtreid_extractor_config(prtreid_cfg))

    import cv2
    from sklearn.cluster import DBSCAN
    import numpy as np

    flagged = []
    skipped_too_few = 0
    for display_id, items in sorted(by_display.items(), key=lambda item: int(item[0])):
        items.sort(key=lambda row: int(row["frame"]))
        selected = temporal_spread_sample(items, args.max_samples_per_tracklet)
        if len(selected) < args.min_samples_for_clustering:
            skipped_too_few += 1
            continue
        crops = [cv2.imread(row["crop_path"]) for row in selected]
        features = extractor.extract_crops(crops)
        embeddings = [feature["visual_embedding"] for feature in features if feature.get("visual_embedding") is not None]
        if len(embeddings) < args.min_samples_for_clustering:
            skipped_too_few += 1
            continue
        matrix = np.asarray(embeddings, dtype=np.float64)
        labels = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, metric="cosine").fit_predict(matrix)
        non_noise = sorted(set(labels) - {-1})
        n_clusters = len(non_noise)
        cluster_sizes = {int(label): int((labels == label).sum()) for label in non_noise}
        noise_count = int((labels == -1).sum())
        print(
            f"display_track_id={display_id} samples={len(embeddings)}"
            f" clusters={n_clusters} cluster_sizes={cluster_sizes} noise={noise_count}"
        )
        if n_clusters > 1:
            flagged.append({
                "display_track_id": display_id,
                "samples": len(embeddings),
                "clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "sampled_frames": [int(row["frame"]) for row in selected],
            })

    print()
    print(f"tracklets_evaluated={len(by_display) - skipped_too_few} skipped_too_few_samples={skipped_too_few}")
    print(f"flagged_as_mixed_identity={len(flagged)}")
    for row in flagged:
        print(f"  {json.dumps(row)}")


def temporal_spread_sample(items, max_samples):
    if max_samples <= 0 or len(items) <= max_samples:
        return items
    indices = [round(index * (len(items) - 1) / (max_samples - 1)) for index in range(max_samples)]
    seen = set()
    selected = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        selected.append(items[index])
    return selected


if __name__ == "__main__":
    main()
