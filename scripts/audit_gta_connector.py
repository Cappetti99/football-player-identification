#!/usr/bin/env python3
"""Audit-only GTA-style "Connector": propose merges between *different* raw
tracklets (not overlapping in time) whose PRTReID appearance prototypes are
close, and check the proposals against offline GT before writing any merge
logic into the real pipeline.

Motivation: the linker's local candidate window (max_gap/max_distance) was
tested and rejected as a fix for the 44% of GSR link targets whose correct
candidate never enters the pool (see handoff.md). GTA's Connector addresses
this differently -- global tracklet association via mean appearance distance,
independent of temporal proximity. But a near-identical idea (real-time
appearance-based "shadow tracker") was already tried and rejected on
SNGS-025 (309 vs 45 ID switches) because appearance does not discriminate
teammates wearing near-identical kits well. This script measures whether the
*offline, mean-embedding* version of the idea has the same failure mode
before any pipeline change: for every proposed merge, it records whether the
two raw tracks are really the same GT identity, and whether they are on the
same team (the specific weakness already observed).

Zero pipeline mutation: reads already-exported tracklets.csv (bbox/frame/team
per raw_track_id) and re-crops frames on demand for PRTReID embedding
extraction; GT is read only for offline scoring, never fed back into the
proposal logic.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.prtreid import PRTReIDFeatureExtractor  # noqa: E402
from ft.features.visual import prtreid_extractor_config  # noqa: E402

PRTREID_CONFIG = {
    "enabled": True,
    "weights_path": "models/reid/prtreid-soccernet-baseline.pth.tar",
    "hrnet_pretrained_path": "models/reid",
    "device": "auto",
    "batch_size": 32,
    "image_width": 128,
    "image_height": 256,
    "test_embeddings": ["globl"],
    "download_weights": False,
    "role_enabled": False,
    "role_min_confidence": 0.6,
    "role_protect_existing": True,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="GSR pilot manifest (sequence/frames_dir/labels)")
    parser.add_argument("--artifacts-root", required=True, help="e.g. artifacts/detection_tracking/gsr_valid_pilot12_conf012_v1")
    parser.add_argument("--evaluation-root", required=True, help="e.g. evaluation_outputs/detection_tracking/gsr_valid_pilot12_conf012_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge-dist-thres", type=float, default=0.4)
    parser.add_argument("--min-len", type=int, default=100, help="minimum frames per raw track to be considered")
    parser.add_argument("--max-samples-per-track", type=int, default=16)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    artifacts_root = Path(args.artifacts_root)
    evaluation_root = Path(args.evaluation_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    extractor = PRTReIDFeatureExtractor(**prtreid_extractor_config(PRTREID_CONFIG))

    all_pairs = []
    per_sequence_summary = []
    for entry in manifest.get("sequences", []):
        sequence = entry["sequence"]
        tracklets_path = artifacts_root / sequence / "metadata" / f"{sequence}_tracklets.csv"
        matches_path = evaluation_root / sequence / "frame_matches.csv"
        if not tracklets_path.is_file() or not matches_path.is_file():
            print(f"skip {sequence}: missing tracklets or frame_matches")
            continue

        gt_by_raw = read_gt_majority(matches_path)
        tracks = read_tracklets(tracklets_path)
        prototypes, lengths, team_by_raw = build_prototypes(
            tracks, Path(entry["frames_dir"]), extractor, args.max_samples_per_track
        )
        eligible = [raw for raw, length in lengths.items() if length >= args.min_len and raw in prototypes]
        eligible.sort()

        pairs = []
        for i, raw_a in enumerate(eligible):
            frames_a = tracks[raw_a]["frames"]
            for raw_b in eligible[i + 1:]:
                frames_b = tracks[raw_b]["frames"]
                if frames_a & frames_b:
                    continue  # overlapping in time: never the same identity, not a Connector case
                dist = cosine_distance(prototypes[raw_a], prototypes[raw_b])
                if dist > args.merge_dist_thres:
                    continue
                gt_a, gt_b = gt_by_raw.get(raw_a), gt_by_raw.get(raw_b)
                same_identity = None if gt_a is None or gt_b is None else gt_a == gt_b
                same_team = None
                if team_by_raw.get(raw_a) is not None and team_by_raw.get(raw_b) is not None:
                    same_team = team_by_raw[raw_a] == team_by_raw[raw_b]
                row = {
                    "sequence": sequence,
                    "raw_a": raw_a,
                    "raw_b": raw_b,
                    "cosine_distance": dist,
                    "len_a": lengths[raw_a],
                    "len_b": lengths[raw_b],
                    "gt_a": gt_a,
                    "gt_b": gt_b,
                    "same_gt_identity": same_identity,
                    "same_team": same_team,
                }
                pairs.append(row)
                all_pairs.append(row)
        per_sequence_summary.append({
            "sequence": sequence,
            "eligible_tracks": len(eligible),
            "proposed_pairs": len(pairs),
        })
        print(f"{sequence}: eligible_tracks={len(eligible)} proposed_pairs={len(pairs)}")

    scored = [row for row in all_pairs if row["same_gt_identity"] is not None]
    correct = sum(1 for row in scored if row["same_gt_identity"])
    incorrect = [row for row in scored if not row["same_gt_identity"]]
    incorrect_same_team = sum(1 for row in incorrect if row["same_team"])
    incorrect_diff_team = sum(1 for row in incorrect if row["same_team"] is False)

    summary = {
        "mode": "offline_connector_proposal_audit",
        "mutates_tracking": False,
        "gt_usage": "offline scoring only; absent from proposal logic",
        "merge_dist_thres": args.merge_dist_thres,
        "min_len": args.min_len,
        "sequences": per_sequence_summary,
        "total_proposed_pairs": len(all_pairs),
        "gt_scored_pairs": len(scored),
        "correct_merges": correct,
        "incorrect_merges": len(incorrect),
        "precision": (correct / len(scored)) if scored else None,
        "incorrect_merges_same_team": incorrect_same_team,
        "incorrect_merges_different_team": incorrect_diff_team,
        "incorrect_merges_team_unknown": len(incorrect) - incorrect_same_team - incorrect_diff_team,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "proposed_pairs.csv", all_pairs)
    print(json.dumps(summary, indent=2))


def read_gt_majority(path):
    votes = defaultdict(Counter)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = integer(row.get("raw_pred_track_id") or row.get("raw_track_id"))
            gt = row.get("gt_track_id")
            if raw is not None and gt not in {None, "", "None", "null"}:
                votes[raw][str(gt)] += 1
    return {raw: counts.most_common(1)[0][0] for raw, counts in votes.items() if counts}


def read_tracklets(path):
    tracks = defaultdict(lambda: {"frames": set(), "items": [], "team_votes": Counter()})
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = integer(row.get("raw_track_id") or row.get("track_id"))
            frame = integer(row.get("frame"))
            bbox = parse_bbox(row.get("bbox"))
            if raw is None or frame is None or bbox is None:
                continue
            tracks[raw]["frames"].add(frame)
            tracks[raw]["items"].append((frame, bbox))
            team = row.get("team_id")
            if team not in {None, "", "None"}:
                tracks[raw]["team_votes"][team] += 1
    return tracks


def build_prototypes(tracks, frames_dir, extractor, max_samples):
    import cv2

    frame_cache = {}

    def read_frame(frame_num):
        if frame_num not in frame_cache:
            path = frame_path_for(frames_dir, frame_num)
            frame_cache[frame_num] = cv2.imread(str(path)) if path else None
        return frame_cache[frame_num]

    prototypes = {}
    lengths = {}
    team_by_raw = {}
    for raw, data in tracks.items():
        items = sorted(data["items"], key=lambda item: item[0])
        lengths[raw] = len(items)
        team_votes = data["team_votes"]
        team_by_raw[raw] = team_votes.most_common(1)[0][0] if team_votes else None
        sampled = temporal_sample(items, max_samples)
        crops = []
        for frame_num, bbox in sampled:
            frame = read_frame(frame_num)
            if frame is None:
                continue
            crop = crop_from_bbox(frame, bbox)
            if crop is not None:
                crops.append(crop)
        if not crops:
            continue
        features = extractor.extract_crops(crops)
        embeddings = [f["visual_embedding"] for f in features if f.get("visual_embedding") is not None]
        if embeddings:
            prototypes[raw] = mean_vector(embeddings)
    return prototypes, lengths, team_by_raw


_FRAME_INDEX_CACHE = {}


def frame_path_for(frames_dir, frame_num):
    key = str(frames_dir)
    if key not in _FRAME_INDEX_CACHE:
        paths = sorted(
            (p for p in Path(frames_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            key=frame_sort_key,
        )
        _FRAME_INDEX_CACHE[key] = paths
    paths = _FRAME_INDEX_CACHE[key]
    return paths[frame_num] if 0 <= frame_num < len(paths) else None


def frame_sort_key(path):
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def temporal_sample(items, max_samples):
    if max_samples <= 0 or len(items) <= max_samples:
        return items
    step = len(items) / max_samples
    indices = sorted({int(i * step) for i in range(max_samples)})
    return [items[i] for i in indices]


def crop_from_bbox(frame, bbox):
    if frame is None:
        return None
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    height, width = frame.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def parse_bbox(value):
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, list) and len(parsed) == 4:
        return [float(v) for v in parsed]
    return None


def mean_vector(vectors):
    import numpy as np

    return np.mean(np.asarray(vectors, dtype=np.float64), axis=0)


def cosine_distance(a, b):
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(1.0 - (a @ b) / denom)


def integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["sequence"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
