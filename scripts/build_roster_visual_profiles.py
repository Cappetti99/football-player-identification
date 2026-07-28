#!/usr/bin/env python3
"""Build roster visual_embedding profiles from vetted anchor tracklets in one
video's crops, and write them into another video's roster JSON.

Motivation: ft/identity/hungarian.py's strong_combined gate / visual_similarity
cost component are structurally inert on every video processed so far because
no roster ships player.visual_embedding (see handoff.md, 2026-07-24(3) and
2026-07-25). Inter-Juve.json and Inter-Atalanta.json share 20 Inter players
with identical player_id (same club, two different matches) -- confidently
identified crops from one match can seed a visual profile reused in the other,
entirely in-domain (broadcast crops, same PRTReID embedding space), no
external photos or licensing concerns.

Anchor tracklets below were vetted by hand against Inter-Juve's identity gate
report (winner_margin, tracklet_frames, reliable_jersey status) -- see
handoff.md 2026-07-25. Track 1 (Zielinski) is deliberately excluded: its OCR
winner was "2" (Dumfries' number), not "7", an unresolved anomaly not safe to
use as a visual anchor.

Matched by player name, not player_id: checking Inter-Juve.json vs
Inter-Atalanta.json directly showed player_id is NOT stable across the two
rosters for every player (e.g. Dumfries is "team1_02" in one,
"team1_sub_02" in the other -- likely starter vs substitute numbering).
Name is what's actually consistent between the two files.

This script must run in the prtreid-ft conda env (PRTReID's dependencies are
not installed in jersey-yolo-ocr, see handoff.md 2026-07-22(8)).

Run from ~/FT:
    conda activate prtreid-ft
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT python3 \
      scripts/build_roster_visual_profiles.py \
      --source-run Inter-Juve_identity_evidence_patch_1200f \
      --source-video-id Inter-Juve \
      --target-roster costume-video/Inter-Atalanta/Inter-Atalanta.json \
      --dry-run   # inspect the plan first; drop --dry-run to actually write
"""
import argparse
import glob
import json
from pathlib import Path

import cv2

from ft.features.prtreid import PRTReIDFeatureExtractor
from ft.features.visual import mean_embedding

# player name -> Inter-Juve display_track_ids to pool crops from. Thuram is
# included for completeness (in case a future target roster has him) even
# though Inter-Atalanta.json does not list him -- the script reports and
# skips names with no match rather than failing.
ANCHORS = {
    "Stefan de Vrij": {"track_ids": [5, 27, 35]},
    "Denzel Dumfries": {"track_ids": [30]},
    "Marcus Thuram": {"track_ids": [9]},
}
MAX_CROPS_PER_TRACK = 60


def sample(paths, limit):
    paths = sorted(paths)
    if len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def load_crops(crops_dir, video_id, track_id, limit):
    pattern = str(Path(crops_dir) / video_id / f"players_track_{int(track_id):04d}_frame_*.jpg")
    paths = sample(glob.glob(pattern), limit)
    crops = []
    for path in paths:
        image = cv2.imread(path)
        if image is not None and image.size:
            crops.append(image)
    return paths, crops


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-video-id", default="Inter-Juve")
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--target-roster", required=True)
    parser.add_argument("--weights-path", default="models/reid/prtreid-soccernet-baseline.pth.tar")
    parser.add_argument("--hrnet-pretrained-path", default="models/reid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, extract nothing, write nothing")
    args = parser.parse_args()

    crops_dir = Path(args.artifacts_root) / args.source_run / "crops"

    target_path = Path(args.target_roster)
    roster = json.loads(target_path.read_text(encoding="utf-8"))
    names_in_target = {entry.get("name") for entry in roster}

    plan = {}
    for name, info in ANCHORS.items():
        if name not in names_in_target:
            print(f"{name}: SKIP, not present in {target_path} (no matching name)")
            continue
        all_paths = []
        for track_id in info["track_ids"]:
            paths, _ = load_crops(crops_dir, args.source_video_id, track_id, MAX_CROPS_PER_TRACK)
            all_paths.extend(paths)
        if not all_paths:
            print(f"{name}: SKIP, 0 crops found for tracks={info['track_ids']} under {crops_dir}")
            continue
        plan[name] = all_paths
        print(f"{name}: tracks={info['track_ids']} crops_found={len(all_paths)}")

    if not plan:
        raise SystemExit("no anchor has both a name match in the target roster and crops on disk -- nothing to do")

    if args.dry_run:
        print("\n--dry-run: stopping before extraction/write.")
        return

    extractor = PRTReIDFeatureExtractor(
        enabled=True,
        weights_path=args.weights_path,
        hrnet_pretrained_path=args.hrnet_pretrained_path,
        device=args.device,
    )

    embeddings = {}
    for name, paths in plan.items():
        crops = [image for image in (cv2.imread(path) for path in paths) if image is not None and image.size]
        features = extractor.extract_crops(crops)
        vectors = [feature.get("visual_embedding") for feature in features]
        profile = mean_embedding(vectors)
        if profile is None:
            print(f"{name}: FAILED to build a profile ({len(crops)} crops read)")
            continue
        embeddings[name] = profile
        print(f"{name}: profile built from {len(crops)} crops, dim={len(profile)}")

    if not embeddings:
        raise SystemExit("no embeddings were built -- nothing to write")

    updated = 0
    for entry in roster:
        name = entry.get("name")
        if name in embeddings:
            entry["visual_embedding"] = embeddings[name]
            entry.setdefault("metadata", {})["visual_embedding_source"] = f"{args.source_video_id}:{args.source_run}"
            updated += 1
    if updated != len(embeddings):
        print(f"WARNING: built {len(embeddings)} profiles but only matched {updated} roster entries in {target_path}")
    target_path.write_text(json.dumps(roster, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {updated} visual_embedding profiles into {target_path}")


if __name__ == "__main__":
    main()
