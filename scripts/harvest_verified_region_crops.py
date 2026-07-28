#!/usr/bin/env python3
"""Harvest new region-detector training candidates from an existing run's
jersey_region_ctc_audit.json, keeping only crops that pass three independent
checks:
  1. the detector found a region at all (crop appears in "crops");
  2. the crop's own single-frame CTC prediction (ctc_top1) matches the
     tracklet's final multi-frame consensus number;
  3. that consensus number matches real ground truth (Identity Benchmark V1),
     not just the model agreeing with itself.

This is verified pseudo-labeling, not naive self-training: check 3 anchors
the label to an independent source (roster/manual ground truth), so it
cannot simply reinforce the detector's own existing blind spots. Candidates
are real broadcast-domain crops (Int-Ata/etc.), not GSR -- the domain the
detector needs to generalize to.

Writes nothing back into any pipeline artifact. Prints candidates and, if
--output-dir is given, copies the crop images plus a YOLO-format label file
(one box per image, class 0, normalized xywh) so they are ready to be added
to the detector's training set later -- as a separate, explicit step, not
automatically.
"""
import argparse
import json
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.jersey_region_ctc_audit import tracklet_key  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--benchmark-manifest", default="evaluation_outputs/identity_benchmark_v1_full/benchmark_manifest.json")
    parser.add_argument("--ground-truth-csv", default="evaluation/identity_benchmark_v1_full/ground_truth.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--include-corrected-misreads", action="store_true",
        help=(
            "Also harvest crops where the single-frame CTC prediction disagreed "
            "with the tracklet consensus, as long as that consensus matches "
            "ground truth. These are the recognizer's own frame-level mistakes, "
            "corrected via multi-frame + roster context and anchored to an "
            "independent GT label (not the model's own agreement) -- likely the "
            "highest-value hard examples, tagged separately from clean-agreement "
            "crops for transparency."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    audit = json.loads((run_dir / "metadata" / f"{args.video_id}_jersey_region_ctc_audit.json").read_text())
    crops = audit.get("crops", [])
    standalone = audit.get("standalone_assignments", {})
    roster_preview = audit.get("roster_rerank_preview", {})

    import csv
    with open(args.ground_truth_csv, newline="", encoding="utf-8") as handle:
        ground_truth = {row["item_id"]: row for row in csv.DictReader(handle)}
    manifest = json.loads(Path(args.benchmark_manifest).read_text())
    units_by_id = {unit["item_id"]: unit for unit in manifest.get("identity_units", [])}

    # Build (display_track_id, scene_segment_id) -> gt_jersey_number from the
    # already-labeled benchmark, restricted to this video and determinate units.
    gt_by_track_key = {}
    for item_id, unit in units_by_id.items():
        if unit.get("video_id") != args.video_id:
            continue
        row = ground_truth.get(item_id)
        if row is None or row.get("annotation_status") != "determinate":
            continue
        gt_jersey = row.get("gt_jersey_number")
        if not gt_jersey:
            continue
        for member in unit.get("members", []):
            scene_segment_id = member.get("scene_segment_id")
            for display_track_id in member.get("display_track_ids", []):
                gt_by_track_key[tracklet_key(display_track_id, scene_segment_id)] = str(gt_jersey)

    verified = []
    rejected_reasons = {
        "no_gt_for_track": 0,
        "no_tracklet_consensus": 0,
        "ctc_disagrees_with_consensus": 0,
        "consensus_not_ground_truth": 0,
    }
    for crop in crops:
        display_track_id = crop["display_track_id"]
        # crops list only carries plain display_track_id (no scene_segment_id
        # attached at crop level pre-fix); try the compound key via the
        # tracklets dict order used elsewhere is not available here, so fall
        # back to matching by plain id against any gt track key with that id.
        candidate_keys = [key for key in gt_by_track_key if key == str(display_track_id) or key.startswith(f"{display_track_id}#")]
        if not candidate_keys:
            rejected_reasons["no_gt_for_track"] += 1
            continue
        # tracklet-level consensus: prefer roster-reranked preview if present.
        key = candidate_keys[0]
        proposal = standalone.get(key, {})
        preview = roster_preview.get(key, {})
        consensus = preview.get("preview_number") if preview.get("preview_number") is not None else proposal.get("jersey_number")
        if consensus is None:
            rejected_reasons["no_tracklet_consensus"] += 1
            continue
        agrees_with_ctc = str(crop["ctc_top1"]) == str(consensus)
        if not agrees_with_ctc and not args.include_corrected_misreads:
            rejected_reasons["ctc_disagrees_with_consensus"] += 1
            continue
        gt = gt_by_track_key[key]
        if str(consensus) != gt:
            rejected_reasons["consensus_not_ground_truth"] += 1
            continue
        verification_type = "clean_agreement" if agrees_with_ctc else "corrected_misread"
        verified.append({
            **crop,
            "verified_jersey_number": gt,
            "track_key": key,
            "verification_type": verification_type,
        })

    print(f"total_crops_with_region={len(crops)}")
    print(f"verified_candidates={len(verified)}")
    print(f"  clean_agreement={sum(1 for row in verified if row['verification_type'] == 'clean_agreement')}")
    print(f"  corrected_misread={sum(1 for row in verified if row['verification_type'] == 'corrected_misread')}")
    print(f"rejected_reasons={rejected_reasons}")
    for row in verified:
        print(
            f"  crop_path={row['crop_path']} track_key={row['track_key']} "
            f"jersey={row['verified_jersey_number']} type={row['verification_type']} "
            f"region_xyxyn={row['region_xyxyn']}"
        )

    if args.output_dir and verified:
        output = Path(args.output_dir)
        images_dir = output / "images"
        labels_dir = output / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for row in verified:
            source = Path(row["crop_path"])
            target_image = images_dir / source.name
            shutil.copy2(source, target_image)
            x1, y1, x2, y2 = row["region_xyxyn"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            label_path = labels_dir / (source.stem + ".txt")
            label_path.write_text(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")
        print(f"wrote {len(verified)} image+label pairs to {output}")


if __name__ == "__main__":
    main()
