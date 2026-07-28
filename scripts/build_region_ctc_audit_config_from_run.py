#!/usr/bin/env python3
"""Build a new, standalone config for a jersey_region_ctc_audit-enabled run
that otherwise reproduces an existing run's tracking/linking behavior
exactly, by reading that run's already-fully-resolved config out of its
run_manifest.json (no need to locate/re-chain the original base_config
file) and layering the region-CTC audit block on top.

This keeps display_track_id/scene_segment_id numbering aligned with the
existing run (and therefore with ground truth already built against it),
since nothing about detection/tracking/linking is changed -- only an
additional, audit-only stage is turned on.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.tracking.yolo_bytetrack import YoloByteTracker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, help="e.g. artifacts/costume-video/Inter-Juve_scenecuts_..._1200f")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--roster-path", required=True)
    parser.add_argument("--ctc-checkpoint", default="/home/cappetti/FT/models/jersey_number_ctc_sjn_to_gsr_v1.pth")
    parser.add_argument("--ctc-checkpoint-sha256", default="11c6356895c94e734b795d9ca2403580187d1d726c5aada51273738042c7c1c5")
    parser.add_argument("--detector-checkpoint", default="/home/cappetti/FT/runs/jersey_number_region/yolo26s_smoke_v1/weights/best.pt")
    parser.add_argument("--detector-checkpoint-sha256", default="ec3c4d06df75aff92f49779827e6cff96d99b84da496ced8a76ecfbb12c02ae4")
    parser.add_argument(
        "--max-crops-per-tracklet", type=int, default=5,
        help="cap on exported crops per tracklet; raise to widen the verified pseudo-labeling pool",
    )
    parser.add_argument(
        "--min-frame-gap", type=int, default=5,
        help="minimum frame spacing between exported crops of the same tracklet",
    )
    args = parser.parse_args()
    if args.max_crops_per_tracklet < 1:
        raise ValueError("--max-crops-per-tracklet must be positive")
    if args.min_frame_gap < 1:
        raise ValueError("--min-frame-gap must be positive")

    source_run_dir = Path(args.source_run_dir).resolve()
    metadata_dir = source_run_dir / "metadata"
    manifest_path = next(metadata_dir.glob("*_run_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest["config"])

    # The resolved config was captured by an older run and can carry tracker
    # kwargs that no longer exist in the current YoloByteTracker (config
    # schemas drift over time). Drop anything the current constructor
    # doesn't accept rather than changing tracking behavior in any other way.
    accepted_tracking_keys = set(inspect.signature(YoloByteTracker.__init__).parameters) - {"self"}
    tracking_cfg = dict(config.get("tracking", {}))
    dropped = {key: value for key, value in tracking_cfg.items() if key not in accepted_tracking_keys}
    if dropped:
        print(f"dropping obsolete tracking config keys not accepted by the current tracker: {dropped}")
    config["tracking"] = {key: value for key, value in tracking_cfg.items() if key in accepted_tracking_keys}

    # Fresh, non-mutating audit stage. Everything else in `config` (detector,
    # tracker, linking, scene_cuts, etc.) is left byte-for-byte as resolved
    # for the source run, so display_track_id/scene_segment_id numbering
    # matches what the ground truth was built against.
    config["jersey_region_ctc_audit"] = {
        "enabled": True,
        "mode": "audit",
        "audit_only": True,
        "ctc_checkpoint": args.ctc_checkpoint,
        "ctc_checkpoint_sha256": args.ctc_checkpoint_sha256,
        "detector_checkpoint": args.detector_checkpoint,
        "detector_checkpoint_sha256": args.detector_checkpoint_sha256,
        "detector_confidence": 0.25,
        "box_padding": 0.25,
        "batch_size": 64,
        "detector_batch_size": 8,
        "device": "cuda",
        "detector_device": "0",
        "min_override_confidence": 0.90,
        "fusion_preview_enabled": False,
        "max_crops_per_tracklet": args.max_crops_per_tracklet,
        "min_frame_gap": args.min_frame_gap,
        "roster_reranking_enabled": True,
        "roster_reranking_top_k": 5,
    }
    config["roster_path"] = args.roster_path

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# Auto-generated from {manifest_path} to add jersey_region_ctc_audit\n"
        f"# without changing any detection/tracking/linking behavior.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
