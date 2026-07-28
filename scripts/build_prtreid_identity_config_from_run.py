#!/usr/bin/env python3
"""Build a new, standalone config that reproduces an existing run's
detection/tracking/linking behavior exactly (read from that run's already-
resolved config in run_manifest.json, same approach as
build_region_ctc_audit_config_from_run.py), with only visual.embedding_mode
switched to "prtreid" and prtreid.enabled turned on.

Motivation: the roster's player.visual_embedding is now populated with
PRTReID vectors (scripts/build_roster_visual_profiles.py), but the identity
gate's visual_similarity/strong_combined path also needs the *tracklet* side
of the comparison to be a PRTReID vector, not the default "hsv"/
"hsv_lab_gradient" color descriptor computed by visual.embedding_mode=hsv
(see ft/features/visual.py -- VisualFeatureExtractor). hungarian.py's
visual_distance() silently returns None on a shape mismatch, so mixing a
256-dim PRTReID roster profile with an hsv-mode tracklet embedding does not
crash, it just never engages -- checked directly against
Inter-Atalanta_identity_evidence_v1_resetbytetrack_full's manifest before
writing this: reid_status was "not_requested", embedding_mode "hsv_lab_gradient".

IMPORTANT: linking.embedding_mode is a *different* config section (used by
the tracklet linker to merge fragmented tracks) and is deliberately left
untouched here -- changing it would alter tracking/linking behavior and
break the display_track_id/scene_segment_id alignment the frozen Int-Ata/
Inter-Juve/Inter-Atalanta ground truth was built against. Only
visual.embedding_mode (consumed by VisualFeatureExtractor for the identity
gate's visual cost/gate, not by the linker) changes.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run-dir", required=True, help="e.g. artifacts/costume-video/Inter-Atalanta_identity_evidence_v1_resetbytetrack_full")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--weights-path", default="models/reid/prtreid-soccernet-baseline.pth.tar")
    parser.add_argument("--hrnet-pretrained-path", default="models/reid")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    source_run_dir = Path(args.source_run_dir).resolve()
    metadata_dir = source_run_dir / "metadata"
    manifest_path = next(metadata_dir.glob("*_run_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    config = dict(manifest["config"])

    # Same schema-drift safety net as build_region_ctc_audit_config_from_run.py:
    # the resolved config was captured by an older run and can carry tracker
    # kwargs the current YoloByteTracker no longer accepts.
    accepted_tracking_keys = set(inspect.signature(YoloByteTracker.__init__).parameters) - {"self"}
    tracking_cfg = dict(config.get("tracking", {}))
    dropped = {key: value for key, value in tracking_cfg.items() if key not in accepted_tracking_keys}
    if dropped:
        print(f"dropping obsolete tracking config keys not accepted by the current tracker: {dropped}")
    config["tracking"] = {key: value for key, value in tracking_cfg.items() if key in accepted_tracking_keys}

    config["visual"] = {
        **config.get("visual", {}),
        "embedding_mode": "prtreid",
    }
    config["prtreid"] = {
        **config.get("prtreid", {}),
        "enabled": True,
        "weights_path": args.weights_path,
        "hrnet_pretrained_path": args.hrnet_pretrained_path,
        "device": args.device,
    }

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# Auto-generated from {manifest_path}: identical detection/tracking/\n"
        f"# linking/identity-gate thresholds, only visual.embedding_mode switched\n"
        f"# to prtreid so the identity gate's visual_similarity/strong_combined\n"
        f"# path has a tracklet-side embedding compatible with the roster's\n"
        f"# newly-populated player.visual_embedding profiles.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
