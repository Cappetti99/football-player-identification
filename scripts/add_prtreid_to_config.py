#!/usr/bin/env python3
"""Add a `prtreid` block to an existing resolved config (e.g. one produced by
build_region_ctc_audit_config_from_run.py), without touching anything else.

Used to enable PRTReID tracklet-embedding extraction on top of a config that
already reproduces a source run's tracking/linking behavior exactly (so
display_track_id/scene_segment_id numbering stays aligned with ground truth
built against that source run) plus the jersey_region_ctc_audit block.
"""
import argparse
from pathlib import Path

import yaml

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
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_config)
    config = yaml.safe_load(input_path.read_text())
    config["prtreid"] = dict(PRTREID_CONFIG)

    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"# Auto-generated from {input_path} to add prtreid (tracklet-embedding\n"
        f"# extraction, matching Int-Ata's prtreid_conservative_final_1200f config)\n"
        f"# without changing detection/tracking/linking behavior.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
