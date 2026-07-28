#!/usr/bin/env python3
"""Report how many scene cuts the current config's detector finds on a set of
videos, without running detection/tracking/OCR.

Motivation: the GSR detection/tracking benchmark's scene_cuts settings are
inherited from a config tuned specifically for the Int-Ata custom broadcast
video (a multi-camera match feed). SoccerNet-GSR clips are short,
single-camera-angle sequences, so it's unknown whether that inherited
threshold ever fires there, and if it does, whether it fires on real camera
cuts or on in-play visual noise (motion blur, a player filling the frame,
a referee walking through). Cheap (no GPU): just video decode + HSV
histograms.
"""
import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.config import load_config  # noqa: E402
from ft.pipeline import scene_cut_config  # noqa: E402
from ft.utils.scene_cuts import detect_scene_cuts, select_tracking_reset_frames  # noqa: E402
from ft.utils.video import read_video  # noqa: E402


def frame_key(path):
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def read_frames_dir(frames_dir, max_frames=None):
    import cv2

    paths = sorted(
        (path for path in Path(frames_dir).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=frame_key,
    )
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No frames: {frames_dir}")
    return [cv2.imread(str(path)) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="config to read scene_cuts.* settings from")
    parser.add_argument("--video-path", action="append", default=[], help="repeatable")
    parser.add_argument("--frames-dir", action="append", default=[], help="repeatable, e.g. a GSR sequence's img1/ dir")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cut_cfg = scene_cut_config(config)
    reset_enabled = bool(config.get("scene_cuts", {}).get("tracking_reset_enabled", False))
    hard_only = bool(config.get("scene_cuts", {}).get("tracking_reset_hard_only", False))
    print(f"config={args.config}")
    print(f"scene_cuts settings: {json.dumps(cut_cfg, indent=2)}")
    print(f"tracking_reset_enabled={reset_enabled} tracking_reset_hard_only={hard_only}")
    print()

    sources = [("video", path) for path in args.video_path] + [("frames_dir", path) for path in args.frames_dir]
    for kind, source in sources:
        frames = (
            read_video(source, max_frames=args.max_frames)
            if kind == "video"
            else read_frames_dir(source, max_frames=args.max_frames)
        )
        diagnostics = detect_scene_cuts(frames, **{**cut_cfg, "enabled": True})
        reset_frames = select_tracking_reset_frames(diagnostics, hard_only=hard_only) if reset_enabled else []
        print(f"{kind}={source} frames={len(frames)}")
        print(f"  cuts_found={len(diagnostics.get('cuts', []))} reset_frames={len(reset_frames)}")
        for cut in diagnostics.get("cuts", [])[:20]:
            print(f"    frame={cut.get('frame')} score={cut.get('score'):.4f} type={cut.get('type')}")
        threshold = float(cut_cfg.get("threshold", 0.0) or 0.0)
        top_scores = diagnostics.get("top_scores", [])
        print(f"  top_scores (threshold={threshold}, hard_cut_threshold={cut_cfg.get('hard_cut_threshold')}):")
        for row in top_scores[:15]:
            flag = " <-- WOULD TRIGGER" if row["score"] >= threshold else ""
            print(
                f"    frame={row['frame']} score={row['score']:.4f} correlation={row['correlation']:.4f}{flag}"
            )
        print()


if __name__ == "__main__":
    main()
