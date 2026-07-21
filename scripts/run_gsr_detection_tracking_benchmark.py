#!/usr/bin/env python3
"""Run FT inference and offline GSR detection/tracking evaluation per sequence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2

from ft.config import load_config
from ft.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/gsr_detection_tracking_benchmark_v1.yaml")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-dir", default="input_videos/soccernet_gsr")
    parser.add_argument("--artifacts-root", default="artifacts/detection_tracking")
    parser.add_argument("--outputs-root", default="output_videos/detection_tracking")
    parser.add_argument("--evaluation-root", default="evaluation_outputs/detection_tracking")
    parser.add_argument("--run-name", default="baseline_v1")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--tvcalib-root",
        type=Path,
        default=None,
        help="Optional root containing <sequence>/per_sample_output.json.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") == "test" and not args.allow_test:
        raise ValueError(
            "The GSR test split is frozen. Pass --allow-test only for the final locked evaluation."
        )
    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_dir = Path(args.video_dir)
    artifacts_root = Path(args.artifacts_root) / args.run_name
    outputs_root = Path(args.outputs_root) / args.run_name
    evaluation_root = Path(args.evaluation_root) / args.run_name
    for path in (video_dir, artifacts_root, outputs_root, evaluation_root):
        path.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    enforce_benchmark_profile(base_config, wandb=args.wandb)
    provenance = {
        "benchmark": manifest.get("benchmark"),
        "run_name": args.run_name,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config": str(Path(args.config).resolve()),
        "resolved_config_sha256": canonical_sha256(base_config),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "max_frames": args.max_frames,
        "tvcalib_root": str(args.tvcalib_root.resolve()) if args.tvcalib_root else None,
        "gt_boundary": "labels are passed only to the post-inference evaluator",
        "started_unix": time.time(),
        "sequences": [],
    }
    write_json(provenance, evaluation_root / "run_provenance.json")

    script_root = Path(__file__).resolve().parent
    entries = manifest.get("sequences") or []
    for index, entry in enumerate(entries, start=1):
        sequence = entry["sequence"]
        video_path = video_dir / f"{sequence}.mp4"
        artifact_dir = artifacts_root / sequence
        output_path = outputs_root / f"{sequence}.mp4"
        evaluation_dir = evaluation_root / sequence
        summary_path = evaluation_dir / "summary.json"
        tracklets_path = artifact_dir / "metadata" / f"{sequence}_tracklets.csv"
        detections_path = artifact_dir / "metadata" / f"{sequence}_detections.csv"

        print(f"\nBENCHMARK {index}/{len(entries)}: {sequence}", flush=True)
        if args.resume and summary_path.is_file():
            print(f"Reuse completed evaluation: {summary_path}", flush=True)
            provenance["sequences"].append({"sequence": sequence, "status": "reused"})
            continue

        build_video(Path(entry["frames_dir"]), video_path, max_frames=args.max_frames)
        if not (args.resume and tracklets_path.is_file() and detections_path.is_file()):
            config = load_config(args.config)
            enforce_benchmark_profile(config, wandb=args.wandb)
            config.update({
                "model_path": str(model_path),
                "video_path": str(video_path),
                "output_path": str(output_path),
                "artifacts_dir": str(artifact_dir),
                "max_frames": args.max_frames,
                "roster_path": None,
            })
            if args.tvcalib_root is not None:
                tvcalib_path = (
                    args.tvcalib_root
                    / sequence
                    / "per_sample_output.json"
                ).resolve()
                if not tvcalib_path.is_file():
                    raise FileNotFoundError(
                        f"TVCalib output not found for {sequence}: {tvcalib_path}"
                    )
                config.setdefault("calibration", {}).setdefault("tvcalib", {}).update({
                    "enabled": True,
                    "path": str(tvcalib_path),
                })
            run_pipeline(config)
        else:
            print(f"Reuse inference artifacts: {tracklets_path}", flush=True)

        evaluation_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(script_root / "evaluate_ft_gsr.py"),
            "--labels",
            entry["labels"],
            "--tracklets",
            str(tracklets_path),
            "--detections",
            str(detections_path),
            "--output-dir",
            str(evaluation_dir),
            "--detection-confidence-threshold",
            str(base_config["detection"]["confidence"]),
        ]
        if args.max_frames is not None:
            command.extend(["--max-frames", str(args.max_frames)])
        subprocess.run(command, check=True)
        provenance["sequences"].append({
            "sequence": sequence,
            "status": "completed",
            "tracklets": str(tracklets_path.resolve()),
            "detections": str(detections_path.resolve()),
            "summary": str(summary_path.resolve()),
        })
        write_json(provenance, evaluation_root / "run_provenance.json")

    provenance["completed_unix"] = time.time()
    provenance["status"] = "complete"
    write_json(provenance, evaluation_root / "run_provenance.json")
    print("\nDETECTION/TRACKING BENCHMARK COMPLETE", flush=True)


def enforce_benchmark_profile(config, wandb=False):
    """Disable downstream identity evidence while preserving detection/tracking."""
    config.setdefault("number_region", {})["enabled"] = False
    config.setdefault("jersey_ocr", {})["enabled"] = False
    config.setdefault("jersey_frame_selection", {})["enabled"] = False
    config.setdefault("prtreid", {}).update({"enabled": False, "role_enabled": False})
    config.setdefault("prtreid_linking", {})["enabled"] = False
    config.setdefault("prtreid_identity_bridge", {})["enabled"] = False
    config.setdefault("jersey_identity_linking", {})["enabled"] = False
    config.setdefault("identity_propagation", {})["enabled"] = False
    config.setdefault("export", {}).update({
        "save_crops": False,
        "save_detections_json": False,
        "save_detections_csv": True,
        "save_pre_identity_json": False,
        "save_pre_identity_csv": False,
        "save_final_json": False,
        "save_final_csv": True,
    })
    config.setdefault("wandb", {}).update({
        "enabled": bool(wandb),
        "log_artifacts": False,
        "log_video": False,
        "alert_on_finish": False,
        "alert_on_failure": False,
    })


def build_video(frames_dir, output, max_frames=None, fps=25.0):
    frames = sorted(
        (
            path for path in frames_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=frame_key,
    )
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise FileNotFoundError(f"No frames: {frames_dir}")
    if output.is_file():
        capture = cv2.VideoCapture(str(output))
        existing_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if existing_frames == len(frames):
            print(f"Reuse video: {output} frames={existing_frames}", flush=True)
            return
        print(
            f"Rebuild video with wrong length: {output}"
            f" existing={existing_frames} expected={len(frames)}",
            flush=True,
        )
        output.unlink()
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Unreadable frame: {frames[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output}")
    written = 0
    try:
        for path in frames:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Unreadable frame: {path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    print(f"Built video: {output} frames={written}", flush=True)


def frame_key(path):
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
