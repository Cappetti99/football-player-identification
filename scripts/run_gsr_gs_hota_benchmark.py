#!/usr/bin/env python3
"""Run FT inference (jersey OCR + region CTC + TVCalib) and GS-HOTA evaluation
per sequence, for one jersey decision-policy arm, across a manifest of GSR
sequences.

Companion to scripts/run_gsr_detection_tracking_benchmark.py, which
deliberately disables jersey OCR (enforce_benchmark_profile) to isolate
detection/tracking. This script does the opposite: jersey OCR and the region
CTC recognizer stay on, TVCalib is required (GS-HOTA's LocSim needs pitch
error within its tau=5m tolerance, unreachable with the automatic
field-quad fallback -- see docs/additional_thesis_metrics.md), and the base
config selects which source decides the jersey number:

    --arm a   SAR primary, region CTC fallback   (configs/gsr_jersey_pipeline_baseline_apply_v1.yaml)
    --arm b   region CTC primary, SAR fallback   (configs/gsr_jersey_pipeline_ctc_primary_apply_v1.yaml)

Run both arms on the same manifest, then compare with
scripts/aggregate_gsr_gs_hota_benchmark.py.

    python3 scripts/run_gsr_gs_hota_benchmark.py \
        --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \
        --arm a \
        --model-path best_yolo26x_gsr_light.pt \
        --tvcalib-root evaluation_outputs/tvcalib_gsr_val_10s \
        --resume
"""

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


ARM_CONFIGS = {
    "a": "configs/gsr_jersey_pipeline_baseline_apply_v1.yaml",
    "b": "configs/gsr_jersey_pipeline_ctc_primary_apply_v1.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arm", required=True, choices=sorted(ARM_CONFIGS))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video-dir", default="input_videos/soccernet_gsr")
    parser.add_argument("--artifacts-root", default="artifacts/gs_hota_benchmark")
    parser.add_argument("--outputs-root", default="output_videos/gs_hota_benchmark")
    parser.add_argument("--evaluation-root", default="evaluation_outputs/gs_hota_benchmark")
    parser.add_argument("--run-name", default=None, help="defaults to arm_<a|b>")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--tvcalib-root",
        type=Path,
        default=None,
        help="Root containing <sequence>/per_sample_output.json. Omit to fall back "
        "to the automatic field-quad calibration, which is far less accurate "
        "(~29 m vs ~1.1 m median error): GS-HOTA is then meaningless, since its "
        "similarity tolerates only a few metres, but HOTA, MOTA and the jersey "
        "metrics remain valid.",
    )
    args = parser.parse_args()

    run_name = args.run_name or f"arm_{args.arm}"
    config_path = ARM_CONFIGS[args.arm]

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
    artifacts_root = Path(args.artifacts_root) / run_name
    outputs_root = Path(args.outputs_root) / run_name
    evaluation_root = Path(args.evaluation_root) / run_name
    for path in (video_dir, artifacts_root, outputs_root, evaluation_root):
        path.mkdir(parents=True, exist_ok=True)

    base_config = load_config(config_path)
    provenance = {
        "arm": args.arm,
        "config": str(Path(config_path).resolve()),
        "resolved_config_sha256": canonical_sha256(base_config),
        "decision_policy": base_config["decision_policy"]["jersey_number"],
        "run_name": run_name,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "max_frames": args.max_frames,
        "tvcalib_root": str(args.tvcalib_root.resolve()) if args.tvcalib_root else None,
        "calibration_source": "tvcalib" if args.tvcalib_root else "field_quad_fallback",
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

        tvcalib_path = None
        if args.tvcalib_root is not None:
            tvcalib_path = (args.tvcalib_root / sequence / "per_sample_output.json").resolve()
            if not tvcalib_path.is_file():
                raise FileNotFoundError(f"TVCalib output not found for {sequence}: {tvcalib_path}")

        print(f"\nGS-HOTA arm {args.arm.upper()} {index}/{len(entries)}: {sequence}", flush=True)
        if args.resume and summary_path.is_file():
            print(f"Reuse completed evaluation: {summary_path}", flush=True)
            provenance["sequences"].append({"sequence": sequence, "status": "reused"})
            continue

        build_video(Path(entry["frames_dir"]), video_path, max_frames=args.max_frames)
        if not (args.resume and tracklets_path.is_file() and detections_path.is_file()):
            config = load_config(config_path)
            config.update({
                "model_path": str(model_path),
                "video_path": str(video_path),
                "output_path": str(output_path),
                "artifacts_dir": str(artifact_dir),
                "max_frames": args.max_frames,
                "roster_path": None,
            })
            config.setdefault("export", {})["save_detections_csv"] = True
            if tvcalib_path is not None:
                config.setdefault("calibration", {}).update({"enabled": True, "auto": False})
                config["calibration"].setdefault("tvcalib", {}).update({
                    "enabled": True,
                    "path": str(tvcalib_path),
                    "per_frame": True,
                    "coordinate_system": "tvcalib_centered",
                    "frame_offset": 0,
                    "nearest_frame": True,
                    "max_frame_gap": 75,
                })
            else:
                config.setdefault("calibration", {}).update({"enabled": True, "auto": True})
                config["calibration"].setdefault("tvcalib", {})["enabled"] = False
            config.setdefault("wandb", {})["enabled"] = bool(args.wandb)
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
    print(f"\nGS-HOTA BENCHMARK ARM {args.arm.upper()} COMPLETE", flush=True)


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
