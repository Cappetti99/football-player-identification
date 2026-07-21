#!/usr/bin/env python3
"""Run corrected pitch evaluation and pitch-link audit over a GSR manifest."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/detection_tracking")
    parser.add_argument("--output-root", default="evaluation_outputs/calibration_link_audit")
    parser.add_argument("--detection-confidence", type=float, default=0.12)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-gap", type=int, default=90)
    parser.add_argument("--max-speed-mps", type=float, default=12.0)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root) / args.run_name
    completed = []
    for index, entry in enumerate(manifest.get("sequences") or [], start=1):
        sequence = entry["sequence"]
        metadata = Path(args.artifacts_root) / args.run_name / sequence / "metadata"
        tracklets = metadata / f"{sequence}_tracklets.csv"
        detections = metadata / f"{sequence}_detections.csv"
        calibration = metadata / f"{sequence}_calibration.json"
        evaluation = output_root / sequence / "evaluation"
        audit = output_root / sequence / "link_audit"
        print(f"PITCH AUDIT {index}/{len(manifest['sequences'])}: {sequence}", flush=True)
        evaluate_command = [
            sys.executable, str(script_dir / "evaluate_ft_gsr.py"),
            "--labels", entry["labels"], "--tracklets", str(tracklets),
            "--detections", str(detections), "--output-dir", str(evaluation),
            "--detection-confidence-threshold", str(args.detection_confidence),
            "--gt-pitch-coordinate-system", "soccernet_centered",
        ]
        if args.max_frames is not None:
            evaluate_command.extend(["--max-frames", str(args.max_frames)])
        subprocess.run(evaluate_command, check=True)
        subprocess.run([
            sys.executable, str(script_dir / "audit_pitch_linking.py"),
            "--tracklets", str(tracklets),
            "--frame-matches", str(evaluation / "frame_matches.csv"),
            "--calibration", str(calibration), "--output-dir", str(audit),
            "--max-gap", str(args.max_gap), "--max-speed-mps", str(args.max_speed_mps),
        ], check=True)
        completed.append(sequence)
    summary = {
        "mode": "audit_only", "run_name": args.run_name,
        "sequence_count": len(completed), "sequences": completed,
        "gt_boundary": "GT is used only by evaluate_ft_gsr and offline link labels",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PITCH LINK AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
