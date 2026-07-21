#!/usr/bin/env python3
"""Rewrite position_pitch in a copy of FT tracklets using TVCalib outputs."""

import argparse
import ast
import csv
import json
from pathlib import Path

from ft.calibration.pitch_transform import PitchTransform
from ft.calibration.tvcalib_adapter import load_tvcalib_homography_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--tvcalib", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--frame-offset", type=int, default=1)
    parser.add_argument("--max-frame-gap", type=int, default=13)
    parser.add_argument("--coordinate-system", default="tvcalib_centered")
    parser.add_argument("--invert", action="store_true")
    args = parser.parse_args()

    homographies = load_tvcalib_homography_map(
        args.tvcalib,
        coordinate_system=args.coordinate_system,
        invert=args.invert,
        frame_offset=args.frame_offset,
    )
    transform = PitchTransform(
        homographies_by_frame=homographies,
        source=f"tvcalib:{Path(args.tvcalib).resolve()}",
        nearest_frame=True,
        max_frame_gap=args.max_frame_gap,
    )
    rows, fields = read_csv(Path(args.tracklets))
    transformed = missing_position = unavailable_frame = 0
    for row in rows:
        position = vector(row.get("position")) or bbox_bottom_middle(vector(row.get("bbox")))
        if position is None:
            row["position_pitch"] = ""
            missing_position += 1
            continue
        pitch = transform.transform_point(position, frame_index=int(row["frame"]))
        if pitch is None:
            row["position_pitch"] = ""
            unavailable_frame += 1
            continue
        row["position_pitch"] = json.dumps(pitch)
        transformed += 1
    if "position_pitch" not in fields:
        fields.append("position_pitch")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    diagnostics = {
        "mode": "offline_copy",
        "mutates_source": False,
        "source": f"tvcalib:{Path(args.tvcalib).resolve()}",
        "source_tracklets": str(Path(args.tracklets).resolve()),
        "output_tracklets": str(output.resolve()),
        "tvcalib": str(Path(args.tvcalib).resolve()),
        "coordinate_system": args.coordinate_system,
        "invert": args.invert,
        "frame_offset": args.frame_offset,
        "max_frame_gap": args.max_frame_gap,
        "homography_frames": sorted(homographies),
        "rows": len(rows),
        "transformed": transformed,
        "missing_image_position": missing_position,
        "unavailable_calibration_frame": unavailable_frame,
        "transform": transform.diagnostics(),
    }
    diagnostics_path = Path(args.diagnostics)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: diagnostics[key] for key in (
        "rows", "transformed", "missing_image_position", "unavailable_calibration_frame",
        "frame_offset", "max_frame_gap")}, indent=2))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def vector(value):
    if value is None or (isinstance(value, str) and value in {"", "None", "null"}):
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_bottom_middle(bbox):
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, y2]


if __name__ == "__main__":
    main()
