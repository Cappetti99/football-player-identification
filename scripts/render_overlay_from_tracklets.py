#!/usr/bin/env python3
"""Render an overlay video from exported tracklet metadata.

This avoids rerunning YOLO/OCR when a post-processing script changes only
metadata, such as an auxiliary OCR merge. It rebuilds the per-frame track dict
expected by ft.visualization.overlay and writes an MP4 from the original video.
"""

import argparse
import ast
import csv
import json
from pathlib import Path

from ft.features.groups import GROUP_COLORS, player_group_id
from ft.visualization.overlay import draw_overlay


EMPTY = {"", "None", "unknown", None}


def nonempty(value):
    return value not in EMPTY


def parse_jsonish(value, default=None):
    if value in EMPTY:
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return default


def parse_int(value, default=None):
    if value in EMPTY:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value, default=0.0):
    if value in EMPTY:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_bbox(row):
    bbox = parse_jsonish(row.get("bbox"), default=None)
    if bbox is None:
        raise ValueError(f"Missing bbox for frame={row.get('frame')} track={row.get('track_id')}")
    return [int(float(value)) for value in bbox]


def row_track(row):
    team_id = parse_int(row.get("team_id"))
    jersey_number = parse_int(row.get("jersey_number"))
    jersey_confidence = parse_float(row.get("jersey_confidence"))
    jersey_votes = parse_int(row.get("jersey_votes"), default=0) or 0
    identity_confidence = parse_float(row.get("identity_confidence"))
    track = {
        "bbox": row_bbox(row),
        "raw_track_id": parse_int(row.get("raw_track_id"), parse_int(row.get("track_id"), 0)),
        "display_track_id": parse_int(row.get("display_track_id"), parse_int(row.get("track_id"), 0)),
        "identity_tracklet_id": parse_int(row.get("identity_tracklet_id")),
        "role_detection": row.get("role_detection"),
        "team": team_id,
        "team_confidence": parse_float(row.get("team_confidence")),
        "frame_team": parse_int(row.get("frame_team_id")),
        "frame_team_confidence": parse_float(row.get("frame_team_confidence")),
        "semantic_group_id": parse_int(row.get("semantic_group_id")),
        "semantic_group": row.get("semantic_group"),
        "jersey_number": jersey_number,
        "jersey_confidence": jersey_confidence,
        "jersey_head_confidence": parse_float(row.get("jersey_head_confidence")),
        "jersey_winner_margin": parse_float(row.get("jersey_winner_margin")),
        "jersey_votes": jersey_votes,
        "player_id": row.get("player_id") if nonempty(row.get("player_id")) else "unknown",
        "player_name": row.get("player_name") if nonempty(row.get("player_name")) else "unknown",
        "identity_confidence": identity_confidence,
        "referee_like_score": parse_float(row.get("referee_like_score")),
        "referee_like_color": row.get("referee_like_color") if nonempty(row.get("referee_like_color")) else None,
        "referee_palette_match": parse_bool(row.get("referee_palette_match")),
        "goalkeeper_like_score": parse_float(row.get("goalkeeper_like_score")),
        "goalkeeper_like_team": parse_int(row.get("goalkeeper_like_team")),
        "goalkeeper_like_color": row.get("goalkeeper_like_color") if nonempty(row.get("goalkeeper_like_color")) else None,
        "goalkeeper_palette_match": parse_bool(row.get("goalkeeper_palette_match")),
    }
    if jersey_number is not None:
        track["jersey_evidence"] = {
            "confidence": jersey_confidence,
            "head_confidence": parse_float(row.get("jersey_head_confidence")),
            "winner_margin": parse_float(row.get("jersey_winner_margin")),
            "votes": jersey_votes,
        }
    if track["semantic_group_id"] is None:
        track["semantic_group_id"] = player_group_id(track)
    color = GROUP_COLORS.get(track["semantic_group_id"])
    if color:
        track["semantic_group_color"] = color
    return track


def build_tracks(rows, num_frames):
    tracks = {
        "players": [dict() for _ in range(num_frames)],
        "referees": [dict() for _ in range(num_frames)],
        "ball": [dict() for _ in range(num_frames)],
    }
    for row in rows:
        frame = parse_int(row.get("frame"))
        if frame is None or frame < 0 or frame >= num_frames:
            continue
        group = row.get("track_group", "players")
        raw_id = parse_int(row.get("track_id"), default=0)
        track = row_track(row)
        if group == "referees":
            tracks["referees"][frame][raw_id] = track
        elif group == "ball":
            tracks["ball"][frame][raw_id] = track
        else:
            tracks["players"][frame][raw_id] = track
    return tracks


def main():
    from ft.utils.video import read_video, save_video

    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True, type=Path)
    parser.add_argument("--tracklets-csv", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--show-player-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-identity-confidence", action="store_true")
    parser.add_argument("--show-jersey-winner", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = load_rows(args.tracklets_csv)
    max_frame = max((parse_int(row.get("frame"), default=-1) for row in rows), default=-1) + 1
    max_frames = args.max_frames if args.max_frames is not None else max_frame
    frames = read_video(args.video_path, max_frames=max_frames)
    if max_frame > len(frames):
        raise RuntimeError(f"Tracklets reference {max_frame} frames, but video read only {len(frames)} frames")
    tracks = build_tracks(rows, len(frames))
    overlay_config = {
        "show_player_id": bool(args.show_player_id),
        "show_player_id_min_confidence": 0.0,
        "show_identity_confidence": bool(args.show_identity_confidence),
        "show_jersey": True,
        "show_jersey_winner": bool(args.show_jersey_winner),
        "require_ocr_jersey_evidence": False,
    }
    output_frames = draw_overlay(frames, tracks, config=overlay_config)
    save_video(output_frames, args.output_path, fps=args.fps)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
