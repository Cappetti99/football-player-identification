import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ft.utils.geometry import bbox_height, clip_bbox


class ArtifactExporter:
    """Persist crops and metadata rows used by downstream identification."""

    def __init__(self, artifacts_dir, video_id, progress_every=5000, save_crops=True):
        self.artifacts_dir = Path(artifacts_dir)
        self.video_id = video_id
        self.progress_every = int(progress_every or 0)
        self.save_crops = bool(save_crops)
        self.metadata_dir = self.artifacts_dir / "metadata"
        self.crops_dir = self.artifacts_dir / "crops" / video_id
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def export_tracklets(self, frames, tracks, stage="tracklets"):
        rows = []
        frame_groups = tracks.get("players", [])
        print(
            f"FT export {stage}: start frames={len(frame_groups)}",
            flush=True,
        )
        for frame_num, frame_tracks in enumerate(frame_groups):
            frame = frames[frame_num]
            for raw_track_id, track in sorted(frame_tracks.items()):
                bbox = clip_bbox(track["bbox"], frame)
                crop_path = self._save_crop(frame, frame_num, raw_track_id, bbox)
                # Keep both raw and display IDs: raw_track_id is useful for
                # tracker debugging, while display_track_id is the semantic
                # identity surface after linking/splitting.
                row = {
                    "video_id": self.video_id,
                    "frame": int(frame_num),
                    "track_id": int(raw_track_id),
                    "raw_track_id": int(track.get("raw_track_id", raw_track_id)),
                    "previous_display_track_id": track.get("previous_display_track_id"),
                    "jersey_link_previous_display_track_id": track.get("jersey_link_previous_display_track_id"),
                    "display_track_id": int(track.get("display_track_id", raw_track_id)),
                    "bbox": bbox,
                    "role_detection": track.get("role_detection"),
                    "team_id": track.get("team"),
                    "team_confidence": float(track.get("team_confidence", 0.0)),
                    "team_evidence": track.get("team_evidence", {}),
                    "previous_team_evidence": track.get("previous_team_evidence"),
                    "frame_team_conflict": bool(track.get("frame_team_conflict", False)),
                    "display_split": track.get("display_split"),
                    "frame_team_id": track.get("frame_team"),
                    "frame_team_confidence": float(track.get("frame_team_confidence", 0.0)),
                    "frame_team_margin": float(track.get("frame_team_margin", 0.0)),
                    "frame_team_evidence": track.get("frame_team_evidence", {}),
                    "semantic_group_id": track.get("semantic_group_id"),
                    "semantic_group": track.get("semantic_group"),
                    "position_image": to_float_list(track.get("position")),
                    "position_pitch": to_float_list(track.get("position_pitch")),
                    "crop_path": str(crop_path) if crop_path else None,
                    "crop_quality": crop_quality(bbox, frame),
                    "jersey_number": track.get("jersey_number"),
                    "jersey_confidence": track.get("jersey_confidence", 0.0),
                    "jersey_head_confidence": track.get("jersey_head_confidence"),
                    "jersey_winner_margin": track.get("jersey_winner_margin"),
                    "jersey_winner_score_ratio": track.get("jersey_winner_score_ratio"),
                    "jersey_votes": track.get("jersey_votes", 0),
                    "jersey_roster_filter": track.get("jersey_roster_filter"),
                    "jersey_candidates": track.get("jersey_candidates"),
                    "jersey_distribution": track.get("jersey_distribution"),
                    "jersey_roster_mass": float(track.get("jersey_roster_mass", 0.0)),
                    "jersey_constraint": track.get("jersey_constraint"),
                    "visual_embedding": track.get("visual_embedding"),
                    "player_id": track.get("player_id", "unknown"),
                    "player_name": track.get("player_name", "unknown"),
                    "identity_confidence": float(track.get("identity_confidence", 0.0)),
                    "identity_evidence": track.get("identity_evidence", {}),
                    "referee_like_score": float(track.get("referee_like_score", 0.0)),
                    "referee_like_color": track.get("referee_like_color"),
                    "referee_palette_match": bool(track.get("referee_palette_match", False)),
                    "goalkeeper_like_score": float(track.get("goalkeeper_like_score", 0.0)),
                    "goalkeeper_like_team": track.get("goalkeeper_like_team"),
                    "goalkeeper_like_color": track.get("goalkeeper_like_color"),
                    "goalkeeper_palette_match": bool(track.get("goalkeeper_palette_match", False)),
                }
                rows.append(row)
                if self.progress_every > 0 and len(rows) % self.progress_every == 0:
                    print(
                        f"FT export {stage}: rows={len(rows)} frame={frame_num + 1}/{len(frame_groups)}",
                        flush=True,
                    )
        print(f"FT export {stage}: writing metadata rows={len(rows)}", flush=True)
        self.write_json(rows, self.metadata_dir / f"{self.video_id}_tracklets.json")
        self.write_csv(rows, self.metadata_dir / f"{self.video_id}_tracklets.csv")
        print(f"FT export {stage}: done rows={len(rows)}", flush=True)
        return rows

    def _save_crop(self, frame, frame_num, track_id, bbox):
        if not self.save_crops:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        path = self.crops_dir / f"track_{int(track_id):04d}_frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(path), crop)
        return path

    @staticmethod
    def write_json(payload, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def write_csv(rows, path):
        if not rows:
            return
        fields = list(rows[0].keys())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                out = {}
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        # CSV stays spreadsheet-friendly while preserving rich
                        # diagnostics as JSON strings for later inspection.
                        out[key] = json.dumps(value)
                    else:
                        out[key] = value
                writer.writerow(out)


def crop_quality(bbox, frame):
    height, width = frame.shape[:2]
    bw = max(0, bbox[2] - bbox[0])
    bh = max(0, bbox[3] - bbox[1])
    # This score ranks crops for OCR sampling. It favors visible, non-border
    # players without pretending to be a learned image-quality model.
    area_score = min(1.0, (bw * bh) / float(width * height) / 0.02)
    border_penalty = 0.4 if bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 1 else 0.0
    height_bonus = min(0.2, bbox_height(bbox) / 500.0)
    return max(0.0, float(area_score + height_bonus - border_penalty))


def to_float_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(v) for v in value]


def write_table(rows, path):
    ArtifactExporter.write_csv(rows, Path(path))


def write_json(payload, path):
    ArtifactExporter.write_json(payload, Path(path))
