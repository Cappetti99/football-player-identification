import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ft.utils.geometry import bbox_height, clip_bbox


class ArtifactExporter:
    """Persist crops and metadata rows used by downstream identification."""

    def __init__(
        self,
        artifacts_dir,
        video_id,
        progress_every=5000,
        save_crops=True,
        deduplicate_crops=True,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.video_id = video_id
        self.progress_every = int(progress_every or 0)
        self.save_crops = bool(save_crops)
        self.deduplicate_crops = bool(deduplicate_crops)
        self.metadata_dir = self.artifacts_dir / "metadata"
        self.crops_dir = self.artifacts_dir / "crops" / video_id
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self._crop_cache = {}
        self.crop_stats = {
            "save_crops": self.save_crops,
            "deduplicate_crops": self.deduplicate_crops,
            "saved": 0,
            "reused": 0,
            "skipped_empty": 0,
            "write_failed": 0,
            "bytes_written": 0,
        }

    def export_tracklets(self, frames, tracks, stage="tracklets", save_json=True, save_csv=True):
        rows = []
        frame_groups = tracks.get("players", [])
        print(
            f"FT export {stage}: start frames={len(frame_groups)}",
            flush=True,
        )
        for frame_num, frame_tracks in enumerate(frame_groups):
            frame = frames[frame_num]
            for track_group, raw_track_id, track in self._frame_export_items(tracks, frame_num):
                bbox = clip_bbox(track["bbox"], frame)
                crop_path = self._save_crop(frame, frame_num, raw_track_id, bbox, track_group=track_group)
                # Keep both raw and display IDs: raw_track_id is useful for
                # tracker debugging, while display_track_id is the semantic
                # identity surface after linking/splitting.
                row = {
                    "video_id": self.video_id,
                    "track_group": track_group,
                    "frame": int(frame_num),
                    "track_id": int(raw_track_id),
                    "raw_track_id": int(track.get("raw_track_id", raw_track_id)),
                    "previous_display_track_id": track.get("previous_display_track_id"),
                    "jersey_link_previous_display_track_id": track.get("jersey_link_previous_display_track_id"),
                    "display_track_id": int(track.get("display_track_id", raw_track_id)),
                    "identity_tracklet_id": track.get("identity_tracklet_id"),
                    "scene_segment_id": track.get("scene_segment_id"),
                    "scene_cut_boundary": bool(track.get("scene_cut_boundary", False)),
                    "scene_cut_score": track.get("scene_cut_score"),
                    "scene_cut_type": track.get("scene_cut_type"),
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
                    "jersey_segment_index": track.get("jersey_segment_index"),
                    "jersey_roster_filter": track.get("jersey_roster_filter"),
                    "jersey_candidates": track.get("jersey_candidates"),
                    "raw_jersey_distribution": track.get("raw_jersey_distribution"),
                    "jersey_distribution": track.get("jersey_distribution"),
                    "jersey_roster_mass": float(track.get("jersey_roster_mass", 0.0)),
                    "jersey_constraint": track.get("jersey_constraint"),
                    "visual_embedding": track.get("visual_embedding"),
                    "player_id": track.get("player_id", "unknown"),
                    "player_name": track.get("player_name", "unknown"),
                    "identity_confidence": float(track.get("identity_confidence", 0.0)),
                    "identity_evidence": track.get("identity_evidence", {}),
                    "candidate_player_id": track.get("candidate_player_id"),
                    "candidate_player_name": track.get("candidate_player_name"),
                    "candidate_team_id": track.get("candidate_team_id"),
                    "candidate_jersey_number": track.get("candidate_jersey_number"),
                    "candidate_confidence": float(track.get("candidate_confidence", 0.0)),
                    "candidate_cost": track.get("candidate_cost"),
                    "candidate_margin": track.get("candidate_margin"),
                    "candidate_reason": track.get("candidate_reason"),
                    "candidate_evidence": track.get("candidate_evidence", {}),
                    "segment_candidate_player_id": track.get("segment_candidate_player_id"),
                    "segment_candidate_player_name": track.get("segment_candidate_player_name"),
                    "segment_candidate_team_id": track.get("segment_candidate_team_id"),
                    "segment_candidate_jersey_number": track.get("segment_candidate_jersey_number"),
                    "segment_candidate_confidence": float(track.get("segment_candidate_confidence", 0.0)),
                    "segment_candidate_votes": track.get("segment_candidate_votes"),
                    "segment_candidate_reason": track.get("segment_candidate_reason"),
                    "segment_candidate_evidence": track.get("segment_candidate_evidence", {}),
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
        if save_json:
            self.write_json(rows, self.metadata_dir / f"{self.video_id}_tracklets.json")
        if save_csv:
            self.write_csv(rows, self.metadata_dir / f"{self.video_id}_tracklets.csv")
        print(
            f"FT export {stage}: done rows={len(rows)}"
            f" crops_saved={self.crop_stats['saved']}"
            f" crops_reused={self.crop_stats['reused']}",
            flush=True,
        )
        return rows

    @staticmethod
    def _frame_export_items(tracks, frame_num):
        for track_group in ("players", "referees"):
            frame_groups = tracks.get(track_group, [])
            if frame_num >= len(frame_groups):
                continue
            for raw_track_id, track in sorted(frame_groups[frame_num].items()):
                yield track_group, raw_track_id, track

    def _save_crop(self, frame, frame_num, track_id, bbox, track_group="players"):
        if not self.save_crops:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            self.crop_stats["skipped_empty"] += 1
            return None
        cache_key = (track_group, int(frame_num), int(track_id), x1, y1, x2, y2)
        if self.deduplicate_crops and cache_key in self._crop_cache:
            self.crop_stats["reused"] += 1
            return self._crop_cache[cache_key]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.crop_stats["skipped_empty"] += 1
            return None
        path = self.crops_dir / f"{track_group}_track_{int(track_id):04d}_frame_{frame_num:06d}.jpg"
        if not cv2.imwrite(str(path), crop):
            self.crop_stats["write_failed"] += 1
            return None
        self.crop_stats["saved"] += 1
        try:
            self.crop_stats["bytes_written"] += path.stat().st_size
        except OSError:
            pass
        if self.deduplicate_crops:
            self._crop_cache[cache_key] = path
        return path

    def diagnostics(self):
        return {"crops": dict(self.crop_stats)}

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
