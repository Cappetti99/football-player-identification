"""Audit-only jersey number-region localization and OCR.

This module deliberately does not mutate track rows or identity state.  Its
outputs are an independent evidence source that can be reviewed before any
future integration with identity assignment.
"""

from collections import Counter, defaultdict
from pathlib import Path
import re

import cv2


class NumberRegionAuditor:
    """Localize candidate number regions and run raw OCR on their crops."""

    def __init__(
        self,
        output_dir,
        frame_interval=5,
        backend="torso_heuristic",
        torso_x_min=0.15,
        torso_x_max=0.85,
        torso_y_min=0.12,
        torso_y_max=0.62,
        padding=0.08,
        min_width=8,
        min_height=8,
        max_regions_per_tracklet=0,
    ):
        self.output_dir = Path(output_dir)
        self.frame_interval = max(1, int(frame_interval))
        self.backend = str(backend or "torso_heuristic").lower()
        self.torso_x_min = float(torso_x_min)
        self.torso_x_max = float(torso_x_max)
        self.torso_y_min = float(torso_y_min)
        self.torso_y_max = float(torso_y_max)
        self.padding = max(0.0, float(padding))
        self.min_width = max(1, int(min_width))
        self.min_height = max(1, int(min_height))
        self.max_regions_per_tracklet = max(0, int(max_regions_per_tracklet or 0))

    def run(self, frames, player_rows, recognizer):
        if self.backend != "torso_heuristic":
            raise ValueError(f"unsupported number-region backend: {self.backend!r}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        selected = self._select_rows(player_rows)
        detections = []
        recognition_rows = []
        for row in selected:
            frame_index = int(row.get("frame", -1))
            if frame_index < 0 or frame_index >= len(frames):
                continue
            detection, recognition_row = self._localize_row(frames[frame_index], row)
            detections.append(detection)
            if recognition_row is not None:
                recognition_rows.append(recognition_row)

        assignments, ocr_diagnostics = recognizer.recognize(recognition_rows)
        raw_by_crop = raw_detections_by_crop(ocr_diagnostics)
        for detection in detections:
            candidates = raw_by_crop.get(str(detection.get("number_region_crop_path")), [])
            best = best_raw_candidate(candidates)
            detection["recognition_raw"] = best.get("text")
            detection["recognition_normalized"] = normalize_partial_number(best.get("text"))
            detection["recognition_confidence"] = float(best.get("confidence", 0.0) or 0.0)
            detection["recognition_source"] = best.get("source")
            detection["recognition_candidates"] = compact_candidates(candidates)

        evidence = build_tracklet_evidence(detections, self.frame_interval)
        recognized = sum(1 for row in detections if row.get("recognition_normalized"))
        diagnostics = {
            "enabled": True,
            "audit_only": True,
            "status": "ok" if recognition_rows else "no_usable_regions",
            "backend": self.backend,
            "frame_interval": self.frame_interval,
            "input_player_rows": len(player_rows),
            "sampled_player_rows": len(selected),
            "localized_regions": len(detections),
            "usable_region_crops": len(recognition_rows),
            "recognized_regions": recognized,
            "tracklet_evidence_rows": len(evidence),
            "ocr": {
                "status": ocr_diagnostics.get("status"),
                "backend": ocr_diagnostics.get("backend"),
                "requested_backend": ocr_diagnostics.get("requested_backend"),
                "message": ocr_diagnostics.get("message"),
                "backend_attempts": ocr_diagnostics.get("backend_attempts", []),
            },
            "identity_mutations": 0,
            "legacy_jersey_mutations": 0,
        }
        return detections, evidence, diagnostics

    def _select_rows(self, rows):
        grouped = defaultdict(list)
        for row in rows:
            if not is_player_row(row):
                continue
            frame = int(row.get("frame", -1))
            if frame < 0 or frame % self.frame_interval != 0:
                continue
            grouped[int(row.get("display_track_id", row.get("raw_track_id", -1)))].append(row)

        selected = []
        for _, items in sorted(grouped.items()):
            items = sorted(items, key=lambda item: int(item.get("frame", 0)))
            if self.max_regions_per_tracklet and len(items) > self.max_regions_per_tracklet:
                items = select_spread(items, self.max_regions_per_tracklet)
            selected.extend(items)
        return selected

    def _localize_row(self, frame, row):
        height, width = frame.shape[:2]
        player_bbox = clip_bbox(row.get("bbox"), width, height)
        display_id = int(row.get("display_track_id", row.get("raw_track_id", -1)))
        frame_index = int(row.get("frame", -1))
        base = {
            "frame": frame_index,
            "raw_track_id": int(row.get("raw_track_id", row.get("track_id", -1))),
            "display_track_id": display_id,
            "team_id": row.get("team_id"),
            "player_bbox": player_bbox,
            "localization_source": self.backend,
            # A geometric proposal has no learned confidence.  Zero prevents it
            # from being mistaken for calibrated detector evidence downstream.
            "number_region_confidence": 0.0,
            "player_number_overlap": 1.0,
            "association_status": "associated",
            "recognition_raw": None,
            "recognition_normalized": None,
            "recognition_confidence": 0.0,
            "recognition_source": None,
            "consecutive_support": 0,
        }
        if player_bbox is None:
            return {**base, "association_status": "invalid_player_bbox", "number_region_bbox": None}, None

        x1, y1, x2, y2 = player_bbox
        player_width = x2 - x1
        player_height = y2 - y1
        region = [
            x1 + self.torso_x_min * player_width,
            y1 + self.torso_y_min * player_height,
            x1 + self.torso_x_max * player_width,
            y1 + self.torso_y_max * player_height,
        ]
        region = pad_and_clip(region, player_bbox, self.padding)
        rx1, ry1, rx2, ry2 = [int(round(value)) for value in region]
        base["number_region_bbox"] = [rx1, ry1, rx2, ry2]
        if rx2 - rx1 < self.min_width or ry2 - ry1 < self.min_height:
            return {**base, "association_status": "region_too_small"}, None

        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return {**base, "association_status": "empty_region_crop"}, None
        crop_path = self.output_dir / f"display_{display_id:04d}_frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(crop_path), crop):
            return {**base, "association_status": "crop_write_failed"}, None

        base["number_region_crop_path"] = str(crop_path)
        recognition_row = {
            **row,
            "bbox": base["number_region_bbox"],
            "crop_path": str(crop_path),
            "crop_quality": float(row.get("crop_quality", 0.0) or 0.0),
            "number_region_source": self.backend,
        }
        return base, recognition_row


def is_player_row(row):
    if row.get("track_group", "players") != "players":
        return False
    return str(row.get("role_detection") or "player").lower() not in {"referee", "ball"}


def clip_bbox(bbox, width, height):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = min(max(0.0, x1), float(width))
    y1 = min(max(0.0, y1), float(height))
    x2 = min(max(0.0, x2), float(width))
    y2 = min(max(0.0, y2), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def pad_and_clip(region, player_bbox, padding):
    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1
    px1, py1, px2, py2 = player_bbox
    return [
        max(px1, x1 - padding * width),
        max(py1, y1 - padding * height),
        min(px2, x2 + padding * width),
        min(py2, y2 + padding * height),
    ]


def select_spread(items, limit):
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)}
    return [items[index] for index in sorted(indices)]


def raw_detections_by_crop(ocr_diagnostics):
    grouped = defaultdict(list)
    for tracklet in (ocr_diagnostics.get("tracklets") or {}).values():
        for detection in tracklet.get("detections", []):
            crop_path = detection.get("crop_path")
            if crop_path:
                grouped[str(crop_path)].append(detection)
    return grouped


def best_raw_candidate(candidates):
    usable = [item for item in candidates if normalize_partial_number(item.get("text"))]
    if not usable:
        return {}
    return max(usable, key=lambda item: float(item.get("confidence", 0.0) or 0.0))


def compact_candidates(candidates):
    best = {}
    for item in candidates:
        text = str(item.get("text") or "").strip()
        normalized = normalize_partial_number(text)
        if not normalized:
            continue
        candidate = {
            "raw": text,
            "normalized": normalized,
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "source": item.get("source"),
            "variant": item.get("variant"),
        }
        previous = best.get((normalized, candidate["source"]))
        if previous is None or candidate["confidence"] > previous["confidence"]:
            best[(normalized, candidate["source"])] = candidate
    return sorted(best.values(), key=lambda item: item["confidence"], reverse=True)


def normalize_partial_number(text):
    value = str(text or "").strip().replace(" ", "")
    if not value:
        return None
    value = re.sub(r"[^0-9?]", "", value)
    if not value or len(value) > 2 or value == "??":
        return None
    if "?" not in value:
        number = int(value)
        return str(number) if 0 <= number <= 99 else None
    return value


def build_tracklet_evidence(detections, frame_interval):
    grouped = defaultdict(list)
    for row in detections:
        grouped[int(row.get("display_track_id", -1))].append(row)

    evidence = []
    for display_id, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: int(item.get("frame", 0)))
        recognized = [row for row in rows if row.get("recognition_normalized")]
        counts = Counter(row["recognition_normalized"] for row in recognized)
        confidence_sums = defaultdict(float)
        for row in recognized:
            confidence_sums[row["recognition_normalized"]] += float(row.get("recognition_confidence", 0.0) or 0.0)
        winner = None
        if counts:
            winner = max(counts, key=lambda value: (counts[value], confidence_sums[value], value))
        max_consecutive = consecutive_support(rows, winner, frame_interval) if winner else 0
        for row in rows:
            row["consecutive_support"] = consecutive_support_at_row(rows, row, frame_interval)
        evidence.append(
            {
                "display_track_id": display_id,
                "team_id": first_nonempty(rows, "team_id"),
                "sampled_regions": len(rows),
                "recognized_regions": len(recognized),
                "frame_diversity": len({int(row.get("frame", -1)) for row in recognized}),
                "winner": winner,
                "winner_votes": int(counts.get(winner, 0)) if winner else 0,
                "winner_mean_confidence": (
                    confidence_sums[winner] / counts[winner] if winner and counts[winner] else 0.0
                ),
                "max_consecutive_support": max_consecutive,
                "raw_distribution": [
                    {
                        "value": value,
                        "votes": int(count),
                        "mean_confidence": confidence_sums[value] / count,
                    }
                    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                ],
                "localization_source": first_nonempty(rows, "localization_source"),
                "audit_only": True,
            }
        )
    return evidence


def consecutive_support(rows, value, frame_interval):
    longest = 0
    current = 0
    previous_frame = None
    for row in rows:
        frame = int(row.get("frame", -1))
        contiguous = previous_frame is not None and frame - previous_frame <= frame_interval * 2
        if row.get("recognition_normalized") == value:
            current = current + 1 if contiguous else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_frame = frame
    return longest


def consecutive_support_at_row(rows, target, frame_interval):
    value = target.get("recognition_normalized")
    if not value:
        return 0
    target_frame = int(target.get("frame", -1))
    run = 0
    previous_frame = None
    for row in rows:
        frame = int(row.get("frame", -1))
        if frame > target_frame:
            break
        contiguous = previous_frame is not None and frame - previous_frame <= frame_interval * 2
        if row.get("recognition_normalized") == value:
            run = run + 1 if contiguous else 1
        else:
            run = 0
        previous_frame = frame
    return run


def first_nonempty(rows, key):
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None
