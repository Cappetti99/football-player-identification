from collections import defaultdict

import numpy as np

from ft.features.visual import cosine_similarity, mean_embedding
from ft.utils.geometry import bbox_height, clip_bbox


def build_prtreid_tracklet_features(
    frames,
    tracks,
    extractor,
    max_samples_per_tracklet=8,
    min_crop_quality=0.15,
    min_samples=3,
):
    """Build one deterministic PRTReID prototype for every display tracklet."""
    grouped = defaultdict(list)
    for frame_num, frame_tracks in enumerate(tracks.get("players", [])):
        if frame_num >= len(frames):
            continue
        frame = frames[frame_num]
        for raw_id, track in frame_tracks.items():
            if track.get("bbox") is None:
                continue
            bbox = clip_bbox(track["bbox"], frame)
            crop = crop_from_bbox(frame, bbox)
            if crop is None:
                continue
            quality = tracklet_crop_quality(bbox, frame)
            display_id = int(track.get("display_track_id", raw_id))
            grouped[display_id].append(
                {
                    "frame": int(frame_num),
                    "raw_track_id": int(raw_id),
                    "crop": crop,
                    "crop_quality": float(quality),
                }
            )

    selected = []
    for display_id, items in sorted(grouped.items()):
        usable = [item for item in items if item["crop_quality"] >= float(min_crop_quality)]
        chosen = temporal_quality_sample(usable, max_samples_per_tracklet)
        for item in chosen:
            item["display_track_id"] = int(display_id)
            selected.append(item)

    features = extractor.extract_crops([item["crop"] for item in selected]) if selected else []
    by_display = defaultdict(list)
    for item, feature in zip(selected, features):
        embedding = feature.get("visual_embedding")
        if embedding is None:
            continue
        by_display[item["display_track_id"]].append((item, feature))

    prototypes = []
    prototype_by_display = {}
    for display_id in sorted(grouped):
        samples = by_display.get(display_id, [])
        embeddings = [feature["visual_embedding"] for _item, feature in samples]
        prototype = mean_embedding(embeddings)
        similarities = [cosine_similarity(prototype, embedding) for embedding in embeddings] if prototype else []
        row = {
            "display_track_id": int(display_id),
            "visual_embedding": prototype,
            "reid_model": "prtreid" if prototype is not None else None,
            "sample_count": len(embeddings),
            "eligible": bool(prototype is not None and len(embeddings) >= int(min_samples)),
            "prototype_consistency": float(np.mean(similarities)) if similarities else None,
            "mean_crop_quality": float(np.mean([item["crop_quality"] for item, _feature in samples])) if samples else None,
            "sampled_frames": [int(item["frame"]) for item, _feature in samples],
            "sampled_raw_track_ids": [int(item["raw_track_id"]) for item, _feature in samples],
        }
        prototypes.append(row)
        prototype_by_display[int(display_id)] = row

    attach_tracklet_prototypes(tracks, prototype_by_display)
    return prototypes


def temporal_quality_sample(items, max_samples):
    items = sorted(items, key=lambda item: (int(item["frame"]), int(item["raw_track_id"])))
    limit = int(max_samples or 0)
    if limit <= 0 or len(items) <= limit:
        return items
    bins = np.array_split(np.arange(len(items)), limit)
    chosen = []
    for indices in bins:
        bucket = [items[int(index)] for index in indices]
        chosen.append(max(bucket, key=lambda item: (float(item["crop_quality"]), -int(item["frame"]))))
    return sorted(chosen, key=lambda item: int(item["frame"]))


def attach_tracklet_prototypes(tracks, prototype_by_display):
    """Attach linker-only prototypes without changing downstream visual evidence."""
    for frame_tracks in tracks.get("players", []):
        for raw_id, track in frame_tracks.items():
            display_id = int(track.get("display_track_id", raw_id))
            feature = prototype_by_display.get(display_id)
            if not feature or feature.get("visual_embedding") is None:
                continue
            track["prtreid_tracklet_embedding"] = feature["visual_embedding"]
            track["prtreid_tracklet_sample_count"] = int(feature["sample_count"])
            track["prtreid_tracklet_eligible"] = bool(feature["eligible"])
            track["prtreid_tracklet_consistency"] = feature.get("prototype_consistency")
            track["prtreid_tracklet_mean_crop_quality"] = feature.get("mean_crop_quality")
            track["prtreid_tracklet_sampled_frames"] = feature.get("sampled_frames")


def crop_from_bbox(frame, bbox):
    if frame is None or bbox is None:
        return None
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def tracklet_crop_quality(bbox, frame):
    height, width = frame.shape[:2]
    bw = max(0, bbox[2] - bbox[0])
    bh = max(0, bbox[3] - bbox[1])
    area_score = min(1.0, (bw * bh) / float(max(1, width * height)) / 0.02)
    border_penalty = 0.4 if bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 1 else 0.0
    height_bonus = min(0.2, bbox_height(bbox) / 500.0)
    return max(0.0, float(area_score + height_bonus - border_penalty))
