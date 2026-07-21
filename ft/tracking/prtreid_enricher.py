"""Detection-level PRTReID enrichment, before temporal association."""

import math

from ft.features.prtreid import PRTReIDFeatureExtractor


class PRTReIDDetectionEnricher:
    """Attach appearance and role evidence to athlete observations.

    This stage deliberately does not alter bounding boxes, detector roles or
    association. A future appearance tracker may consume ``embedding`` while
    role aggregation can consume the separate metadata evidence.
    """

    def __init__(self, extractor=None, **extractor_options):
        enabled = bool(extractor_options.pop("enabled", True))
        if extractor is None and not enabled:
            raise ValueError("PRTReID detection enrichment requires an enabled extractor")
        self.extractor = extractor or PRTReIDFeatureExtractor(
            enabled=True,
            **extractor_options,
        )
        self.reset()

    def reset(self):
        self.stats = {
            "eligible": 0,
            "computed": 0,
            "invalid_crop": 0,
            "with_embedding": 0,
            "embedding_dim": None,
            "embedding_norm_sum": 0.0,
            "with_role": 0,
            "role_confidence_sum": 0.0,
            "role_counts": {},
            "visibility_value_sum": 0.0,
            "visibility_value_count": 0,
        }

    def enrich_batch(self, frames, detections):
        if len(frames) != len(detections):
            raise ValueError("PRTReID enrichment requires one detection record per frame")

        crops = []
        targets = []
        for image, frame_detections in zip(frames, detections):
            for observation in frame_detections.observations:
                if not observation.is_athlete:
                    continue
                self.stats["eligible"] += 1
                crop = crop_bbox(image, observation.bbox)
                if crop is None:
                    observation.metadata["prtreid_error"] = "invalid_crop"
                    self.stats["invalid_crop"] += 1
                    continue
                crops.append(crop)
                targets.append(observation)

        features = self.extractor.extract_crops(crops) if crops else []
        if len(features) != len(targets):
            raise RuntimeError(
                "PRTReID returned a different feature count "
                f"({len(features)}) than valid detections ({len(targets)})"
            )
        for observation, feature in zip(targets, features):
            observation.embedding = feature.get("visual_embedding")
            role = feature.get("reid_role_detection")
            confidence = feature.get("reid_role_confidence")
            observation.role_scores = (
                {str(role): float(confidence)}
                if role is not None and confidence is not None
                else None
            )
            observation.metadata.update({
                "reid_model": feature.get("reid_model", "prtreid"),
                "reid_role_detection": role,
                "reid_role_confidence": confidence,
                "reid_visibility_scores": feature.get("reid_visibility_scores"),
            })
            self.stats["computed"] += 1
            embedding = observation.embedding
            if embedding is not None:
                norm = math.sqrt(sum(float(value) ** 2 for value in embedding))
                self.stats["with_embedding"] += 1
                self.stats["embedding_dim"] = len(embedding)
                self.stats["embedding_norm_sum"] += norm
            if role is not None and confidence is not None:
                role_key = str(role)
                self.stats["with_role"] += 1
                self.stats["role_confidence_sum"] += float(confidence)
                counts = self.stats["role_counts"]
                counts[role_key] = counts.get(role_key, 0) + 1
            visibility = numeric_values(feature.get("reid_visibility_scores"))
            self.stats["visibility_value_sum"] += sum(visibility)
            self.stats["visibility_value_count"] += len(visibility)
        return detections

    def diagnostics(self):
        output = dict(self.stats)
        eligible = int(output["eligible"])
        computed = int(output["computed"])
        with_embedding = int(output.pop("with_embedding"))
        embedding_norm_sum = float(output.pop("embedding_norm_sum"))
        with_role = int(output.pop("with_role"))
        role_confidence_sum = float(output.pop("role_confidence_sum"))
        visibility_sum = float(output.pop("visibility_value_sum"))
        visibility_count = int(output.pop("visibility_value_count"))
        output.update({
            "coverage": with_embedding / eligible if eligible else 0.0,
            "valid_crop_rate": computed / eligible if eligible else 0.0,
            "embedding_count": with_embedding,
            "embedding_norm_mean": embedding_norm_sum / with_embedding if with_embedding else None,
            "role_count": with_role,
            "role_confidence_mean": role_confidence_sum / with_role if with_role else None,
            "visibility_mean": visibility_sum / visibility_count if visibility_count else None,
        })
        if hasattr(self.extractor, "diagnostics"):
            output["extractor"] = self.extractor.diagnostics()
        return output


def crop_bbox(image, bbox):
    if image is None or getattr(image, "size", 0) == 0:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    left = max(0, min(width, int(x1)))
    top = max(0, min(height, int(y1)))
    right = max(0, min(width, int(x2 + 0.999999)))
    bottom = max(0, min(height, int(y2 + 0.999999)))
    if right <= left or bottom <= top:
        return None
    crop = image[top:bottom, left:right]
    return crop if crop.size else None


def numeric_values(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(numeric_values(item))
        return output
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []
