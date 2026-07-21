"""PRTReID main-subject filtering for jersey OCR crop sequences."""

from pathlib import Path

import numpy as np

from ft.caching.cache_manager import hash_file
from ft.features.jersey_frame_selector import predicted_role
from ft.features.prtreid import PRTReIDFeatureExtractor


class JerseySubjectFilter:
    MODES = {"audit", "propose"}

    def __init__(
        self,
        checkpoint,
        checkpoint_sha256,
        mode="audit",
        device="cuda",
        batch_size=32,
        allowed_roles=None,
        min_crop_quality=0.08,
        min_samples=3,
        min_remaining=2,
        rounds=3,
        std_threshold=3.5,
        hrnet_pretrained_path="models/reid",
        extractor=None,
    ):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.expected_sha256 = str(checkpoint_sha256 or "").lower()
        self.mode = str(mode).lower()
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.allowed_roles = set(allowed_roles or ["player"])
        self.min_crop_quality = float(min_crop_quality)
        self.min_samples = int(min_samples)
        self.min_remaining = int(min_remaining)
        self.rounds = int(rounds)
        self.std_threshold = float(std_threshold)
        self.hrnet_pretrained_path = str(hrnet_pretrained_path)
        self.checkpoint_sha256 = None
        self.score_rows = []
        self.track_rows = []
        self._validate()
        self.extractor = extractor or PRTReIDFeatureExtractor(
            enabled=True,
            weights_path=str(self.checkpoint),
            hrnet_pretrained_path=self.hrnet_pretrained_path,
            device=self.device,
            batch_size=self.batch_size,
            role_enabled=False,
        )
        # Fail before OCR initialization; enabled research features must never
        # silently fall back to unfiltered crops.
        if extractor is None:
            self.extractor._ensure_ready()

    def _validate(self):
        if self.mode not in self.MODES:
            raise ValueError(f"unsupported jersey subject filter mode: {self.mode!r}")
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise FileNotFoundError(f"jersey subject-filter checkpoint not found: {self.checkpoint}")
        self.checkpoint_sha256 = hash_file(self.checkpoint)
        if not self.expected_sha256:
            raise ValueError("jersey subject-filter checkpoint_sha256 is required")
        if self.checkpoint_sha256.lower() != self.expected_sha256:
            raise ValueError(
                "jersey subject-filter checkpoint SHA-256 mismatch: "
                f"expected {self.expected_sha256}, got {self.checkpoint_sha256}"
            )
        if self.batch_size <= 0 or self.min_samples <= 0 or self.min_remaining <= 0:
            raise ValueError("batch_size, min_samples and min_remaining must be positive")
        if self.rounds <= 0 or self.std_threshold < 0:
            raise ValueError("rounds must be positive and std_threshold non-negative")

    def filter(self, display_track_id, rows):
        records = []
        feature_rows = []
        feature_indices = []
        for index, row in enumerate(rows):
            role = predicted_role(row)
            quality = float(row.get("crop_quality", 0.0) or 0.0)
            path = Path(str(row.get("crop_path") or ""))
            record = {
                "display_track_id": int(display_track_id),
                "row_index": int(index),
                "frame": int(row.get("frame", 0) or 0),
                "raw_track_id": row.get("raw_track_id", row.get("track_id")),
                "crop_path": str(path),
                "pred_role": role,
                "crop_quality": quality,
                "embedding_available": False,
                "excluded_round": None,
                "subject_outlier": False,
            }
            if role not in self.allowed_roles:
                record["filter_reason"] = "role_not_allowed"
            elif quality < self.min_crop_quality:
                record["filter_reason"] = "below_crop_quality"
            elif not path.is_file():
                record["filter_reason"] = "missing_crop"
            else:
                record["filter_reason"] = "candidate"
                feature_rows.append({"crop_path": str(path)})
                feature_indices.append(index)
            records.append(record)

        if feature_rows:
            self.extractor.add_row_features(feature_rows)
        embeddings = []
        embedded_indices = []
        for row_index, feature in zip(feature_indices, feature_rows):
            embedding = feature.get("visual_embedding")
            if embedding is None:
                records[row_index]["filter_reason"] = "embedding_unavailable"
                continue
            records[row_index]["embedding_available"] = True
            embeddings.append(np.asarray(embedding, dtype=np.float64))
            embedded_indices.append(row_index)

        result = iterative_gaussian_filter(
            embeddings,
            rounds=self.rounds,
            std_threshold=self.std_threshold,
            min_samples=self.min_samples,
        )
        final_local = set(result["kept_indices"])
        for local_index, row_index in enumerate(embedded_indices):
            record = records[row_index]
            record["distance"] = result["distances"].get(local_index)
            record["excluded_round"] = result["excluded_rounds"].get(local_index)
            record["subject_outlier"] = local_index not in final_local
            record["filter_reason"] = "subject_outlier" if record["subject_outlier"] else "kept"

        fallback = bool(
            result["eligible"] and len(final_local) < self.min_remaining
        )
        if fallback:
            for row_index in embedded_indices:
                records[row_index]["subject_outlier"] = False
                records[row_index]["filter_reason"] = "fallback_min_remaining"

        excluded_row_indices = {
            record["row_index"] for record in records if record["subject_outlier"]
        }
        effective = (
            [row for index, row in enumerate(rows) if index not in excluded_row_indices]
            if self.mode == "propose"
            else list(rows)
        )
        self.score_rows.extend(dict(record) for record in records)
        self.track_rows.append({
            "display_track_id": int(display_track_id),
            "mode": self.mode,
            "input_crops": len(rows),
            "candidate_crops": len(feature_indices),
            "embedded_crops": len(embedded_indices),
            "eligible": bool(result["eligible"]),
            "kept_crops": len(embedded_indices) - len(excluded_row_indices),
            "excluded_crops": len(excluded_row_indices),
            "fallback": fallback,
            "effective_crops": len(effective),
        })
        return effective

    def diagnostics(self):
        return {
            "enabled": True,
            "mode": self.mode,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "batch_size": self.batch_size,
            "allowed_roles": sorted(self.allowed_roles),
            "min_crop_quality": self.min_crop_quality,
            "min_samples": self.min_samples,
            "min_remaining": self.min_remaining,
            "rounds": self.rounds,
            "std_threshold": self.std_threshold,
            "scored_crops": len(self.score_rows),
            "excluded_crops": sum(bool(row["subject_outlier"]) for row in self.score_rows),
            "tracks": len(self.track_rows),
            "fallback_tracks": sum(bool(row["fallback"]) for row in self.track_rows),
            "extractor": self.extractor.diagnostics(),
            "license": "CC BY-NC 3.0 (upstream method/checkpoint; research/non-commercial use)",
        }


def iterative_gaussian_filter(embeddings, rounds=3, std_threshold=3.5, min_samples=3):
    vectors = np.asarray(embeddings, dtype=np.float64)
    count = len(vectors)
    if count < int(min_samples):
        return {
            "eligible": False,
            "kept_indices": list(range(count)),
            "excluded_rounds": {},
            "distances": {},
        }
    active = list(range(count))
    excluded_rounds = {}
    final_distances = {}
    for round_index in range(1, int(rounds) + 1):
        active_vectors = vectors[active]
        centroid = np.mean(active_vectors, axis=0)
        distances = np.linalg.norm(active_vectors - centroid, axis=1)
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances))
        threshold = mean_distance + float(std_threshold) * std_distance
        keep_mask = distances <= threshold if std_distance > 0 else np.ones(len(active), dtype=bool)
        next_active = []
        for index, distance, keep in zip(active, distances, keep_mask):
            final_distances[int(index)] = float(distance)
            if keep:
                next_active.append(index)
            else:
                excluded_rounds.setdefault(int(index), int(round_index))
        active = next_active
        if not active or bool(np.all(keep_mask)):
            break
    if active:
        centroid = np.mean(vectors[active], axis=0)
        for index in range(count):
            final_distances[index] = float(np.linalg.norm(vectors[index] - centroid))
    return {
        "eligible": True,
        "kept_indices": sorted(int(index) for index in active),
        "excluded_rounds": excluded_rounds,
        "distances": final_distances,
    }
