"""Rank jersey crops without consulting OCR output or ground truth."""

import math
from pathlib import Path

from ft.caching.cache_manager import hash_file


class JerseyFrameSelector:
    MODES = {"audit", "propose", "apply"}
    YOLO_BACK_MODEL_TYPE = "jersey_back_yolo11s_cls"
    YOLO_READABILITY_MODEL_TYPE = "jersey_number_readability_yolo26s_cls"
    YOLO_MODEL_TYPES = {YOLO_BACK_MODEL_TYPE, YOLO_READABILITY_MODEL_TYPE}
    MODEL_TYPES = {"legibility_resnet34", *YOLO_MODEL_TYPES}

    def __init__(
        self,
        checkpoint,
        checkpoint_sha256,
        model_type="legibility_resnet34",
        mode="audit",
        device="cuda",
        batch_size=32,
        top_k=5,
        min_legibility_score=0.50,
        min_selection_score=None,
        min_frame_gap=5,
        min_winner_votes=2,
        min_margin=0.0,
        allowed_roles=None,
        gate_each_frame_role=True,
        clean_back_weight=0.70,
        sharpness_weight=0.15,
        size_weight=0.05,
        crop_quality_weight=0.10,
        sharpness_scale=100.0,
        size_scale=160.0,
        use_torso_crop=True,
        torso_x_min=0.15,
        torso_x_max=0.85,
        torso_y_min=0.08,
        torso_y_max=0.68,
    ):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.expected_sha256 = str(checkpoint_sha256 or "").lower()
        self.model_type = str(model_type or "legibility_resnet34").lower()
        self.mode = str(mode).lower()
        self.device_name = str(device)
        self.batch_size = int(batch_size)
        self.top_k = int(top_k)
        self.min_legibility_score = float(min_legibility_score)
        self.min_selection_score = None if min_selection_score is None else float(min_selection_score)
        self.min_frame_gap = int(min_frame_gap)
        self.min_winner_votes = int(min_winner_votes)
        self.min_margin = float(min_margin)
        self.allowed_roles = set(allowed_roles or ["player"])
        self.gate_each_frame_role = bool(gate_each_frame_role)
        self.score_weights = {
            "clean_back": float(clean_back_weight),
            "sharpness": float(sharpness_weight),
            "size": float(size_weight),
            "crop_quality": float(crop_quality_weight),
        }
        self.sharpness_scale = float(sharpness_scale)
        self.size_scale = float(size_scale)
        self.use_torso_crop = bool(use_torso_crop)
        self.torso_bounds = (
            float(torso_x_min), float(torso_y_min),
            float(torso_x_max), float(torso_y_max),
        )
        self.score_rows = []
        self.selection_rows = []
        self.decision_rows = []
        self._model = None
        self._transform = None
        self._device = None
        self.checkpoint_sha256 = None
        self.model_metadata = {}
        self._positive_class_index = None
        self._validate_configuration()
        self._load_model()

    def _validate_configuration(self):
        if self.mode not in self.MODES:
            raise ValueError(f"unsupported jersey frame selection mode: {self.mode!r}")
        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(f"unsupported jersey frame selection model_type: {self.model_type!r}")
        if self.model_type in self.YOLO_MODEL_TYPES and self.mode == "apply":
            raise ValueError("YOLO jersey frame classifiers support audit/propose, not apply")
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise FileNotFoundError(f"jersey frame selector checkpoint not found: {self.checkpoint}")
        self.checkpoint_sha256 = hash_file(self.checkpoint)
        if not self.expected_sha256:
            raise ValueError("jersey frame selector checkpoint_sha256 is required")
        if self.checkpoint_sha256.lower() != self.expected_sha256:
            raise ValueError(
                "jersey frame selector checkpoint SHA-256 mismatch: "
                f"expected {self.expected_sha256}, got {self.checkpoint_sha256}"
            )
        if self.batch_size <= 0 or self.top_k <= 0 or self.min_winner_votes <= 0:
            raise ValueError("batch_size, top_k and min_winner_votes must be positive")
        if self.min_frame_gap < 0:
            raise ValueError("min_frame_gap must be non-negative")
        if self.min_selection_score is not None and self.min_selection_score < 0.0:
            raise ValueError("min_selection_score must be non-negative")
        if any(value < 0.0 for value in self.score_weights.values()):
            raise ValueError("jersey selector score weights must be non-negative")
        if sum(self.score_weights.values()) <= 0.0:
            raise ValueError("at least one jersey selector score weight must be positive")
        if self.sharpness_scale <= 0.0 or self.size_scale <= 0.0:
            raise ValueError("sharpness_scale and size_scale must be positive")
        x0, y0, x1, y1 = self.torso_bounds
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("torso crop bounds must be ordered values between 0 and 1")

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for jersey frame selection") from exc
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested jersey selector device is unavailable: {self.device_name}")
        self._device = torch.device(self.device_name)

        if self.model_type in self.YOLO_MODEL_TYPES:
            try:
                from ultralytics import YOLO
            except Exception as exc:
                raise RuntimeError("Ultralytics is required for YOLO jersey frame selection") from exc
            self._model = YOLO(str(self.checkpoint))
            names = self._model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            target = (
                "number_readable"
                if self.model_type == self.YOLO_READABILITY_MODEL_TYPE
                else "clean_back"
            )
            self._positive_class_index = next(
                (int(index) for index, name in items if str(name) == target), None
            )
            if self._positive_class_index is None:
                raise ValueError(f"YOLO jersey checkpoint lacks {target} class: {names}")
            self.model_metadata = {"class_names": names, "target": target}
            return

        from torch import nn
        from torchvision import models, transforms

        class LegibilityClassifier34(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_ft = models.resnet34(weights=None)
                self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

            def forward(self, images):
                return torch.sigmoid(self.model_ft(images))

        model = LegibilityClassifier34()
        model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"), strict=True)
        self._model = model.to(self._device).eval()
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def select(self, display_track_id, rows, min_crop_quality=0.0):
        """Score and rank one display track without consulting OCR output."""
        scored = self._score(display_track_id, rows)
        ranked = sorted(
            scored,
            key=lambda item: (-self._ranking_score(item), item["frame"], item["crop_path"]),
        )
        selected = []
        for item in ranked:
            if item["pred_role"] not in self.allowed_roles:
                reason = "role_not_allowed"
            elif item.get("score_status") != "ok":
                reason = "score_unavailable"
            elif item["crop_quality"] < float(min_crop_quality):
                reason = "below_crop_quality"
            elif self.model_type in self.YOLO_MODEL_TYPES and (
                self.min_selection_score is not None
                and item["selection_score"] < self.min_selection_score
            ):
                reason = "below_selection_threshold"
            elif self.model_type == "legibility_resnet34" and item["legibility_score"] < self.min_legibility_score:
                reason = "below_legibility_threshold"
            elif any(abs(item["frame"] - other["frame"]) < self.min_frame_gap for other in selected):
                reason = "temporal_near_duplicate"
            elif len(selected) >= self.top_k:
                reason = "outside_top_k"
            else:
                selected.append(item)
                reason = "selected"
            item["selection_reason"] = reason
            item["selected"] = reason == "selected"
        for rank, item in enumerate(selected, start=1):
            item["selection_rank"] = rank

        selected_rows = []
        for item in ranked:
            item.setdefault("selection_rank", None)
            if self.model_type == self.YOLO_BACK_MODEL_TYPE:
                item.setdefault("selector_torso_crop_path", None)
                item.setdefault("torso_box_pixels", None)
                item.setdefault("torso_crop_status", "not_materialized")
            if item.get("selected"):
                output_row = rows[item["row_index"]]
                if (
                    self.model_type == self.YOLO_BACK_MODEL_TYPE
                    and self.mode == "propose"
                    and self.use_torso_crop
                ):
                    output_row, provenance = self._materialize_torso_crop(dict(output_row))
                    item.update(provenance)
                selected_rows.append(output_row)
            self.selection_rows.append(dict(item))
        return selected_rows

    def _ranking_score(self, item):
        key = "selection_score" if self.model_type in self.YOLO_MODEL_TYPES else "legibility_score"
        return float(item.get(key, 0.0) or 0.0)

    def _score(self, display_track_id, rows):
        import torch
        from PIL import Image

        valid = []
        scored = []
        for index, row in enumerate(rows):
            path = Path(str(row.get("crop_path") or ""))
            base = {
                "display_track_id": int(display_track_id),
                "row_index": index,
                "frame": int(row.get("frame", 0) or 0),
                "crop_path": str(path),
                "pred_role": predicted_role(row),
                "crop_quality": float(row.get("crop_quality", 0.0) or 0.0),
                "legibility_score": 0.0,
            }
            if self.model_type in self.YOLO_MODEL_TYPES:
                base.update({
                    "clean_back_score": None,
                    "number_readability_score": None,
                    "sharpness_score": None,
                    "size_score": None,
                    "selection_score": 0.0,
                    "selector_crop_variant": "full_player",
                    "selector_original_crop_path": str(path),
                    "torso_bounds_normalized": list(self.torso_bounds),
                })
            if base["pred_role"] not in self.allowed_roles:
                base["score_status"] = "role_not_allowed"
                scored.append(base)
            elif not path.is_file():
                base["score_status"] = "missing_crop"
                scored.append(base)
            else:
                valid.append((base, path))

        if self.model_type in self.YOLO_MODEL_TYPES:
            for start in range(0, len(valid), self.batch_size):
                batch = valid[start:start + self.batch_size]
                predictions = self._model.predict(
                    [str(path) for _, path in batch],
                    batch=self.batch_size,
                    device=self.device_name,
                    verbose=False,
                )
                for (base, path), prediction in zip(batch, predictions):
                    positive_score = float(
                        prediction.probs.data[self._positive_class_index].item()
                    )
                    if self.model_type == self.YOLO_READABILITY_MODEL_TYPE:
                        base.update({
                            "number_readability_score": positive_score,
                            "selection_score": positive_score,
                            "score_status": "ok",
                        })
                    else:
                        sharpness, size = crop_observation_scores(
                            path, self.sharpness_scale, self.size_scale
                        )
                        base.update({
                            "clean_back_score": positive_score,
                            "sharpness_score": sharpness,
                            "size_score": size,
                            "selection_score": composite_selection_score(
                                positive_score,
                                sharpness,
                                size,
                                base["crop_quality"],
                                self.score_weights,
                            ),
                            "score_status": "ok",
                        })
                    scored.append(base)
        else:
            with torch.inference_mode():
                for start in range(0, len(valid), self.batch_size):
                    batch = valid[start:start + self.batch_size]
                    tensors = []
                    for _, path in batch:
                        with Image.open(path) as image:
                            tensors.append(self._transform(image.convert("RGB")))
                    values = self._model(torch.stack(tensors).to(self._device)).reshape(-1).cpu().tolist()
                    for (base, _), value in zip(batch, values):
                        base["legibility_score"] = float(value)
                        base["score_status"] = "ok"
                        scored.append(base)

        scored.sort(key=lambda item: item["row_index"])
        self.score_rows.extend(dict(item) for item in scored)
        return scored

    def _materialize_torso_crop(self, row):
        from PIL import Image

        source = Path(str(row.get("crop_path") or ""))
        provenance = {
            "selector_original_crop_path": str(source),
            "selector_crop_variant": "full_player_fallback",
            "selector_torso_crop_path": None,
            "torso_bounds_normalized": list(self.torso_bounds),
        }
        if not source.is_file():
            provenance["torso_crop_status"] = "missing_source"
            return row, provenance
        target = source.parent / "selector_torso" / source.name
        try:
            with Image.open(source) as image:
                width, height = image.size
                x0, y0, x1, y1 = self.torso_bounds
                box = (
                    int(round(x0 * width)), int(round(y0 * height)),
                    int(round(x1 * width)), int(round(y1 * height)),
                )
                torso = image.convert("RGB").crop(box)
                target.parent.mkdir(parents=True, exist_ok=True)
                torso.save(target, quality=95)
            row.update({
                "selector_original_crop_path": str(source),
                "crop_path": str(target),
                "selector_crop_variant": "torso",
                "torso_bounds_normalized": list(self.torso_bounds),
                "torso_box_pixels": list(box),
            })
            provenance.update({
                "selector_crop_variant": "torso",
                "selector_torso_crop_path": str(target),
                "torso_box_pixels": list(box),
                "torso_crop_status": "ok",
            })
        except Exception as exc:
            provenance["torso_crop_status"] = f"error:{type(exc).__name__}"
        return row, provenance

    def accept_decision(self, display_track_id, voted, selected_count):
        evaluated = self.mode in {"propose", "apply"}
        accepted = bool(
            evaluated and voted
            and int(voted.get("votes", 0)) >= self.min_winner_votes
            and float(voted.get("winner_margin", 0.0)) >= self.min_margin
        )
        self.decision_rows.append({
            "display_track_id": int(display_track_id),
            "mode": self.mode,
            "model_type": self.model_type,
            "evaluated": evaluated,
            "selected_crops": int(selected_count),
            "raw_prediction": voted.get("jersey_number") if voted else None,
            "winner_votes": voted.get("votes", 0) if voted else 0,
            "winner_margin": voted.get("winner_margin") if voted else None,
            "accepted": accepted,
            "decision": voted.get("jersey_number") if accepted else None,
            "rejection_reason": (
                "audit_selection_only" if not evaluated
                else "insufficient_votes_or_margin" if not accepted else None
            ),
        })
        return accepted

    def diagnostics(self):
        return {
            "enabled": True,
            "mode": self.mode,
            "model_type": self.model_type,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self._device or self.device_name),
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "min_legibility_score": self.min_legibility_score,
            "min_selection_score": self.min_selection_score,
            "min_frame_gap": self.min_frame_gap,
            "min_winner_votes": self.min_winner_votes,
            "min_margin": self.min_margin,
            "allowed_roles": sorted(self.allowed_roles),
            "gate_each_frame_role": self.gate_each_frame_role,
            "score_weights": dict(self.score_weights),
            "sharpness_scale": self.sharpness_scale,
            "size_scale": self.size_scale,
            "use_torso_crop": self.use_torso_crop,
            "torso_bounds_normalized": list(self.torso_bounds),
            "scored_crops": len(self.score_rows),
            "selected_crops": sum(bool(row.get("selected")) for row in self.selection_rows),
            "decisions": len(self.decision_rows),
            "accepted_decisions": sum(bool(row.get("accepted")) for row in self.decision_rows),
            "target_class": self.model_metadata.get("target"),
            "license": (
                "project-trained checkpoint; see checkpoint metadata"
                if self.model_type in self.YOLO_MODEL_TYPES
                else "CC BY-NC 3.0 (upstream checkpoint; research/non-commercial use)"
            ),
        }


def composite_selection_score(clean_back, sharpness, size, crop_quality, weights):
    values = {
        "clean_back": clamp01(clean_back),
        "sharpness": clamp01(sharpness),
        "size": clamp01(size),
        "crop_quality": clamp01(crop_quality),
    }
    denominator = sum(float(weights.get(key, 0.0)) for key in values)
    if denominator <= 0.0:
        return 0.0
    return sum(float(weights.get(key, 0.0)) * value for key, value in values.items()) / denominator


def crop_observation_scores(path, sharpness_scale, size_scale):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return 0.0, 0.0
    sharpness_raw = float(cv2.Laplacian(image, cv2.CV_64F).var())
    short_side = float(min(image.shape[:2]))
    return (
        1.0 - math.exp(-sharpness_raw / float(sharpness_scale)),
        1.0 - math.exp(-short_side / float(size_scale)),
    )


def clamp01(value):
    return max(0.0, min(1.0, float(value or 0.0)))


def predicted_role(row):
    return str(
        row.get("role_detection") or row.get("pred_role") or row.get("role") or "unknown"
    ).lower()
