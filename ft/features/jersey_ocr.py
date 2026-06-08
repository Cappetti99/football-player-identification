import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ft.caching.cache_manager import DisabledCache, JsonDiskCache, hash_file, stable_hash
from ft.features.jersey_template import JerseyTemplateMatcher


class JerseyOCR:
    """Optional OCR over player crops, aggregated by display_track_id.

    OCR backends are used as proposal generators. The final jersey number comes
    from crop-level aggregation and tracklet voting, not from a single OCR call.
    """

    def __init__(
        self,
        backend="auto",
        min_confidence=0.4,
        max_crops_per_tracklet=12,
        temporal_passes=1,
        augment=True,
        min_crop_quality=0.08,
        min_votes=2,
        min_raw_confidence=0.05,
        min_winner_margin=0.15,
        easyocr_gpu=False,
        debug_dir=None,
        template_matching=False,
        template_font_image=None,
        template_min_score=0.62,
        template_weight=0.03,
        template_max_candidates=4,
        aggregate_by_crop=True,
        max_candidates_per_crop=3,
        min_crop_candidate_ratio=0.35,
        crop_quality_vote_weighting=False,
        crop_quality_min_vote_weight=0.35,
        crop_quality_vote_power=1.0,
        mmocr_device=None,
        mmocr_det="dbnet_resnet18_fpnc_1200e_icdar2015",
        mmocr_rec="SAR",
        mmocr_batch_size=8,
        mmocr_direct_recognition=None,
        progress_every=5,
        cache_enabled=True,
        cache_dir=".ft_cache/ocr",
        number_roi_enabled=False,
        number_roi_upscale=3,
        number_roi_clahe=True,
        broadcast_contrast_enabled=False,
        broadcast_contrast_clip_limit=4.0,
        broadcast_contrast_tile_grid_size=4,
        super_resolution_enabled=False,
        super_resolution_scale=4,
        super_resolution_max_side=100,
        segment_frames=0,
        demote_direct_only_single_digits=True,
        prefer_two_digit_candidates=True,
        digit_confusion_overrides=None,
    ):
        self.requested_backend_name = backend
        self.backend_name = backend
        self.min_confidence = float(min_confidence)
        self.max_crops_per_tracklet = int(max_crops_per_tracklet)
        self.temporal_passes = max(1, int(temporal_passes))
        self.augment = bool(augment)
        self.min_crop_quality = float(min_crop_quality)
        self.min_votes = int(min_votes)
        self.min_raw_confidence = float(min_raw_confidence)
        self.min_winner_margin = float(min_winner_margin)
        self.easyocr_gpu = bool(easyocr_gpu)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.template_matching = bool(template_matching)
        self.template_font_image = template_font_image
        self.template_min_score = float(template_min_score)
        self.template_weight = float(template_weight)
        self.template_max_candidates = int(template_max_candidates)
        self.aggregate_by_crop = bool(aggregate_by_crop)
        self.max_candidates_per_crop = int(max_candidates_per_crop)
        self.min_crop_candidate_ratio = float(min_crop_candidate_ratio)
        self.crop_quality_vote_weighting = bool(crop_quality_vote_weighting)
        self.crop_quality_min_vote_weight = max(0.0, min(1.0, float(crop_quality_min_vote_weight)))
        self.crop_quality_vote_power = max(0.01, float(crop_quality_vote_power))
        self.mmocr_device = mmocr_device
        self.mmocr_det = str(mmocr_det)
        self.mmocr_rec = str(mmocr_rec)
        self.mmocr_batch_size = int(mmocr_batch_size)
        self.mmocr_direct_recognition = (
            bool(number_roi_enabled) if mmocr_direct_recognition is None else bool(mmocr_direct_recognition)
        )
        self.progress_every = int(progress_every or 0)
        self.backends = []
        self.cache_enabled = bool(cache_enabled)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".ft_cache/ocr")
        self.number_roi_enabled = bool(number_roi_enabled)
        self.number_roi_upscale = max(1, int(number_roi_upscale))
        self.number_roi_clahe = bool(number_roi_clahe)
        self.broadcast_contrast_enabled = bool(broadcast_contrast_enabled)
        self.broadcast_contrast_clip_limit = float(broadcast_contrast_clip_limit)
        self.broadcast_contrast_tile_grid_size = max(2, int(broadcast_contrast_tile_grid_size))
        self.super_resolution_enabled = bool(super_resolution_enabled)
        self.super_resolution_scale = max(1, int(super_resolution_scale))
        self.super_resolution_max_side = max(1, int(super_resolution_max_side))
        self.segment_frames = max(0, int(segment_frames or 0))
        self.demote_direct_only_single_digits = bool(demote_direct_only_single_digits)
        self.prefer_two_digit_candidates = bool(prefer_two_digit_candidates)
        self.digit_confusion_overrides = normalize_digit_confusion_overrides(digit_confusion_overrides)
        self.cache = JsonDiskCache(self.cache_dir, "jersey_ocr") if self.cache_enabled else DisabledCache()
        self.backend = None
        self.message = None
        self.backend_attempts = []
        self.template_matcher = None
        self.template_message = None

    def recognize(self, rows):
        self.backends = self._load_backends()
        self.backend = self.backends[0]["reader"] if self.backends else None
        self.template_matcher = self._load_template_matcher()
        if not self.backends and self.template_matcher is None:
            return {}, {
                "enabled": True,
                "status": "missing_backend",
                "backend": self.backend_name,
                "backends": [],
                "requested_backend": self.requested_backend_name,
                "message": self.message,
                "backend_attempts": self.backend_attempts,
                "template_matching": self._template_diagnostics(),
                "tracklets": {},
            }
        grouped = defaultdict(list)
        for row in rows:
            if row.get("crop_path") and is_ocr_player_row(row):
                grouped[jersey_group_key(row, self.segment_frames)].append(row)
        assignments = {}
        diagnostics = {}
        grouped_items = sorted(grouped.items(), key=lambda item: diagnostic_key(item[0]))
        print(
            f"FT jersey OCR: start tracklets={len(grouped_items)}"
            f" backend={self.backend_name}"
            f" template={self.template_matching}",
            flush=True,
        )
        for index, (track_id, items) in enumerate(grouped_items, start=1):
            display_id, segment_index = split_jersey_group_key(track_id)
            usable_items = filter_quality_items(items, self.min_crop_quality)
            selected = []
            detections = []
            for pass_index in range(self.temporal_passes):
                # Each pass samples a different temporal offset. This gives long
                # tracklets several chances to expose the jersey without reading
                # every crop in a long broadcast sequence.
                pass_rows = select_spread_crops(
                    usable_items,
                    self.max_crops_per_tracklet,
                    pass_index=pass_index,
                    pass_count=self.temporal_passes,
                )
                for row in pass_rows:
                    selected.append((pass_index, row))
                    detections.extend(self._recognize_row(row, display_id, pass_index))
            detections = apply_crop_quality_vote_weights(
                detections,
                selected,
                enabled=self.crop_quality_vote_weighting,
                min_weight=self.crop_quality_min_vote_weight,
                power=self.crop_quality_vote_power,
            )
            voting_detections = self._voting_detections(detections)
            voted = vote_numbers(
                voting_detections,
                min_raw_confidence=self.min_raw_confidence,
                digit_confusion_overrides=self.digit_confusion_overrides,
            )
            decision = ocr_decision_diagnostics(
                detections,
                voting_detections,
                voted,
                min_votes=self.min_votes,
                min_raw_confidence=self.min_raw_confidence,
            )
            diagnostics[diagnostic_key(track_id)] = {
                "display_track_id": int(display_id),
                "segment_index": segment_index,
                "segment_start_frame": segment_start_frame(segment_index, self.segment_frames),
                "segment_end_frame": segment_end_frame(segment_index, self.segment_frames),
                "available_crops": len(items),
                "usable_crops": len(usable_items),
                "crop_selection": crop_selection_diagnostics(items, usable_items, selected),
                "selected_crops": [
                    {
                        "pass": pass_index,
                        "crop_path": row.get("crop_path"),
                        "frame": row.get("frame"),
                        "crop_quality": row.get("crop_quality", 0.0),
                    }
                    for pass_index, row in selected
                ],
                "detections": detections,
                "template_detections": [
                    item for item in detections if item.get("source") == "template"
                ],
                "aggregated_detections": voting_detections if self.aggregate_by_crop else [],
                "raw_detection_count": len(detections),
                "voting_detection_count": len(voting_detections),
                "voted": voted,
                "decision": decision,
            }
            if voted and voted["votes"] >= self.min_votes:
                voted["display_track_id"] = int(display_id)
                voted["segment_index"] = segment_index
                voted["segment_start_frame"] = segment_start_frame(segment_index, self.segment_frames)
                voted["segment_end_frame"] = segment_end_frame(segment_index, self.segment_frames)
                assignments[track_id] = voted
            if self.progress_every > 0 and (index % self.progress_every == 0 or index == len(grouped_items)):
                print(
                    f"FT jersey OCR: tracklets={index}/{len(grouped_items)}"
                    f" assigned={len(assignments)}"
                    f" last_track={track_id}"
                    f" detections={len(detections)}",
                    flush=True,
                )
        return assignments, {
            "enabled": True,
            "status": "ok" if self.backends else "template_only",
            "requested_backend": self.requested_backend_name,
            "backend": self.backend_name if self.backends else None,
            "backends": [item["name"] for item in self.backends],
            "backend_attempts": self.backend_attempts,
            "min_confidence": self.min_confidence,
            "max_crops_per_tracklet": self.max_crops_per_tracklet,
            "temporal_passes": self.temporal_passes,
            "augment": self.augment,
            "min_crop_quality": self.min_crop_quality,
            "min_votes": self.min_votes,
            "min_raw_confidence": self.min_raw_confidence,
            "min_winner_margin": self.min_winner_margin,
            "easyocr_gpu": self.easyocr_gpu,
            "debug_dir": str(self.debug_dir) if self.debug_dir else None,
            "aggregate_by_crop": self.aggregate_by_crop,
            "max_candidates_per_crop": self.max_candidates_per_crop,
            "min_crop_candidate_ratio": self.min_crop_candidate_ratio,
            "mmocr": {
                "device": self.mmocr_device,
                "det": self.mmocr_det,
                "rec": self.mmocr_rec,
                "batch_size": self.mmocr_batch_size,
                "direct_recognition": self.mmocr_direct_recognition,
            },
            "number_roi": {
                "enabled": self.number_roi_enabled,
                "upscale": self.number_roi_upscale,
                "clahe": self.number_roi_clahe,
            },
            "broadcast_contrast": {
                "enabled": self.broadcast_contrast_enabled,
                "clip_limit": self.broadcast_contrast_clip_limit,
                "tile_grid_size": self.broadcast_contrast_tile_grid_size,
            },
            "super_resolution": {
                "enabled": self.super_resolution_enabled,
                "scale": self.super_resolution_scale,
                "max_side": self.super_resolution_max_side,
            },
            "segment_frames": self.segment_frames,
            "voting_policy": {
                "demote_direct_only_single_digits": self.demote_direct_only_single_digits,
                "prefer_two_digit_candidates": self.prefer_two_digit_candidates,
                "digit_confusion_overrides": self.digit_confusion_overrides,
                "crop_quality_vote_weighting": self.crop_quality_vote_weighting,
                "crop_quality_min_vote_weight": self.crop_quality_min_vote_weight,
                "crop_quality_vote_power": self.crop_quality_vote_power,
            },
            "cache": self.cache.diagnostics(),
            "template_matching": self._template_diagnostics(),
            "tracklets": diagnostics,
            "assigned_tracklets": {diagnostic_key(k): v for k, v in assignments.items()},
        }

    def _voting_detections(self, detections):
        if not self.aggregate_by_crop:
            return detections
        # Multiple variants of the same crop often produce duplicate OCR hits.
        # Collapsing them first prevents one good frame from dominating the
        # whole tracklet vote just because it had many preprocessing variants.
        return aggregate_detections_by_crop(
            detections,
            min_raw_confidence=self.min_raw_confidence,
            max_candidates_per_crop=self.max_candidates_per_crop,
            min_candidate_ratio=self.min_crop_candidate_ratio,
            demote_direct_only_single_digits_enabled=self.demote_direct_only_single_digits,
            prefer_two_digit_candidates_enabled=self.prefer_two_digit_candidates,
        )

    def _load_backends(self):
        requested = requested_backends(self.requested_backend_name)
        mode = backend_load_mode(self.requested_backend_name)
        loaded_backends = []
        errors = []
        for backend in requested:
            try:
                loaded = self._load_backend_instance(backend)
            except Exception as exc:
                message = f"{backend}: {type(exc).__name__}: {exc}"
                errors.append(message)
                self.backend_attempts.append(
                    {"backend": backend, "status": "failed", "message": message}
                )
                continue
            loaded_backends.append({"name": backend, "reader": loaded})
            self.backend_attempts.append({"backend": backend, "status": "ok"})
            if mode == "fallback":
                break
        self.message = "; ".join(errors) if errors else None
        self.backend_name = "+".join(item["name"] for item in loaded_backends) if loaded_backends else str(self.requested_backend_name)
        return loaded_backends

    def _load_backend_instance(self, backend):
        if backend == "mmocr":
            return MMOCRBackend(
                device=self.mmocr_device or ("cuda:0" if self.easyocr_gpu else "cpu"),
                det=self.mmocr_det,
                rec=self.mmocr_rec,
                batch_size=self.mmocr_batch_size,
                direct_recognition=self.mmocr_direct_recognition,
            )
        if backend == "mmocr_rec":
            return MMOCRRecognitionBackend(
                device=self.mmocr_device or ("cuda:0" if self.easyocr_gpu else "cpu"),
                rec=self.mmocr_rec,
                batch_size=self.mmocr_batch_size,
            )
        if backend == "easyocr":
            import easyocr

            return EasyOCRBackend(easyocr.Reader(["en"], gpu=self.easyocr_gpu), gpu=self.easyocr_gpu)
        if backend == "paddleocr":
            return PaddleOCRBackend(use_gpu=self.easyocr_gpu)
        if backend == "pytesseract":
            import pytesseract

            return TesseractBackend(pytesseract)
        raise ValueError("unsupported backend")

    def _load_template_matcher(self):
        if not self.template_matching:
            return None
        if not self.template_font_image:
            self.template_message = "template_font_image is not configured"
            return None
        font_image = self._resolve_template_font_image()
        if font_image is None:
            return None
        try:
            return JerseyTemplateMatcher(
                font_image,
                min_score=self.template_min_score,
                max_candidates=self.template_max_candidates,
            ).load()
        except Exception as exc:
            self.template_message = f"{type(exc).__name__}: {exc}"
            return None

    def _resolve_template_font_image(self):
        path = Path(self.template_font_image)
        candidates = [path] if path.is_absolute() else [
            Path.cwd() / path,
            Path(__file__).resolve().parents[2] / path,
        ]
        # Runs can start from the repository root or from an installed package.
        # Both locations are checked to avoid fragile relative paths on SSH.
        for candidate in candidates:
            if candidate.exists():
                return candidate
        self.template_message = "template font image not found; checked: " + ", ".join(str(item) for item in candidates)
        return None

    def _template_diagnostics(self):
        return {
            "enabled": self.template_matching,
            "status": self._template_status(),
            "font_image": str(self.template_font_image) if self.template_font_image else None,
            "min_score": self.template_min_score,
            "weight": self.template_weight,
            "max_candidates": self.template_max_candidates,
            "templates": sorted(self.template_matcher.templates) if self.template_matcher else [],
            "message": self.template_message,
        }

    def _template_status(self):
        if not self.template_matching:
            return "disabled"
        if self.template_matcher is not None:
            return "ok"
        return "unavailable"

    def _recognize_row(self, row, track_id, pass_index=0):
        cache_key = self._row_cache_key(row)
        cached = self.cache.get(cache_key) if cache_key else None
        if cached is not None:
            return [
                self._materialize_cached_detection(item, row, pass_index)
                for item in cached.get("detections", [])
            ]

        detections = []
        for backend in self.backends:
            reader = backend["reader"]
            backend_name = backend["name"]
            if getattr(reader, "uses_text_detection", False):
                variants = preprocess_text_detection_variants(
                    row["crop_path"],
                    number_roi_enabled=self.number_roi_enabled,
                    number_roi_upscale=self.number_roi_upscale,
                    number_roi_clahe=self.number_roi_clahe,
                    broadcast_contrast_enabled=self.broadcast_contrast_enabled,
                    broadcast_contrast_clip_limit=self.broadcast_contrast_clip_limit,
                    broadcast_contrast_tile_grid_size=self.broadcast_contrast_tile_grid_size,
                    super_resolution_enabled=self.super_resolution_enabled,
                    super_resolution_scale=self.super_resolution_scale,
                    super_resolution_max_side=self.super_resolution_max_side,
                )
            else:
                variants = preprocess_variants(
                    row["crop_path"],
                    augment=self.augment,
                    number_roi_enabled=self.number_roi_enabled,
                    number_roi_upscale=self.number_roi_upscale,
                    number_roi_clahe=self.number_roi_clahe,
                    broadcast_contrast_enabled=self.broadcast_contrast_enabled,
                    broadcast_contrast_clip_limit=self.broadcast_contrast_clip_limit,
                    broadcast_contrast_tile_grid_size=self.broadcast_contrast_tile_grid_size,
                    super_resolution_enabled=self.super_resolution_enabled,
                    super_resolution_scale=self.super_resolution_scale,
                    super_resolution_max_side=self.super_resolution_max_side,
                )
            detections.extend(
                self._recognize_backend_variants(
                    row,
                    track_id,
                    pass_index,
                    backend_name,
                    reader,
                    variants,
                )
            )

        if self.template_matcher is not None:
            backend_numbers = {
                int(item["number"])
                for item in detections
                if item.get("number") is not None
                and item.get("source") != "template"
                and float(item.get("confidence", 0.0) or 0.0) >= self.min_raw_confidence
            }
            for name, image in preprocess_variants(
                row["crop_path"],
                augment=self.augment,
                number_roi_enabled=self.number_roi_enabled,
                number_roi_upscale=self.number_roi_upscale,
                number_roi_clahe=self.number_roi_clahe,
                broadcast_contrast_enabled=self.broadcast_contrast_enabled,
                broadcast_contrast_clip_limit=self.broadcast_contrast_clip_limit,
                broadcast_contrast_tile_grid_size=self.broadcast_contrast_tile_grid_size,
                super_resolution_enabled=self.super_resolution_enabled,
                super_resolution_scale=self.super_resolution_scale,
                super_resolution_max_side=self.super_resolution_max_side,
            ):
                self._write_debug(track_id, row, name, image, pass_index)
                if not is_template_variant(name):
                    continue
                for candidate in self.template_matcher.match(image, variant_name=name):
                    if not template_candidate_allowed(candidate, backend_numbers):
                        continue
                    detections.append(
                        {
                            "source": "template",
                            "pass": pass_index,
                            "crop_path": row.get("crop_path"),
                            "frame": row.get("frame"),
                            "crop_quality": row.get("crop_quality", 0.0),
                            "variant": name,
                            "text": str(candidate["jersey_number"]),
                            "number": int(candidate["jersey_number"]),
                            "confidence": float(candidate["confidence"]),
                            "vote_weight": self.template_weight,
                            "template": candidate,
                        }
                    )
        if cache_key:
            self.cache.set(
                cache_key,
                {
                    "schema": 1,
                    "config": self._cache_config(),
                    "detections": [self._cacheable_detection(item) for item in detections],
                },
            )
        return detections

    def _row_cache_key(self, row):
        crop_path = row.get("crop_path")
        if not crop_path:
            return None
        try:
            crop_hash = hash_file(crop_path)
        except OSError:
            return None
        return stable_hash({"crop_hash": crop_hash, "config": self._cache_config()})

    def _cache_config(self):
        return {
            "backend": self.backend_name,
            "min_confidence": self.min_confidence,
            "augment": self.augment,
            "template_matching": self.template_matching,
            "template_font_image": str(self.template_font_image) if self.template_font_image else None,
            "template_min_score": self.template_min_score,
            "template_weight": self.template_weight,
            "template_max_candidates": self.template_max_candidates,
            "mmocr_det": self.mmocr_det,
            "mmocr_rec": self.mmocr_rec,
            "mmocr_direct_recognition": self.mmocr_direct_recognition,
            "direct_recognition_variant_policy": "number_roi_regions_v2",
            "demote_direct_only_single_digits": self.demote_direct_only_single_digits,
            "prefer_two_digit_candidates": self.prefer_two_digit_candidates,
            "number_roi_enabled": self.number_roi_enabled,
            "number_roi_upscale": self.number_roi_upscale,
            "number_roi_clahe": self.number_roi_clahe,
            "broadcast_contrast_enabled": self.broadcast_contrast_enabled,
            "broadcast_contrast_clip_limit": self.broadcast_contrast_clip_limit,
            "broadcast_contrast_tile_grid_size": self.broadcast_contrast_tile_grid_size,
            "super_resolution_enabled": self.super_resolution_enabled,
            "super_resolution_scale": self.super_resolution_scale,
            "super_resolution_max_side": self.super_resolution_max_side,
        }

    @staticmethod
    def _cacheable_detection(item):
        return {
            key: value
            for key, value in item.items()
            if key not in {"crop_path", "frame", "pass", "crop_quality", "crop_quality_vote_weight", "base_vote_weight"}
        }

    @staticmethod
    def _materialize_cached_detection(item, row, pass_index):
        out = dict(item)
        out["pass"] = pass_index
        out["crop_path"] = row.get("crop_path")
        out["frame"] = row.get("frame")
        out["crop_quality"] = row.get("crop_quality", 0.0)
        return out

    def _recognize_backend_variants(self, row, track_id, pass_index, backend_name, backend, variants):
        detections = []
        for name, image in variants:
            self._write_debug(track_id, row, name, image, pass_index)
            try:
                if getattr(backend, "supports_direct_recognition_control", False):
                    raw = backend.read(
                        image,
                        direct_recognition=(
                            self.mmocr_direct_recognition and should_direct_recognize_variant(name)
                        ),
                    )
                else:
                    raw = backend.read(image)
            except Exception as exc:
                detections.append(
                    {
                        "source": backend_name,
                        "pass": pass_index,
                        "crop_path": row.get("crop_path"),
                        "frame": row.get("frame"),
                        "crop_quality": row.get("crop_quality", 0.0),
                        "variant": name,
                        "number": None,
                        "confidence": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raw = []
            for item in raw:
                text, confidence, ocr_channel = normalize_ocr_result(item)
                number = parse_number(text)
                detections.append(
                    {
                        "source": backend_name,
                        "ocr_channel": ocr_channel,
                        "pass": pass_index,
                        "crop_path": row.get("crop_path"),
                        "frame": row.get("frame"),
                        "crop_quality": row.get("crop_quality", 0.0),
                        "variant": name,
                        "text": text,
                        "number": number,
                        "confidence": float(confidence or 0.0),
                    }
                )
        return detections

    def _write_debug(self, track_id, row, variant, image, pass_index=0):
        if self.debug_dir is None:
            return
        import cv2

        out_dir = self.debug_dir / f"track_{int(track_id):04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = int(row.get("frame", 0))
        cv2.imwrite(str(out_dir / f"pass_{pass_index:02d}_frame_{frame:06d}_{variant}.png"), image)


class EasyOCRBackend:
    def __init__(self, reader, gpu=False):
        self.reader = reader
        self.gpu = bool(gpu)

    def read(self, image):
        return [(text, conf) for _, text, conf in self.reader.readtext(np.asarray(image), allowlist="0123456789", detail=1)]


class TesseractBackend:
    def __init__(self, pytesseract):
        self.pytesseract = pytesseract

    def read(self, image):
        data = self.pytesseract.image_to_data(
            image,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
            output_type=self.pytesseract.Output.DICT,
        )
        rows = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            if not str(text).strip():
                continue
            try:
                score = max(0.0, min(1.0, float(conf) / 100.0))
            except ValueError:
                score = 0.0
            rows.append((text, score))
        return rows


class PaddleOCRBackend:
    """Optional PaddleOCR backend loaded only when explicitly requested."""

    def __init__(self, use_gpu=False):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=bool(use_gpu))

    def read(self, image):
        result = self.ocr.ocr(np.asarray(image), cls=False)
        rows = []
        if not result:
            return rows
        for page in result:
            for line in page or []:
                try:
                    text = line[1][0]
                    confidence = float(line[1][1])
                except Exception:
                    continue
                rows.append((text, confidence))
        return rows


class MMOCRBackend:
    uses_text_detection = True
    supports_direct_recognition_control = True

    def __init__(
        self,
        device="cuda:0",
        det="dbnet_resnet18_fpnc_1200e_icdar2015",
        rec="SAR",
        batch_size=8,
        direct_recognition=False,
    ):
        self.device = device
        self.det = normalize_mmocr_model_name(det, task="det")
        self.rec = normalize_mmocr_model_name(rec, task="rec")
        self.batch_size = int(batch_size)
        self.direct_recognition = bool(direct_recognition)
        self.mode = "mmocr_inferencer"
        try:
            from mmocr.apis import MMOCRInferencer

            # The high-level inferencer is the most stable public API for
            # end-to-end text detection + recognition in MMOCR 1.x. It also
            # returns serializable rec_texts/rec_scores, which keeps our OCR
            # diagnostics independent from MMOCR's internal datasample objects.
            self.inferencer = MMOCRInferencer(det=self.det, rec=self.rec, device=device)
        except Exception:
            self.mode = "standard_inferencers"
            from mmocr.apis import TextDetInferencer, TextRecInferencer
            from mmocr.utils import bbox2poly, crop_img, poly2bbox

            self.textdetinferencer = TextDetInferencer(self.det, device=device)
            self.textrecinferencer = TextRecInferencer(self.rec, device=device)
            self.bbox2poly = bbox2poly
            self.crop_img = crop_img
            self.poly2bbox = poly2bbox

    def read(self, image, direct_recognition=None):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.mode == "mmocr_inferencer":
            return self._read_with_mmocr_inferencer(image)
        return self._read_with_standard_inferencers(image, direct_recognition=direct_recognition)

    def _read_with_mmocr_inferencer(self, image):
        result = call_mmocr_inferencer_quiet(
            self.inferencer,
            [image],
            batch_size=1,
            det_batch_size=1,
            rec_batch_size=self.batch_size,
            out_dir="",
            return_vis=False,
            save_vis=False,
            save_pred=False,
            print_result=False,
        )
        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        rows = []
        for prediction in predictions:
            texts, scores = mmocr_prediction_text_scores(prediction)
            det_scores = prediction.get("det_scores", []) if isinstance(prediction, dict) else []
            for index, text in enumerate(texts):
                rec_score = scores[index] if index < len(scores) else None
                det_score = det_scores[index] if index < len(det_scores) else None
                confidence = combine_mmocr_scores(rec_score, det_score)
                rows.append((text, confidence))
        return rows

    def _read_with_standard_inferencers(self, image, direct_recognition=None):
        rec_inputs = []
        rec_channels = []
        use_direct_recognition = self.direct_recognition if direct_recognition is None else bool(direct_recognition)
        if use_direct_recognition:
            rec_inputs.append(image)
            rec_channels.append("direct")

        det_data = self.textdetinferencer(
            [image],
            return_datasamples=True,
            batch_size=1,
            progress_bar=False,
        )["predictions"][0]
        for polygon in pred_instance_polygons(det_data.pred_instances):
            quad = self.bbox2poly(self.poly2bbox(polygon)).tolist()
            rec_input = self.crop_img(image, quad)
            if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                continue
            rec_inputs.append(rec_input)
            rec_channels.append("detected_box")
        if not rec_inputs:
            return []

        rec_predictions = self.textrecinferencer(
            rec_inputs,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )["predictions"]
        rows = []
        for prediction, channel in zip(rec_predictions, rec_channels):
            result = self.textrecinferencer.pred2dict(prediction)
            text = result.get("text")
            confidence = mmocr_score(result.get("scores"))
            rows.append((text, confidence, channel))
        return rows


class MMOCRRecognitionBackend:
    uses_text_detection = False

    def __init__(self, device="cuda:0", rec="SAR", batch_size=8):
        self.device = device
        self.rec = normalize_mmocr_model_name(rec, task="rec")
        self.batch_size = int(batch_size)
        try:
            from mmocr.apis import MMOCRInferencer

            self.mode = "mmocr_inferencer"
            self.inferencer = MMOCRInferencer(rec=self.rec, device=device)
        except Exception:
            self.mode = "standard_inferencer"
            from mmocr.apis import TextRecInferencer

            self.textrecinferencer = TextRecInferencer(self.rec, device=device)

    def read(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.mode == "mmocr_inferencer":
            result = call_mmocr_inferencer_quiet(
                self.inferencer,
                [image],
                batch_size=1,
                rec_batch_size=self.batch_size,
                out_dir="",
                return_vis=False,
                save_vis=False,
                save_pred=False,
                print_result=False,
            )
            predictions = result.get("predictions", []) if isinstance(result, dict) else []
            rows = []
            for prediction in predictions:
                texts, scores = mmocr_prediction_text_scores(prediction)
                for index, text in enumerate(texts):
                    rows.append((text, mmocr_score(scores[index] if index < len(scores) else None)))
            return rows

        predictions = self.textrecinferencer(
            [image],
            return_datasamples=True,
            batch_size=1,
            progress_bar=False,
        )["predictions"]
        rows = []
        for prediction in predictions:
            result = self.textrecinferencer.pred2dict(prediction)
            rows.append((result.get("text"), mmocr_score(result.get("scores"))))
        return rows


def requested_backends(backend_name):
    name = str(backend_name or "auto").strip().lower()
    if name in ("auto", "default"):
        return ["easyocr", "pytesseract"]
    if name in ("paddleocr", "paddle_ocr", "paddle"):
        return ["paddleocr"]
    if name in ("paddleocr_easyocr", "paddleocr+easyocr", "paddle_ocr_easyocr"):
        return ["paddleocr", "easyocr", "pytesseract"]
    if name in ("mmocr_easyocr", "mmocr+easyocr", "mmocr_auto"):
        return ["mmocr", "easyocr", "pytesseract"]
    if name in ("mmocr_rec", "mmocr-rec", "mmocr_recognition"):
        return ["mmocr_rec"]
    if name in ("mmocr_rec_easyocr", "mmocr-rec-easyocr"):
        return ["mmocr_rec", "easyocr", "pytesseract"]
    if name in ("mmocr-fallback", "mmocr_fallback"):
        return ["mmocr", "easyocr", "pytesseract"]
    if "," in name:
        return [normalize_backend_name(part.strip()) for part in name.split(",") if part.strip()]
    if "+" in name:
        return [normalize_backend_name(part.strip()) for part in name.split("+") if part.strip()]
    return [normalize_backend_name(name)]


def backend_load_mode(backend_name):
    name = str(backend_name or "auto").strip().lower()
    if "," in name or "+" in name:
        return "combine"
    if name in ("mmocr_easyocr", "mmocr+easyocr", "mmocr_auto", "mmocr_rec_easyocr", "mmocr-rec-easyocr"):
        return "combine"
    return "fallback"


def normalize_backend_name(name):
    normalized = str(name or "").strip().lower().replace("-", "_")
    if normalized in {"mmocr_recognition", "mmocrrec"}:
        return "mmocr_rec"
    return normalized


def pred_instance_polygons(pred_instances):
    if pred_instances is None:
        return []
    try:
        return pred_instances["polygons"]
    except Exception:
        pass
    try:
        return pred_instances.get("polygons", [])
    except Exception:
        return getattr(pred_instances, "polygons", [])


def normalize_mmocr_model_name(model_name, task):
    value = str(model_name or "").strip()
    if not value:
        return None
    path_like = "/" in value or "\\" in value or value.endswith((".py", ".pth", ".pt"))
    if path_like:
        return value
    key = value.lower()
    det_aliases = {
        "dbnet": "DBNet",
    }
    rec_aliases = {
        "sar": "SAR",
        "satrn": "SATRN",
        "crnn": "CRNN",
        "nrtr": "NRTR",
        "robustscanner": "RobustScanner",
    }
    aliases = det_aliases if task == "det" else rec_aliases
    return aliases.get(key, value)


def call_mmocr_inferencer_quiet(inferencer, inputs, **kwargs):
    """Call MMOCRInferencer without per-crop rich progress bars.

    MMOCR versions differ in which quiet flags they accept. The first call
    disables known console outputs; the fallback preserves compatibility with
    older versions that reject one of those keyword arguments.
    """
    quiet_kwargs = {
        **kwargs,
        "progress_bar": False,
        "show": False,
        "print_result": False,
    }
    try:
        return inferencer(inputs, **quiet_kwargs)
    except TypeError:
        return inferencer(inputs, **kwargs)


def mmocr_prediction_text_scores(prediction):
    if not isinstance(prediction, dict):
        return [], []
    if "rec_texts" in prediction:
        texts = prediction.get("rec_texts") or []
        scores = prediction.get("rec_scores") or []
        return list(texts), list(scores)
    text = prediction.get("rec_text", prediction.get("text"))
    if text is None:
        return [], []
    score = prediction.get("rec_score", prediction.get("scores"))
    return [text], [score]


def combine_mmocr_scores(rec_score, det_score=None):
    rec = mmocr_score(rec_score)
    if det_score is None:
        return rec
    det = mmocr_score(det_score)
    if det <= 0:
        return rec
    # Recognition confidence should dominate; detection confidence only nudges
    # the score down when MMOCR localized uncertain text on a noisy jersey crop.
    return float(max(0.0, min(1.0, rec * (0.75 + 0.25 * det))))


def mmocr_score(scores):
    if scores is None:
        return 0.0
    try:
        array = np.asarray(scores, dtype=np.float32).reshape(-1)
    except Exception:
        try:
            return float(scores)
        except Exception:
            return 0.0
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(max(0.0, min(1.0, float(array.mean()))))


def normalize_ocr_result(item):
    if isinstance(item, dict):
        return item.get("text"), item.get("confidence", 0.0), item.get("ocr_channel", "unknown")
    if isinstance(item, (list, tuple)):
        if len(item) >= 3:
            return item[0], item[1], item[2]
        if len(item) >= 2:
            return item[0], item[1], "unknown"
        if len(item) == 1:
            return item[0], 0.0, "unknown"
    return item, 0.0, "unknown"


def select_spread_crops(items, max_count, pass_index=0, pass_count=1):
    if max_count <= 0:
        return []
    target = min(max_count, len(items))
    ordered = sorted(items, key=lambda row: int(row.get("frame", 0)))
    selected = []
    selected_paths = set()
    pass_count = max(1, int(pass_count))
    pass_index = max(0, min(int(pass_index), pass_count - 1))
    bucket_count = min(len(ordered), target * pass_count)
    for index in range(target):
        bucket_index = min(bucket_count - 1, index * pass_count + pass_index)
        start = int(round(bucket_index * len(ordered) / bucket_count))
        end = int(round((bucket_index + 1) * len(ordered) / bucket_count))
        bucket = ordered[start : max(start + 1, end)]
        best = max(bucket, key=lambda row: float(row.get("crop_quality", 0.0)))
        if best.get("crop_path") not in selected_paths:
            selected.append(best)
            selected_paths.add(best.get("crop_path"))
    top_count = max(1, target // 2)
    ranked_rows = sorted(ordered, key=ocr_crop_score, reverse=True)
    for row in ranked_rows[:top_count]:
        if row.get("crop_path") not in selected_paths:
            selected.append(row)
            selected_paths.add(row.get("crop_path"))
    return sorted(selected[: max_count * 2], key=lambda row: int(row.get("frame", 0)))


def ocr_crop_score(row):
    quality = float(row.get("crop_quality", 0.0) or 0.0)
    bbox = row.get("bbox")
    area_bonus = 0.0
    aspect_bonus = 0.0
    if isinstance(bbox, str):
        try:
            import json

            bbox = json.loads(bbox)
        except Exception:
            bbox = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        width = max(1.0, float(bbox[2]) - float(bbox[0]))
        height = max(1.0, float(bbox[3]) - float(bbox[1]))
        area_bonus = min(0.4, (width * height) / 50000.0)
        aspect = height / width
        aspect_bonus = 0.15 if 1.6 <= aspect <= 4.5 else 0.0
    return quality + area_bonus + aspect_bonus


def filter_quality_items(items, min_crop_quality):
    if min_crop_quality <= 0:
        return list(items)
    filtered = [row for row in items if float(row.get("crop_quality", 0.0) or 0.0) >= min_crop_quality]
    return filtered if len(filtered) >= 3 else list(items)


def crop_selection_diagnostics(items, usable_items, selected):
    selected_keys = {
        (int(pass_index), str(row.get("crop_path")), int(row.get("frame", 0)))
        for pass_index, row in selected
    }
    quality_values = [float(row.get("crop_quality", 0.0) or 0.0) for row in items]
    return {
        "available": len(items),
        "usable": len(usable_items),
        "selected": len(selected),
        "unselected": max(0, len(usable_items) - len(selected_keys)),
        "min_quality": min(quality_values) if quality_values else None,
        "max_quality": max(quality_values) if quality_values else None,
        "mean_quality": float(sum(quality_values) / len(quality_values)) if quality_values else None,
    }


def apply_crop_quality_vote_weights(detections, selected, enabled=False, min_weight=0.35, power=1.0):
    """Attach a relative crop-quality weight to OCR detections before voting.

    The OCR backend can emit many correlated variants for the same crop. This
    weighting keeps the best selected crop at full strength and downweights the
    weakest selected crops, so noisy frames contribute diagnostics without
    dominating tracklet-level voting.
    """
    if not enabled or not detections:
        return detections

    quality_by_key = selected_crop_quality_by_detection_key(selected)
    quality_values = list(quality_by_key.values())
    if not quality_values:
        return detections

    min_quality = min(quality_values)
    max_quality = max(quality_values)
    if max_quality <= min_quality + 1e-9:
        return [
            {
                **item,
                "crop_quality": quality_for_detection(item, quality_by_key),
                "crop_quality_vote_weight": 1.0,
                "base_vote_weight": float(item.get("vote_weight", 1.0) or 0.0),
            }
            for item in detections
        ]

    min_weight = max(0.0, min(1.0, float(min_weight)))
    power = max(0.01, float(power))
    weighted = []
    for item in detections:
        quality = quality_for_detection(item, quality_by_key)
        if quality is None:
            quality_weight = 1.0
        else:
            relative = (float(quality) - min_quality) / (max_quality - min_quality)
            relative = max(0.0, min(1.0, relative))
            quality_weight = min_weight + (1.0 - min_weight) * (relative ** power)
        base_weight = max(0.0, float(item.get("base_vote_weight", item.get("vote_weight", 1.0)) or 0.0))
        out = dict(item)
        out["crop_quality"] = quality
        out["crop_quality_vote_weight"] = float(quality_weight)
        out["base_vote_weight"] = float(base_weight)
        out["vote_weight"] = float(base_weight * quality_weight)
        weighted.append(out)
    return weighted


def selected_crop_quality_by_detection_key(selected):
    quality_by_key = {}
    for pass_index, row in selected:
        quality = safe_float(row.get("crop_quality"), default=0.0)
        crop_path = row.get("crop_path")
        if crop_path:
            quality_by_key[("crop", str(crop_path))] = quality
        quality_by_key[("frame", row.get("frame"), pass_index)] = quality
    return quality_by_key


def quality_for_detection(item, quality_by_key):
    value = safe_float(item.get("crop_quality"), default=None)
    if value is not None:
        return value
    return quality_by_key.get(detection_crop_key(item))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def preprocess_variants(
    crop_path,
    augment=True,
    number_roi_enabled=False,
    number_roi_upscale=3,
    number_roi_clahe=True,
    broadcast_contrast_enabled=False,
    broadcast_contrast_clip_limit=4.0,
    broadcast_contrast_tile_grid_size=4,
    super_resolution_enabled=False,
    super_resolution_scale=4,
    super_resolution_max_side=100,
):
    import cv2

    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        return []
    image = super_resolve_small_crop(
        image,
        cv2,
        enabled=super_resolution_enabled,
        scale=super_resolution_scale,
        max_side=super_resolution_max_side,
    )
    h, w = image.shape[:2]
    regions = number_roi_regions() if number_roi_enabled else legacy_ocr_regions()
    variants = []
    for name, (top_r, bottom_r, left_r, right_r) in regions.items():
        top = int(h * top_r)
        bottom = max(top + 1, int(h * bottom_r))
        left = int(w * left_r)
        right = max(left + 1, int(w * right_r))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        scale = int(number_roi_upscale) if number_roi_enabled else (3 if max(crop.shape[:2]) < 160 else 2)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        variants.append((f"{name}_equalized", equalized))
        if augment:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if number_roi_clahe else equalized
            variants.append((f"{name}_clahe", clahe))
            if broadcast_contrast_enabled:
                broadcast = broadcast_contrast_gray(
                    crop,
                    cv2,
                    clip_limit=broadcast_contrast_clip_limit,
                    tile_grid_size=broadcast_contrast_tile_grid_size,
                )
                variants.append((f"{name}_broadcast_luma", broadcast))
                broadcast_sharp = cv2.filter2D(
                    broadcast,
                    -1,
                    np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
                )
                variants.append((f"{name}_broadcast_luma_sharpened", broadcast_sharp))
            sharpened = cv2.filter2D(equalized, -1, np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
            variants.append((f"{name}_sharpened", sharpened))
            blur = cv2.GaussianBlur(equalized, (3, 3), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append((f"{name}_binary", binary))
            inverted = cv2.bitwise_not(binary)
            variants.append((f"{name}_binary_inv", inverted))
            adaptive = cv2.adaptiveThreshold(
                equalized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            variants.append((f"{name}_adaptive", adaptive))
    return variants


def preprocess_text_detection_variants(
    crop_path,
    number_roi_enabled=False,
    number_roi_upscale=3,
    number_roi_clahe=True,
    broadcast_contrast_enabled=False,
    broadcast_contrast_clip_limit=4.0,
    broadcast_contrast_tile_grid_size=4,
    super_resolution_enabled=False,
    super_resolution_scale=4,
    super_resolution_max_side=100,
):
    import cv2

    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        return []
    image = super_resolve_small_crop(
        image,
        cv2,
        enabled=super_resolution_enabled,
        scale=super_resolution_scale,
        max_side=super_resolution_max_side,
    )
    h, w = image.shape[:2]
    regions = number_roi_regions(prefix="mmocr") if number_roi_enabled else legacy_mmocr_regions()
    variants = []
    for name, (top_r, bottom_r, left_r, right_r) in regions.items():
        top = int(h * top_r)
        bottom = max(top + 1, int(h * bottom_r))
        left = int(w * left_r)
        right = max(left + 1, int(w * right_r))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        scale = int(number_roi_upscale) if number_roi_enabled else (3 if max(crop.shape[:2]) < 160 else 2)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        variants.append((name, crop))
        if number_roi_enabled and number_roi_clahe:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
            clahe = cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
            variants.append((f"{name}_clahe", clahe))
        if broadcast_contrast_enabled:
            variants.append(
                (
                    f"{name}_broadcast_luma",
                    broadcast_contrast_bgr(
                        crop,
                        cv2,
                        clip_limit=broadcast_contrast_clip_limit,
                        tile_grid_size=broadcast_contrast_tile_grid_size,
                    ),
                )
            )
        if number_roi_enabled:
            sharpened = cv2.filter2D(
                crop,
                -1,
                np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
            )
            variants.append((f"{name}_sharpened", sharpened))
    return variants


def super_resolve_small_crop(crop, cv2, enabled=False, scale=4, max_side=100):
    """Upscale small crops before ROI generation using OpenCV cubic fallback."""
    if not enabled or crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if max(h, w) >= int(max_side):
        return crop
    scale = max(1, int(scale))
    if scale <= 1:
        return crop
    return cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def broadcast_contrast_gray(crop, cv2, clip_limit=4.0, tile_grid_size=4):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    tile_grid = (int(tile_grid_size), int(tile_grid_size))
    enhanced = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid).apply(l_chan)
    return enhanced


def broadcast_contrast_bgr(crop, cv2, clip_limit=4.0, tile_grid_size=4):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    tile_grid = (int(tile_grid_size), int(tile_grid_size))
    enhanced_l = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid).apply(l_chan)
    return cv2.cvtColor(cv2.merge([enhanced_l, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def legacy_ocr_regions():
    return {
        "full_body": (0.00, 1.00, 0.00, 1.00),
        "back_number_wide": (0.12, 0.72, 0.04, 0.96),
        "back_number_mid": (0.18, 0.70, 0.12, 0.88),
        "full_upper": (0.04, 0.82, 0.06, 0.94),
        "upper": (0.08, 0.66, 0.10, 0.90),
        "torso": (0.20, 0.78, 0.18, 0.82),
        "center_torso": (0.24, 0.74, 0.28, 0.72),
        "lower_torso": (0.34, 0.86, 0.18, 0.82),
    }


def legacy_mmocr_regions():
    return {
        "mmocr_full_body": (0.00, 1.00, 0.00, 1.00),
        "mmocr_back_number_wide": (0.10, 0.76, 0.04, 0.96),
        "mmocr_upper": (0.06, 0.70, 0.08, 0.92),
        "mmocr_torso": (0.18, 0.78, 0.16, 0.84),
    }


def number_roi_regions(prefix="roi"):
    return {
        f"{prefix}_full_body": (0.00, 1.00, 0.00, 1.00),
        f"{prefix}_upper_back": (0.18, 0.55, 0.20, 0.80),
        f"{prefix}_center_back": (0.20, 0.50, 0.25, 0.75),
        f"{prefix}_torso": (0.12, 0.62, 0.15, 0.85),
        f"{prefix}_number_band": (0.24, 0.48, 0.18, 0.82),
    }


def is_ocr_player_row(row):
    role = str(row.get("role_detection") or "").lower()
    if role in {"referee", "referee_candidate"}:
        return False
    try:
        if int(row.get("semantic_group_id") or 0) == 5:
            return False
    except (TypeError, ValueError):
        pass
    return True


def jersey_group_key(row, segment_frames=0):
    display_id = int(row.get("display_track_id", row["track_id"]))
    segment_frames = int(segment_frames or 0)
    if segment_frames <= 0:
        return display_id
    frame = int(row.get("frame", 0) or 0)
    return (display_id, frame // segment_frames)


def split_jersey_group_key(key):
    if isinstance(key, tuple):
        return int(key[0]), int(key[1])
    return int(key), None


def segment_start_frame(segment_index, segment_frames):
    if segment_index is None or int(segment_frames or 0) <= 0:
        return None
    return int(segment_index) * int(segment_frames)


def segment_end_frame(segment_index, segment_frames):
    if segment_index is None or int(segment_frames or 0) <= 0:
        return None
    return (int(segment_index) + 1) * int(segment_frames) - 1


def diagnostic_key(key):
    if isinstance(key, tuple):
        return f"{int(key[0])}:{int(key[1])}"
    return str(int(key))


def parse_number(text):
    if text is None:
        return None
    # MMOCR's SoccerNet baseline keeps digit characters and truncates to two
    # digits because football jersey numbers are bounded to 1..99. Doing the
    # same here lets reads like "17I" still contribute "17" without accepting
    # zero or impossible jersey values.
    match = re.search(r"\d+", str(text))
    if not match:
        return None
    value = int(match.group(0)[:2])
    if value < 1 or value > 99:
        return None
    return value


def vote_numbers(detections, min_raw_confidence=0.0, digit_confusion_overrides=None):
    valid = [
        item
        for item in detections
        if item.get("number") is not None
        and float(item.get("confidence", 0.0) or 0.0) >= float(min_raw_confidence)
        and not item.get("direct_only_single_digit")
    ]
    if not valid:
        return None
    scores = defaultdict(float)
    counts = Counter()
    for item in valid:
        number = int(item["number"])
        vote_weight = max(0.0, float(item.get("vote_weight", 1.0) or 0.0))
        scores[number] += max(0.01, float(item.get("confidence", 0.0))) * vote_weight
        counts[number] += 1
    ranked = sorted(scores.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
    number, score = apply_digit_confusion_override(ranked, digit_confusion_overrides)
    total = sum(scores.values())
    runner_up = max((candidate_score for candidate, candidate_score in ranked if int(candidate) != int(number)), default=0.0)
    head_total = score + runner_up
    return {
        "jersey_number": int(number),
        "confidence": float(score / total) if total else 0.0,
        "head_confidence": float(score / head_total) if head_total else 1.0,
        "winner_margin": float((score - runner_up) / total) if total else 0.0,
        "winner_score": float(score),
        "runner_up_score": float(runner_up),
        "winner_score_ratio": float(score / runner_up) if runner_up > 0 else None,
        "votes": int(counts[number]),
        "total_detections": len(valid),
        "candidates": [
            {
                "jersey_number": int(candidate),
                "confidence": float(candidate_score / total) if total else 0.0,
                "votes": int(counts[candidate]),
                "score": float(candidate_score),
            }
            for candidate, candidate_score in ranked[:8]
        ],
    }


def apply_digit_confusion_override(ranked, overrides=None):
    if not ranked:
        return None, 0.0
    winner, winner_score = ranked[0]
    override = (overrides or {}).get(str(int(winner))) or (overrides or {}).get(int(winner))
    if not override:
        return int(winner), float(winner_score)
    preferred = int(override.get("prefer", winner))
    if preferred == int(winner):
        return int(winner), float(winner_score)
    scores = {int(number): float(score) for number, score in ranked}
    preferred_score = scores.get(preferred)
    if preferred_score is None:
        return int(winner), float(winner_score)
    min_ratio = float(override.get("min_alternative_ratio", 0.0) or 0.0)
    if preferred_score < float(winner_score) * min_ratio:
        return int(winner), float(winner_score)
    max_winner_ratio = override.get("max_winner_ratio")
    if max_winner_ratio is not None and float(winner_score) / max(0.01, preferred_score) > float(max_winner_ratio):
        return int(winner), float(winner_score)
    return int(preferred), float(preferred_score)


def normalize_digit_confusion_overrides(overrides):
    normalized = {}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if isinstance(value, int):
            normalized[str(int(key))] = {"prefer": int(value)}
            continue
        if not isinstance(value, dict):
            continue
        if "prefer" not in value:
            continue
        item = {"prefer": int(value["prefer"])}
        if "min_alternative_ratio" in value:
            item["min_alternative_ratio"] = float(value["min_alternative_ratio"])
        if "max_winner_ratio" in value:
            item["max_winner_ratio"] = float(value["max_winner_ratio"])
        normalized[str(int(key))] = item
    return normalized


def ocr_decision_diagnostics(detections, voting_detections, voted, min_votes=2, min_raw_confidence=0.0):
    raw_reasons = Counter()
    raw_channels = Counter()
    raw_variants = Counter()
    raw_numbers = Counter()
    for item in detections:
        raw_channels[str(item.get("ocr_channel", "unknown"))] += 1
        if item.get("variant") is not None:
            raw_variants[str(item.get("variant"))] += 1
        number = item.get("number")
        if number is None:
            if item.get("error"):
                raw_reasons["backend_error"] += 1
            elif item.get("text") in (None, ""):
                raw_reasons["empty_text"] += 1
            else:
                raw_reasons["non_numeric_text"] += 1
            continue
        raw_numbers[int(number)] += 1
        if float(item.get("confidence", 0.0) or 0.0) < float(min_raw_confidence):
            raw_reasons["below_min_raw_confidence"] += 1

    voting_reasons = Counter()
    voting_numbers = Counter()
    voting_channels = Counter()
    for item in voting_detections:
        number = item.get("number")
        if number is not None:
            voting_numbers[int(number)] += 1
        for channel in item.get("raw_channels") or [item.get("ocr_channel", "unknown")]:
            voting_channels[str(channel)] += 1
        if item.get("direct_only_single_digit"):
            voting_reasons["direct_only_single_digit"] += 1
        if float(item.get("confidence", 0.0) or 0.0) < float(min_raw_confidence):
            voting_reasons["below_min_raw_confidence"] += 1

    if voted is None:
        if not detections:
            status = "no_raw_detections"
        elif not voting_detections:
            status = "no_voting_detections"
        elif all(item.get("direct_only_single_digit") for item in voting_detections):
            status = "only_direct_single_digit_candidates"
        else:
            status = "no_valid_vote"
    elif int(voted.get("votes", 0) or 0) < int(min_votes):
        status = "insufficient_votes"
    else:
        status = "assigned"

    return {
        "status": status,
        "min_votes": int(min_votes),
        "min_raw_confidence": float(min_raw_confidence),
        "raw_rejection_reasons": dict(raw_reasons),
        "voting_rejection_reasons": dict(voting_reasons),
        "raw_channel_counts": dict(raw_channels),
        "voting_channel_counts": dict(voting_channels),
        "raw_number_counts": {str(k): v for k, v in raw_numbers.items()},
        "voting_number_counts": {str(k): v for k, v in voting_numbers.items()},
        "top_raw_variants": raw_variants.most_common(12),
    }


def aggregate_detections_by_crop(
    detections,
    min_raw_confidence=0.0,
    max_candidates_per_crop=3,
    min_candidate_ratio=0.35,
    demote_direct_only_single_digits_enabled=True,
    prefer_two_digit_candidates_enabled=True,
):
    grouped = defaultdict(list)
    for item in detections:
        if item.get("number") is None:
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if confidence < float(min_raw_confidence):
            continue
        grouped[detection_crop_key(item)].append(item)

    aggregated = []
    for key, items in sorted(grouped.items(), key=lambda entry: str(entry[0])):
        candidates = aggregate_crop_candidates(items)
        if demote_direct_only_single_digits_enabled:
            candidates = demote_direct_only_single_digits(candidates)
        if prefer_two_digit_candidates_enabled:
            candidates = prefer_two_digit_voting_candidates(candidates)
        candidates = filter_crop_candidates(candidates, max_candidates_per_crop, min_candidate_ratio)
        aggregated.extend(candidates)
    return aggregated


def aggregate_crop_candidates(items):
    by_number = defaultdict(list)
    for item in items:
        number = int(item["number"])
        confidence = max(0.01, float(item.get("confidence", 0.0) or 0.0))
        weight = max(0.0, float(item.get("vote_weight", 1.0) or 0.0))
        if weight <= 0:
            continue
        weighted_confidence = confidence * weight
        by_number[number].append((weighted_confidence, item))

    candidates = []
    for number, observations in by_number.items():
        observations = sorted(observations, key=lambda pair: pair[0], reverse=True)
        best_weighted, best_item = observations[0]
        candidates.append(
            {
                "source": "aggregate",
                "crop_path": best_item.get("crop_path"),
                "frame": best_item.get("frame"),
                "pass": best_item.get("pass"),
                "variant": best_item.get("variant"),
                "number": int(number),
                "text": str(number),
                "confidence": float(min(1.0, best_weighted)),
                "raw_observation_count": len(observations),
                "raw_best_confidence": float(best_item.get("confidence", 0.0) or 0.0),
                "raw_crop_quality": best_item.get("crop_quality"),
                "crop_quality_vote_weight": best_item.get("crop_quality_vote_weight"),
                "base_vote_weight": best_item.get("base_vote_weight"),
                "raw_sources": sorted({str(item.get("source", "ocr")) for _, item in observations}),
                "raw_channels": sorted({str(item.get("ocr_channel", "unknown")) for _, item in observations}),
                "raw_variants": [
                    str(item.get("variant"))
                    for _, item in observations[:6]
                    if item.get("variant") is not None
                ],
            }
        )
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)


def demote_direct_only_single_digits(candidates):
    demoted = []
    for candidate in candidates:
        item = dict(candidate)
        number = int(item.get("number", 0) or 0)
        channels = set(item.get("raw_channels") or [])
        if 1 <= number <= 9 and channels == {"direct"}:
            item["direct_only_single_digit"] = True
            item["confidence"] = min(float(item.get("confidence", 0.0) or 0.0), 0.01)
        demoted.append(item)
    return sorted(demoted, key=lambda item: item["confidence"], reverse=True)


def detection_crop_key(item):
    crop_path = item.get("crop_path")
    if crop_path:
        return ("crop", str(crop_path))
    return ("frame", item.get("frame"), item.get("pass"))


def prefer_two_digit_voting_candidates(candidates):
    two_digit = [candidate for candidate in candidates if int(candidate["number"]) >= 10]
    return two_digit if two_digit else candidates


def filter_crop_candidates(candidates, max_candidates_per_crop, min_candidate_ratio):
    if not candidates:
        return []
    max_candidates_per_crop = max(1, int(max_candidates_per_crop))
    best_score = max(0.01, float(candidates[0].get("confidence", 0.0) or 0.0))
    filtered = [
        candidate
        for candidate in candidates
        if float(candidate.get("confidence", 0.0) or 0.0) >= best_score * float(min_candidate_ratio)
    ]
    return filtered[:max_candidates_per_crop]


def template_candidate_allowed(candidate, backend_numbers):
    number = int(candidate.get("jersey_number"))
    if number in backend_numbers:
        return True
    # SoccerNet-GSR eval exposed many template-only false positives on "1":
    # thin limbs, shirt folds and pitch lines often resemble a vertical digit.
    # Let templates introduce standalone evidence only for strong two-digit
    # shapes; single digits need OCR agreement.
    if number < 10:
        return False
    return float(candidate.get("confidence", 0.0) or 0.0) >= 0.78


def is_template_variant(name):
    if "full_body" in str(name):
        return False
    return any(part in str(name) for part in ("back_number", "upper", "torso"))


def should_direct_recognize_variant(name):
    name = str(name)
    if "full_body" in name:
        return False
    return any(part in name for part in ("upper_back", "center_back", "torso", "number_band"))
