import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
        mmocr_device=None,
        mmocr_det="dbnet_resnet18_fpnc_1200e_icdar2015",
        mmocr_rec="SAR",
        mmocr_batch_size=8,
        progress_every=5,
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
        self.mmocr_device = mmocr_device
        self.mmocr_det = str(mmocr_det)
        self.mmocr_rec = str(mmocr_rec)
        self.mmocr_batch_size = int(mmocr_batch_size)
        self.progress_every = int(progress_every or 0)
        self.backends = []
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
                grouped[int(row.get("display_track_id", row["track_id"]))].append(row)
        assignments = {}
        diagnostics = {}
        grouped_items = sorted(grouped.items())
        print(
            f"FT jersey OCR: start tracklets={len(grouped_items)}"
            f" backend={self.backend_name}"
            f" template={self.template_matching}",
            flush=True,
        )
        for index, (track_id, items) in enumerate(grouped_items, start=1):
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
                    detections.extend(self._recognize_row(row, track_id, pass_index))
            voting_detections = self._voting_detections(detections)
            voted = vote_numbers(voting_detections, min_raw_confidence=self.min_raw_confidence)
            diagnostics[str(track_id)] = {
                "available_crops": len(items),
                "usable_crops": len(usable_items),
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
            }
            if voted and voted["votes"] >= self.min_votes:
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
            },
            "template_matching": self._template_diagnostics(),
            "tracklets": diagnostics,
            "assigned_tracklets": {str(k): v for k, v in assignments.items()},
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
        detections = []
        for backend in self.backends:
            reader = backend["reader"]
            backend_name = backend["name"]
            if getattr(reader, "uses_text_detection", False):
                variants = preprocess_text_detection_variants(row["crop_path"])
            else:
                variants = preprocess_variants(row["crop_path"], augment=self.augment)
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
            for name, image in preprocess_variants(row["crop_path"], augment=self.augment):
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
                            "variant": name,
                            "text": str(candidate["jersey_number"]),
                            "number": int(candidate["jersey_number"]),
                            "confidence": float(candidate["confidence"]),
                            "vote_weight": self.template_weight,
                            "template": candidate,
                        }
                    )
        return detections

    def _recognize_backend_variants(self, row, track_id, pass_index, backend_name, backend, variants):
        detections = []
        for name, image in variants:
            self._write_debug(track_id, row, name, image, pass_index)
            try:
                raw = backend.read(image)
            except Exception as exc:
                detections.append(
                    {
                        "source": backend_name,
                        "pass": pass_index,
                        "crop_path": row.get("crop_path"),
                        "frame": row.get("frame"),
                        "variant": name,
                        "number": None,
                        "confidence": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raw = []
            for text, confidence in raw:
                detections.append(
                    {
                        "source": backend_name,
                        "pass": pass_index,
                        "crop_path": row.get("crop_path"),
                        "frame": row.get("frame"),
                        "variant": name,
                        "text": text,
                        "number": parse_number(text),
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


class MMOCRBackend:
    uses_text_detection = True

    def __init__(self, device="cuda:0", det="dbnet_resnet18_fpnc_1200e_icdar2015", rec="SAR", batch_size=8):
        self.device = device
        self.det = normalize_mmocr_model_name(det, task="det")
        self.rec = normalize_mmocr_model_name(rec, task="rec")
        self.batch_size = int(batch_size)
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

    def read(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.mode == "mmocr_inferencer":
            return self._read_with_mmocr_inferencer(image)
        return self._read_with_standard_inferencers(image)

    def _read_with_mmocr_inferencer(self, image):
        result = self.inferencer(
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

    def _read_with_standard_inferencers(self, image):
        det_data = self.textdetinferencer(
            [image],
            return_datasamples=True,
            batch_size=1,
            progress_bar=False,
        )["predictions"][0]
        rec_inputs = []
        for polygon in pred_instance_polygons(det_data.pred_instances):
            quad = self.bbox2poly(self.poly2bbox(polygon)).tolist()
            rec_input = self.crop_img(image, quad)
            if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                continue
            rec_inputs.append(rec_input)
        if not rec_inputs:
            return []

        rec_predictions = self.textrecinferencer(
            rec_inputs,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )["predictions"]
        rows = []
        for prediction in rec_predictions:
            result = self.textrecinferencer.pred2dict(prediction)
            text = result.get("text")
            confidence = mmocr_score(result.get("scores"))
            rows.append((text, confidence))
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
            result = self.inferencer(
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


def preprocess_variants(crop_path, augment=True):
    import cv2

    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    regions = {
        "full_body": (0.00, 1.00, 0.00, 1.00),
        "back_number_wide": (0.12, 0.72, 0.04, 0.96),
        "back_number_mid": (0.18, 0.70, 0.12, 0.88),
        "full_upper": (0.04, 0.82, 0.06, 0.94),
        "upper": (0.08, 0.66, 0.10, 0.90),
        "torso": (0.20, 0.78, 0.18, 0.82),
        "center_torso": (0.24, 0.74, 0.28, 0.72),
        "lower_torso": (0.34, 0.86, 0.18, 0.82),
    }
    variants = []
    for name, (top_r, bottom_r, left_r, right_r) in regions.items():
        top = int(h * top_r)
        bottom = max(top + 1, int(h * bottom_r))
        left = int(w * left_r)
        right = max(left + 1, int(w * right_r))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        scale = 3 if max(crop.shape[:2]) < 160 else 2
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        variants.append((f"{name}_equalized", equalized))
        if augment:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            variants.append((f"{name}_clahe", clahe))
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


def preprocess_text_detection_variants(crop_path):
    import cv2

    image = cv2.imread(str(crop_path))
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    regions = {
        "mmocr_full_body": (0.00, 1.00, 0.00, 1.00),
        "mmocr_back_number_wide": (0.10, 0.76, 0.04, 0.96),
        "mmocr_upper": (0.06, 0.70, 0.08, 0.92),
        "mmocr_torso": (0.18, 0.78, 0.16, 0.84),
    }
    variants = []
    for name, (top_r, bottom_r, left_r, right_r) in regions.items():
        top = int(h * top_r)
        bottom = max(top + 1, int(h * bottom_r))
        left = int(w * left_r)
        right = max(left + 1, int(w * right_r))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        scale = 3 if max(crop.shape[:2]) < 160 else 2
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        variants.append((name, crop))
    return variants


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


def vote_numbers(detections, min_raw_confidence=0.0):
    valid = [
        item
        for item in detections
        if item.get("number") is not None
        and float(item.get("confidence", 0.0) or 0.0) >= float(min_raw_confidence)
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
    number, score = ranked[0]
    total = sum(scores.values())
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
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


def aggregate_detections_by_crop(
    detections,
    min_raw_confidence=0.0,
    max_candidates_per_crop=3,
    min_candidate_ratio=0.35,
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
                "raw_sources": sorted({str(item.get("source", "ocr")) for _, item in observations}),
                "raw_variants": [
                    str(item.get("variant"))
                    for _, item in observations[:6]
                    if item.get("variant") is not None
                ],
            }
        )
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)


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
