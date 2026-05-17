import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ft.features.jersey_template import JerseyTemplateMatcher


class JerseyOCR:
    """Optional OCR over player crops, aggregated by display_track_id."""

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
        template_weight=0.25,
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
        self.backend = None
        self.message = None
        self.template_matcher = None
        self.template_message = None

    def recognize(self, rows):
        self.backend = self._load_backend()
        self.template_matcher = self._load_template_matcher()
        if self.backend is None and self.template_matcher is None:
            return {}, {
                "enabled": True,
                "status": "missing_backend",
                "backend": self.backend_name,
                "message": self.message,
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
            "status": "ok" if self.backend is not None else "template_only",
            "backend": self.backend_name if self.backend is not None else None,
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
        return aggregate_detections_by_crop(
            detections,
            min_raw_confidence=self.min_raw_confidence,
            max_candidates_per_crop=self.max_candidates_per_crop,
            min_candidate_ratio=self.min_crop_candidate_ratio,
        )

    def _load_backend(self):
        requested = requested_backends(self.backend_name)
        errors = []
        for backend in requested:
            if backend == "mmocr":
                try:
                    self.backend_name = "mmocr"
                    return MMOCRBackend(
                        device=self.mmocr_device or ("cuda:0" if self.easyocr_gpu else "cpu"),
                        det=self.mmocr_det,
                        rec=self.mmocr_rec,
                        batch_size=self.mmocr_batch_size,
                    )
                except Exception as exc:
                    errors.append(f"mmocr: {type(exc).__name__}: {exc}")
            elif backend == "easyocr":
                try:
                    import easyocr

                    self.backend_name = "easyocr"
                    return EasyOCRBackend(easyocr.Reader(["en"], gpu=self.easyocr_gpu), gpu=self.easyocr_gpu)
                except Exception as exc:
                    errors.append(f"easyocr: {type(exc).__name__}: {exc}")
            elif backend == "pytesseract":
                try:
                    import pytesseract

                    self.backend_name = "pytesseract"
                    return TesseractBackend(pytesseract)
                except Exception as exc:
                    errors.append(f"pytesseract: {type(exc).__name__}: {exc}")
            else:
                errors.append(f"{backend}: unsupported backend")
        self.message = "; ".join(errors)
        return None

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
        if self.backend is not None:
            if getattr(self.backend, "uses_text_detection", False):
                variants = preprocess_text_detection_variants(row["crop_path"])
            else:
                variants = preprocess_variants(row["crop_path"], augment=self.augment)
            detections.extend(self._recognize_backend_variants(row, track_id, pass_index, variants))

        if self.template_matcher is not None:
            for name, image in preprocess_variants(row["crop_path"], augment=self.augment):
                self._write_debug(track_id, row, name, image, pass_index)
                if not is_template_variant(name):
                    continue
                for candidate in self.template_matcher.match(image, variant_name=name):
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

    def _recognize_backend_variants(self, row, track_id, pass_index, variants):
        detections = []
        for name, image in variants:
            self._write_debug(track_id, row, name, image, pass_index)
            try:
                raw = self.backend.read(image)
            except Exception as exc:
                detections.append(
                    {
                        "source": self.backend_name,
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
                        "source": self.backend_name,
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
        from mmocr.apis import TextDetInferencer, TextRecInferencer
        from mmocr.utils import bbox2poly, crop_img, poly2bbox

        self.device = device
        self.det = det
        self.rec = rec
        self.batch_size = int(batch_size)
        self.textdetinferencer = TextDetInferencer(det, device=device)
        self.textrecinferencer = TextRecInferencer(rec, device=device)
        self.bbox2poly = bbox2poly
        self.crop_img = crop_img
        self.poly2bbox = poly2bbox

    def read(self, image):
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

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


def requested_backends(backend_name):
    name = str(backend_name or "auto").strip().lower()
    if name in ("auto", "default"):
        return ["easyocr", "pytesseract"]
    if name in ("mmocr_easyocr", "mmocr+easyocr", "mmocr-fallback", "mmocr_auto"):
        return ["mmocr", "easyocr", "pytesseract"]
    if "," in name:
        return [part.strip() for part in name.split(",") if part.strip()]
    if "+" in name:
        return [part.strip() for part in name.split("+") if part.strip()]
    return [name]


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
    match = re.search(r"(?<!\d)\d{1,2}(?!\d)", str(text))
    if not match:
        return None
    value = int(match.group(0))
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


def is_template_variant(name):
    if "full_body" in str(name):
        return False
    return any(part in str(name) for part in ("back_number", "upper", "torso"))
