import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


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
        self.backend = None
        self.message = None

    def recognize(self, rows):
        self.backend = self._load_backend()
        if self.backend is None:
            return {}, {
                "enabled": True,
                "status": "missing_backend",
                "backend": self.backend_name,
                "message": self.message,
                "tracklets": {},
            }
        grouped = defaultdict(list)
        for row in rows:
            if row.get("crop_path"):
                grouped[int(row.get("display_track_id", row["track_id"]))].append(row)
        assignments = {}
        diagnostics = {}
        for track_id, items in sorted(grouped.items()):
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
            voted = vote_numbers(detections, min_raw_confidence=self.min_raw_confidence)
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
                "voted": voted,
            }
            if voted and voted["votes"] >= self.min_votes:
                assignments[track_id] = voted
        return assignments, {
            "enabled": True,
            "status": "ok",
            "backend": self.backend_name,
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
            "tracklets": diagnostics,
            "assigned_tracklets": {str(k): v for k, v in assignments.items()},
        }

    def _load_backend(self):
        requested = [self.backend_name]
        if self.backend_name == "auto":
            requested = ["easyocr", "pytesseract"]
        errors = []
        for backend in requested:
            if backend == "easyocr":
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

    def _recognize_row(self, row, track_id, pass_index=0):
        variants = preprocess_variants(row["crop_path"], augment=self.augment)
        detections = []
        for name, image in variants:
            self._write_debug(track_id, row, name, image, pass_index)
            try:
                raw = self.backend.read(image)
            except Exception as exc:
                detections.append(
                    {
                        "pass": pass_index,
                        "crop_path": row.get("crop_path"),
                        "frame": row.get("frame"),
                        "variant": name,
                        "number": None,
                        "confidence": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            for text, confidence in raw:
                detections.append(
                    {
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
        scores[number] += max(0.01, float(item.get("confidence", 0.0)))
        counts[number] += 1
    ranked = sorted(scores.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
    number, score = ranked[0]
    total = sum(scores.values())
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "jersey_number": int(number),
        "confidence": float(score / total) if total else 0.0,
        "winner_margin": float((score - runner_up) / total) if total else 0.0,
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
