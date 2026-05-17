from pathlib import Path

import numpy as np


class JerseyTemplateMatcher:
    """Weak jersey-number scorer based on digit silhouettes from a font sheet."""

    def __init__(
        self,
        font_image_path,
        min_score=0.62,
        max_candidates=4,
        template_size=(48, 72),
    ):
        self.font_image_path = Path(font_image_path)
        self.min_score = float(min_score)
        self.max_candidates = int(max_candidates)
        self.template_size = tuple(template_size)
        self.templates = {}

    def load(self):
        import cv2

        image = cv2.imread(str(self.font_image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError(f"could not read font image: {self.font_image_path}")
        self.templates = extract_digit_templates(image, template_size=self.template_size)
        if len(self.templates) != 10:
            raise ValueError(f"expected 10 digit templates, found {len(self.templates)}")
        return self

    def match(self, image, variant_name=None):
        if not self.templates:
            return []
        gray = ensure_gray_uint8(image)
        candidates = []
        for polarity, mask in threshold_polarities(gray):
            components = extract_digit_components(mask)
            candidates.extend(
                score_components(
                    components,
                    self.templates,
                    template_size=self.template_size,
                    polarity=polarity,
                    variant_name=variant_name,
                )
            )
        candidates = [
            candidate
            for candidate in candidates
            if candidate["jersey_number"] > 0 and candidate["confidence"] >= self.min_score
        ]
        candidates = prefer_two_digit_candidates(candidates)
        return dedupe_candidates(candidates, self.max_candidates)


def extract_digit_templates(image, template_size=(48, 72)):
    import cv2

    gray = ensure_gray_uint8(image)
    h, w = gray.shape[:2]
    top = int(h * 0.34)
    bottom = int(h * 0.75)
    zone = gray[top:bottom, :]
    _, mask = cv2.threshold(zone, 205, 255, cv2.THRESH_BINARY)
    mask = cleanup_mask(mask)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    min_height = max(40, int((bottom - top) * 0.15))
    min_width = max(20, int(w * 0.025))
    min_area = max(800, int(h * w * 0.001))
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if bh < min_height or bw < min_width or area < min_area:
            continue
        component_mask = labels[y : y + bh, x : x + bw] == label
        components.append(
            {
                "x": int(x),
                "y": int(y + top),
                "w": int(bw),
                "h": int(bh),
                "area": int(area),
                "cy": float(y + top + bh / 2.0),
                "mask": fill_holes(component_mask),
            }
        )

    rows = group_component_rows(components)
    if len(rows) < 2:
        return {}
    rows = sorted(rows[:2], key=lambda row: mean_value(row, "cy"))
    digit_rows = [("12345", rows[0]), ("67890", rows[1])]

    templates = {}
    for digits, row in digit_rows:
        row = sorted(row, key=lambda item: item["x"])[: len(digits)]
        if len(row) < len(digits):
            continue
        for digit, component in zip(digits, row):
            templates[int(digit)] = normalize_mask(component["mask"], template_size)
    return templates


def group_component_rows(components):
    if not components:
        return []
    ordered = sorted(components, key=lambda item: item["cy"])
    rows = []
    for component in ordered:
        placed = False
        for row in rows:
            mean_cy = mean_value(row, "cy")
            mean_h = mean_value(row, "h")
            if abs(component["cy"] - mean_cy) <= max(24.0, mean_h * 0.55):
                row.append(component)
                placed = True
                break
        if not placed:
            rows.append([component])
    rows = [row for row in rows if len(row) >= 5]
    return sorted(rows, key=lambda row: (-mean_value(row, "h"), mean_value(row, "cy")))


def score_components(components, templates, template_size=(48, 72), polarity=None, variant_name=None):
    scored = []
    normalized = []
    for component in components:
        component_mask = normalize_mask(component["mask"], template_size)
        digit_scores = sorted(
            (
                {
                    "digit": int(digit),
                    "score": mask_iou(component_mask, template),
                }
                for digit, template in templates.items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        if not digit_scores:
            continue
        best = digit_scores[0]
        normalized.append({**component, "digit_scores": digit_scores, "best": best})
        if best["digit"] > 0:
            scored.append(
                candidate_payload(
                    best["digit"],
                    best["score"],
                    [best],
                    [component],
                    polarity=polarity,
                    variant_name=variant_name,
                )
            )

    ordered = sorted(normalized, key=lambda item: item["x"])
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if not plausible_digit_pair(left, right):
                continue
            left_best = left["best"]
            right_best = right["best"]
            if left_best["digit"] == 0:
                continue
            number = int(left_best["digit"] * 10 + right_best["digit"])
            score = pair_score(left, right)
            score *= (left_best["score"] + right_best["score"]) / 2.0
            scored.append(
                candidate_payload(
                    number,
                    score,
                    [left_best, right_best],
                    [left, right],
                    polarity=polarity,
                    variant_name=variant_name,
                )
            )
    return scored


def extract_digit_components(mask):
    import cv2

    mask = cleanup_mask(mask)
    h, w = mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if bh < max(10, h * 0.12):
            continue
        if bw < max(5, w * 0.025) or bw > w * 0.55:
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 0.16 or aspect > 1.20:
            continue
        fill = area / max(1.0, float(bw * bh))
        if fill < 0.18:
            continue
        component_mask = labels[y : y + bh, x : x + bw] == label
        components.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "area": int(area),
                "cx": float(x + bw / 2.0),
                "cy": float(y + bh / 2.0),
                "mask": fill_holes(component_mask),
            }
        )
    return sorted(components, key=lambda item: item["area"], reverse=True)[:12]


def plausible_digit_pair(left, right):
    if right["x"] <= left["x"]:
        return False
    avg_h = (left["h"] + right["h"]) / 2.0
    height_ratio = min(left["h"], right["h"]) / max(left["h"], right["h"], 1.0)
    gap = right["x"] - (left["x"] + left["w"])
    center_delta = abs(left["cy"] - right["cy"])
    if height_ratio < 0.55:
        return False
    if center_delta > avg_h * 0.35:
        return False
    return -avg_h * 0.15 <= gap <= avg_h * 0.85


def pair_score(left, right):
    avg_h = (left["h"] + right["h"]) / 2.0
    center_penalty = min(0.25, abs(left["cy"] - right["cy"]) / max(avg_h, 1.0))
    height_penalty = 1.0 - min(left["h"], right["h"]) / max(left["h"], right["h"], 1.0)
    gap = max(0.0, right["x"] - (left["x"] + left["w"]))
    gap_penalty = min(0.20, gap / max(avg_h * 3.0, 1.0))
    return max(0.0, 1.0 - center_penalty - height_penalty - gap_penalty)


def candidate_payload(number, confidence, digits, components, polarity=None, variant_name=None):
    return {
        "jersey_number": int(number),
        "confidence": float(confidence),
        "digits": [
            {"digit": int(item["digit"]), "score": float(item["score"])}
            for item in digits
        ],
        "bbox": component_bbox(components),
        "polarity": polarity,
        "variant": variant_name,
    }


def component_bbox(components):
    x1 = min(item["x"] for item in components)
    y1 = min(item["y"] for item in components)
    x2 = max(item["x"] + item["w"] for item in components)
    y2 = max(item["y"] + item["h"] for item in components)
    return [int(x1), int(y1), int(x2), int(y2)]


def dedupe_candidates(candidates, max_candidates):
    best_by_number = {}
    for candidate in candidates:
        number = int(candidate["jersey_number"])
        current = best_by_number.get(number)
        if current is None or candidate["confidence"] > current["confidence"]:
            best_by_number[number] = candidate
    ranked = sorted(best_by_number.values(), key=lambda item: item["confidence"], reverse=True)
    return ranked[: max(1, int(max_candidates))]


def prefer_two_digit_candidates(candidates):
    two_digit = [candidate for candidate in candidates if int(candidate["jersey_number"]) >= 10]
    return two_digit if two_digit else candidates


def threshold_polarities(gray):
    import cv2

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(bright)
    return [("bright", bright > 0), ("dark", dark > 0)]


def cleanup_mask(mask):
    import cv2

    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask.astype(bool)


def fill_holes(mask):
    import cv2

    mask = (mask > 0)
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    max_hole_area = max(4, int(mask.size * 0.03))
    filled = mask.copy()
    h, w = mask.shape[:2]
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if touches_border or area > max_hole_area:
            continue
        filled[labels == label] = True
    return filled


def normalize_mask(mask, template_size=(48, 72)):
    import cv2

    mask = fill_holes(mask)
    ys, xs = np.where(mask)
    target_w, target_h = template_size
    canvas = np.zeros((target_h, target_w), dtype=bool)
    if len(xs) == 0 or len(ys) == 0:
        return canvas
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape[:2]
    scale = min((target_w - 4) / max(w, 1), (target_h - 4) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def ensure_gray_uint8(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def mean_value(items, key):
    return float(sum(float(item[key]) for item in items) / max(len(items), 1))
