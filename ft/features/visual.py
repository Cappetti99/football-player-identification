import numpy as np


class VisualFeatureExtractor:
    """Lightweight visual descriptor for color/appearance continuity.

    This is not a deep sports ReID model. It is a deterministic baseline that
    can use the roster JSON if visual profiles are provided, and gives the
    Hungarian assignment a weak physical/visual cue.
    """

    def __init__(self, cache_enabled=True, embedding_mode="hsv"):
        self.cache_enabled = bool(cache_enabled)
        self.embedding_mode = normalize_embedding_mode(embedding_mode)
        self._cache = {}
        self.stats = {
            "cache_enabled": self.cache_enabled,
            "embedding_mode": self.embedding_mode,
            "computed": 0,
            "reused": 0,
        }

    def add_row_features(self, rows):
        for row in rows:
            crop_path = row.get("crop_path")
            row["visual_embedding"] = self.extract_cached(crop_path) if crop_path else None
        return rows

    def extract_cached(self, crop_path):
        key = str(crop_path)
        if self.cache_enabled and key in self._cache:
            self.stats["reused"] += 1
            return self._cache[key]
        embedding = self.extract(crop_path, mode=self.embedding_mode)
        self.stats["computed"] += 1
        if self.cache_enabled:
            self._cache[key] = embedding
        return embedding

    def diagnostics(self):
        return dict(self.stats)

    @staticmethod
    def extract(crop_path, mode="hsv"):
        import cv2

        image = cv2.imread(str(crop_path))
        return extract_from_crop(image, mode=mode)

    @staticmethod
    def extract_from_crop(crop, mode="hsv"):
        return extract_from_crop(crop, mode=mode)

    @staticmethod
    def extract_from_frame(frame, bbox, mode="hsv"):
        return extract_from_frame(frame, bbox, mode=mode)


def extract_from_frame(frame, bbox, mode="hsv"):
    if frame is None or bbox is None:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(bbox[0])))
    y1 = max(0, min(height - 1, int(bbox[1])))
    x2 = max(0, min(width, int(bbox[2])))
    y2 = max(0, min(height, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return extract_from_crop(frame[y1:y2, x1:x2], mode=mode)


def extract_from_crop(crop, mode="hsv"):
    if crop is None or crop.size == 0:
        return None
    mode = normalize_embedding_mode(mode)
    if mode == "hsv_lab_gradient":
        return extract_hsv_lab_gradient_from_crop(crop)
    return extract_hsv_from_crop(crop)


def extract_hsv_from_crop(crop):
    import cv2

    # HSV histograms are intentionally simple and deterministic. They are used
    # as weak continuity evidence, not as a replacement for a proper ReID model.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    h, w = crop.shape[:2]
    shape = np.asarray([w / max(1.0, h), h / 256.0], dtype=np.float32)
    vec = np.concatenate([hist_h, hist_s, hist_v, shape]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return None
    return (vec / norm).astype(float).tolist()


def extract_hsv_lab_gradient_from_crop(crop):
    """Extract a richer deterministic descriptor for appearance linking.

    This is still not a learned ReID model. It adds LAB chroma and gradient
    texture to the legacy HSV cue so same-team tracklets have slightly more
    appearance signal when explicitly enabled in config.
    """
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    non_pitch_mask = ~((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 35))
    mask = (non_pitch_mask.astype(np.uint8) * 255)
    hist_h = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).flatten()
    hist_a = cv2.calcHist([lab], [1], None, [8], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab], [2], None, [8], [0, 256]).flatten()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(gy, gx) * 180.0 / np.pi % 180.0
    hist_grad = np.histogram(angle, bins=8, range=(0, 180))[0].astype(np.float32)

    h, w = crop.shape[:2]
    shape = np.asarray([w / max(1.0, float(h)), h / 256.0], dtype=np.float32)
    vec = np.concatenate([hist_h, hist_a, hist_b, hist_grad, shape]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return None
    return (vec / norm).astype(float).tolist()


def normalize_embedding_mode(mode):
    value = str(mode or "hsv").strip().lower().replace("-", "_")
    if value in {"legacy", "hsv", "hsv_hist", "hsv_histogram"}:
        return "hsv"
    if value in {"hsv_lab_gradient", "lab_gradient", "rich", "enhanced"}:
        return "hsv_lab_gradient"
    raise ValueError(f"Unknown visual embedding mode: {mode!r}")


def mean_embedding(values):
    vectors = [np.asarray(value, dtype=np.float32) for value in values if value is not None]
    if not vectors:
        return None
    mean = np.mean(np.vstack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 0:
        return None
    # Re-normalize after averaging so cosine similarity stays comparable across
    # short and long tracklets.
    return (mean / norm).astype(float).tolist()


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return None
    return float(np.dot(a, b) / denom)
