import numpy as np


class VisualFeatureExtractor:
    """Lightweight visual descriptor for color/appearance continuity.

    This is not a deep sports ReID model. It is a deterministic baseline that
    can use the roster JSON if visual profiles are provided, and gives the
    Hungarian assignment a weak physical/visual cue.
    """

    def add_row_features(self, rows):
        for row in rows:
            crop_path = row.get("crop_path")
            row["visual_embedding"] = self.extract(crop_path) if crop_path else None
        return rows

    @staticmethod
    def extract(crop_path):
        import cv2

        image = cv2.imread(str(crop_path))
        return extract_from_crop(image)

    @staticmethod
    def extract_from_crop(crop):
        return extract_from_crop(crop)

    @staticmethod
    def extract_from_frame(frame, bbox):
        return extract_from_frame(frame, bbox)


def extract_from_frame(frame, bbox):
    if frame is None or bbox is None:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(bbox[0])))
    y1 = max(0, min(height - 1, int(bbox[1])))
    x2 = max(0, min(width, int(bbox[2])))
    y2 = max(0, min(height, int(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return extract_from_crop(frame[y1:y2, x1:x2])


def extract_from_crop(crop):
    if crop is None or crop.size == 0:
        return None
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
