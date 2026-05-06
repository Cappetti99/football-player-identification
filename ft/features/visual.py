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
        if image is None or image.size == 0:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        h, w = image.shape[:2]
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
    return (mean / norm).astype(float).tolist()
