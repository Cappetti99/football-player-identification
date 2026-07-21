"""YOLO detection stage, independent from the association backend."""

import os

from ft.tracking.observations import normalize_ultralytics_result


try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - validated at runtime
    YOLO = None


class YoloDetectionStage:
    def __init__(self, model_path, confidence=0.002):
        if YOLO is None:
            raise ImportError("Missing ultralytics. Install FT requirements first.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)
        self.confidence = float(confidence)

    def detect_batch(self, frames, frame_offset=0, half=False):
        results = self.model.predict(
            frames,
            conf=self.confidence,
            verbose=False,
            half=bool(half),
        )
        return [
            normalize_ultralytics_result(result, int(frame_offset) + index)
            for index, result in enumerate(results)
        ]
