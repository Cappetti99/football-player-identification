"""Detection-level records shared by tracking and semantic stages."""

from dataclasses import dataclass, field
from typing import Any, Optional


PLAYER_CLASSES = {"person", "player", "goalkeeper"}
REFEREE_CLASSES = {"referee"}
BALL_CLASSES = {"ball", "sports ball"}


@dataclass
class DetectionObservation:
    """One detector output before association or attribute aggregation."""

    frame: int
    bbox: list
    confidence: float
    class_id: int
    class_name: str
    role: str
    embedding: Optional[list] = None
    role_scores: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def athlete_payload(self):
        payload = {
            "bbox": [float(value) for value in self.bbox],
            "detection_confidence": float(self.confidence),
            "class_name": self.class_name,
            "role_detection": self.role,
        }
        if self.embedding is not None:
            payload["visual_embedding"] = self.embedding
        if self.role_scores is not None:
            payload["reid_role_scores"] = self.role_scores
        if self.metadata:
            payload["detection_features"] = dict(self.metadata)
        return payload

    @property
    def is_ball(self):
        return self.class_name in BALL_CLASSES

    @property
    def is_athlete(self):
        return self.class_name in PLAYER_CLASSES or self.class_name in REFEREE_CLASSES


@dataclass
class FrameDetections:
    """Normalized detections for one frame plus a transitional native result."""

    frame: int
    observations: list
    image_shape: tuple
    native_result: Any = None


def role_for_class(class_name):
    name = str(class_name).lower()
    if name in REFEREE_CLASSES:
        return "referee"
    if name == "goalkeeper":
        return "goalkeeper"
    if name in PLAYER_CLASSES:
        return "player"
    if name in BALL_CLASSES:
        return "ball"
    return "other"


def normalize_ultralytics_result(result, frame):
    """Convert an Ultralytics result into backend-independent observations."""
    names = {int(key): str(value).lower() for key, value in result.names.items()}
    boxes = getattr(result, "boxes", None)
    observations = []
    if boxes is not None and len(boxes) > 0:
        for bbox, class_id, confidence in zip(boxes.xyxy, boxes.cls, boxes.conf):
            class_id = int(scalar(class_id))
            class_name = names[class_id]
            if class_name not in PLAYER_CLASSES | REFEREE_CLASSES | BALL_CLASSES:
                continue
            observations.append(DetectionObservation(
                frame=int(frame),
                bbox=[float(value) for value in to_list(bbox)],
                confidence=float(scalar(confidence)),
                class_id=class_id,
                class_name=class_name,
                role=role_for_class(class_name),
            ))
    return FrameDetections(
        frame=int(frame),
        observations=observations,
        image_shape=tuple(int(value) for value in result.orig_shape),
        native_result=result,
    )


def scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
