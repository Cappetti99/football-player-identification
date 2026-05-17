import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ft.features.visual import cosine_similarity, extract_from_frame
from ft.tracking.yolo_bytetrack import BALL_CLASSES, PLAYER_CLASSES, REFEREE_CLASSES
from ft.utils.geometry import bbox_center, bbox_foot

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YoloStrongSortTracker:
    """YOLO detector plus a lightweight StrongSORT-style association backend.

    The SoccerNet Game State baseline wires PRT/BPBreID embeddings into
    StrongSORT through TrackLab. TrackLab is not vendored in this project, so
    this backend keeps the same idea locally: associate detections with a
    motion gate, IoU, and an appearance embedding, while leaving ByteTrack
    available as the default tracker.
    """

    def __init__(
        self,
        model_path,
        detection_confidence=0.05,
        ball_confidence=0.002,
        ball_max_area_ratio=0.0015,
        ball_size_penalty=0.5,
        track_activation_threshold=0.10,
        lost_track_buffer=150,
        minimum_matching_threshold=0.95,
        frame_rate=25,
        minimum_consecutive_frames=2,
        max_age=None,
        min_hits=None,
        max_center_distance=180.0,
        max_cost=0.78,
        iou_weight=0.45,
        appearance_weight=0.35,
        center_weight=0.20,
        appearance_min_similarity=0.15,
        appearance_ema=0.85,
        class_gate=True,
        progress_every=250,
    ):
        if YOLO is None:
            raise ImportError("Missing ultralytics. Install FT requirements first.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)
        self.detection_confidence = float(detection_confidence)
        self.ball_confidence = float(ball_confidence)
        self.ball_max_area_ratio = float(ball_max_area_ratio)
        self.ball_size_penalty = float(ball_size_penalty)
        self.tracker = StrongSortTrackerCore(
            min_confidence=track_activation_threshold,
            max_age=max_age if max_age is not None else lost_track_buffer,
            min_hits=min_hits if min_hits is not None else minimum_consecutive_frames,
            max_center_distance=max_center_distance,
            max_cost=max_cost,
            iou_weight=iou_weight,
            appearance_weight=appearance_weight,
            center_weight=center_weight,
            appearance_min_similarity=appearance_min_similarity,
            appearance_ema=appearance_ema,
            class_gate=class_gate,
        )
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self.frame_rate = int(frame_rate)
        self.progress_every = int(progress_every or 0)

    def run(self, frames, batch_size=20):
        tracks = {"players": [], "referees": [], "ball": []}
        for start in range(0, len(frames), batch_size):
            results = self.model.predict(
                frames[start : start + batch_size],
                conf=self.ball_confidence,
                verbose=False,
            )
            for local_index, result in enumerate(results):
                frame_num = start + local_index
                frame = frames[frame_num]
                frame_tracks = self._process_frame(result, frame)
                for key in tracks:
                    tracks[key].append(frame_tracks[key])
                if self.progress_every > 0 and (frame_num + 1) % self.progress_every == 0:
                    print(
                        "FT strongsort:"
                        f" processed_frames={frame_num + 1}/{len(frames)}"
                        f" active_tracks={len(self.tracker.tracks)}"
                        f" next_id={self.tracker.next_id}",
                        flush=True,
                    )
        self.add_positions(tracks)
        tracks["ball"] = self.interpolate_ball(tracks["ball"])
        self.add_positions(tracks)
        return tracks

    def _process_frame(self, result, frame):
        names = {int(k): str(v).lower() for k, v in result.names.items()}
        detections, ball_candidates = self._extract_detections(result, frame, names)
        tracked = self.tracker.update(detections)
        output = {"players": {}, "referees": {}, "ball": {}}
        for track in tracked:
            if track.class_name in PLAYER_CLASSES:
                role = "goalkeeper" if track.class_name == "goalkeeper" else "player"
                output["players"][track.track_id] = {"bbox": track.bbox.tolist(), "role_detection": role}
            elif track.class_name in REFEREE_CLASSES:
                output["referees"][track.track_id] = {
                    "bbox": track.bbox.tolist(),
                    "role_detection": "referee",
                }

        frame_height, frame_width = result.orig_shape
        frame_area = float(frame_height * frame_width)
        selected_ball = self.select_ball(ball_candidates, frame_area)
        if selected_ball:
            output["ball"][1] = selected_ball
        return output

    def _extract_detections(self, result, frame, names):
        detections = []
        ball_candidates = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections, ball_candidates
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        for bbox, class_id, confidence in zip(xyxy, class_ids, confidences):
            class_name = names[int(class_id)]
            confidence = float(confidence)
            threshold = self.ball_confidence if class_name in BALL_CLASSES else self.detection_confidence
            if confidence < threshold:
                continue
            bbox = [float(v) for v in bbox.tolist()]
            if class_name in BALL_CLASSES:
                ball_candidates.append((bbox, confidence))
                continue
            if class_name not in PLAYER_CLASSES and class_name not in REFEREE_CLASSES:
                continue
            detections.append(
                Detection(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    confidence=confidence,
                    class_name=class_name,
                    embedding=extract_from_frame(frame, bbox),
                )
            )
        return detections, ball_candidates

    def select_ball(self, ball_candidates, frame_area):
        best = None
        best_score = -1e9
        for bbox, confidence in ball_candidates:
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            area_ratio = (width * height) / frame_area if frame_area > 0 else 1.0
            if area_ratio > self.ball_max_area_ratio:
                continue
            size_penalty = (
                area_ratio / self.ball_max_area_ratio
                if self.ball_max_area_ratio > 0
                else 0.0
            )
            score = confidence - self.ball_size_penalty * size_penalty
            if score > best_score:
                best_score = score
                best = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                    "score": score,
                }
        return best

    @staticmethod
    def add_positions(tracks):
        for frame_tracks in tracks.get("players", []):
            for track in frame_tracks.values():
                track["position"] = bbox_foot(track["bbox"])
        for frame_tracks in tracks.get("referees", []):
            for track in frame_tracks.values():
                track["position"] = bbox_foot(track["bbox"])
        for frame_tracks in tracks.get("ball", []):
            for track in frame_tracks.values():
                track["position"] = bbox_center(track["bbox"])

    @staticmethod
    def interpolate_ball(ball_frames):
        values = [frame.get(1, {}).get("bbox", []) for frame in ball_frames]
        if not any(len(value) == 4 for value in values):
            return [{} for _ in values]
        values = [
            value if len(value) == 4 else [np.nan, np.nan, np.nan, np.nan]
            for value in values
        ]
        df = pd.DataFrame(values, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill().ffill()
        return [
            ({1: {"bbox": row.tolist(), "interpolated": True}} if not row.isna().any() else {})
            for _, row in df.iterrows()
        ]


@dataclass
class Detection:
    bbox: np.ndarray
    confidence: float
    class_name: str
    embedding: Optional[list] = None


@dataclass
class StrongSortTrack:
    track_id: int
    bbox: np.ndarray
    confidence: float
    class_name: str
    embedding: Optional[list] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    age: int = 1
    hits: int = 1
    time_since_update: int = 0

    def predict(self):
        self.bbox = self.bbox + self.velocity
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, appearance_ema):
        previous = self.bbox.copy()
        self.bbox = detection.bbox.copy()
        self.velocity = self.bbox - previous
        self.confidence = detection.confidence
        self.class_name = detection.class_name
        self.embedding = update_embedding(self.embedding, detection.embedding, appearance_ema)
        self.hits += 1
        self.time_since_update = 0

    def confirmed(self, min_hits):
        return self.hits >= min_hits and self.time_since_update == 0


class StrongSortTrackerCore:
    def __init__(
        self,
        min_confidence=0.10,
        max_age=150,
        min_hits=2,
        max_center_distance=180.0,
        max_cost=0.78,
        iou_weight=0.45,
        appearance_weight=0.35,
        center_weight=0.20,
        appearance_min_similarity=0.15,
        appearance_ema=0.85,
        class_gate=True,
    ):
        self.min_confidence = float(min_confidence)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.max_center_distance = float(max_center_distance)
        self.max_cost = float(max_cost)
        self.iou_weight = float(iou_weight)
        self.appearance_weight = float(appearance_weight)
        self.center_weight = float(center_weight)
        self.appearance_min_similarity = float(appearance_min_similarity)
        self.appearance_ema = float(appearance_ema)
        self.class_gate = bool(class_gate)
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        detections = [detection for detection in detections if detection.confidence >= self.min_confidence]
        for track in self.tracks:
            track.predict()
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        matches, _, unmatched_detections = self.match(detections)
        for track_index, detection_index in matches:
            self.tracks[track_index].update(detections[detection_index], self.appearance_ema)
        for detection_index in unmatched_detections:
            self.start_track(detections[detection_index])

        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]
        return [track for track in self.tracks if track.confirmed(self.min_hits)]

    def start_track(self, detection):
        self.tracks.append(
            StrongSortTrack(
                track_id=self.next_id,
                bbox=detection.bbox.copy(),
                confidence=detection.confidence,
                class_name=detection.class_name,
                embedding=detection.embedding,
            )
        )
        self.next_id += 1

    def match(self, detections):
        if not self.tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(self.tracks))), []

        cost_matrix = np.full((len(self.tracks), len(detections)), 1e6, dtype=np.float32)
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                cost_matrix[track_index, detection_index] = self.association_cost(track, detection)

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matches = []
        matched_tracks = set()
        matched_detections = set()
        for track_index, detection_index in zip(row_indices, col_indices):
            cost = float(cost_matrix[track_index, detection_index])
            if cost > self.max_cost:
                continue
            matches.append((int(track_index), int(detection_index)))
            matched_tracks.add(int(track_index))
            matched_detections.add(int(detection_index))

        unmatched_tracks = [
            index for index in range(len(self.tracks)) if index not in matched_tracks
        ]
        unmatched_detections = [
            index for index in range(len(detections)) if index not in matched_detections
        ]
        return matches, unmatched_tracks, unmatched_detections

    def association_cost(self, track, detection):
        if self.class_gate and not compatible_classes(track.class_name, detection.class_name):
            return 1e6
        iou = bbox_iou(track.bbox, detection.bbox)
        center = center_distance(track.bbox, detection.bbox)
        center_cost = min(1.0, center / max(1.0, self.max_center_distance))
        if center > self.max_center_distance and iou <= 0.01:
            return 1e6

        similarity = cosine_similarity(track.embedding, detection.embedding)
        if similarity is None:
            appearance_cost = 0.5
        else:
            if similarity < self.appearance_min_similarity and iou <= 0.05:
                return 1e6
            appearance_cost = 1.0 - max(0.0, min(1.0, similarity))

        return (
            self.iou_weight * (1.0 - iou)
            + self.appearance_weight * appearance_cost
            + self.center_weight * center_cost
        )


def compatible_classes(left, right):
    if left == right:
        return True
    return left in PLAYER_CLASSES and right in PLAYER_CLASSES


def bbox_iou(left, right):
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def center_distance(left, right):
    lx = (float(left[0]) + float(left[2])) * 0.5
    ly = (float(left[1]) + float(left[3])) * 0.5
    rx = (float(right[0]) + float(right[2])) * 0.5
    ry = (float(right[1]) + float(right[3])) * 0.5
    return float(np.hypot(lx - rx, ly - ry))


def update_embedding(current, incoming, alpha):
    if incoming is None:
        return current
    if current is None:
        return incoming
    current = np.asarray(current, dtype=np.float32)
    incoming = np.asarray(incoming, dtype=np.float32)
    if current.shape != incoming.shape:
        return incoming.astype(float).tolist()
    merged = float(alpha) * current + (1.0 - float(alpha)) * incoming
    norm = np.linalg.norm(merged)
    if norm <= 0:
        return incoming.astype(float).tolist()
    return (merged / norm).astype(float).tolist()
