import os

import numpy as np
import pandas as pd

from ft.utils.geometry import bbox_center, bbox_foot

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import supervision as sv
except ImportError:
    sv = None


PLAYER_CLASSES = {"person", "player", "goalkeeper"}
REFEREE_CLASSES = {"referee"}
BALL_CLASSES = {"ball", "sports ball"}


class YoloByteTracker:
    """YOLO detector plus ByteTrack association.

    Output format is a plain dictionary so downstream modules stay independent
    from Ultralytics and Supervision internals.
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
    ):
        if YOLO is None:
            raise ImportError("Missing ultralytics. Install FT requirements first.")
        if sv is None:
            raise ImportError("Missing supervision. Install FT requirements first.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)
        self.detection_confidence = float(detection_confidence)
        self.ball_confidence = float(ball_confidence)
        self.ball_max_area_ratio = float(ball_max_area_ratio)
        self.ball_size_penalty = float(ball_size_penalty)
        self.tracker = self._build_tracker(
            track_activation_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
            frame_rate,
            minimum_consecutive_frames,
        )

    @staticmethod
    def _build_tracker(
        track_activation_threshold,
        lost_track_buffer,
        minimum_matching_threshold,
        frame_rate,
        minimum_consecutive_frames,
    ):
        kwargs = {
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
            "frame_rate": frame_rate,
        }
        try:
            return sv.ByteTrack(
                **kwargs,
                minimum_consecutive_frames=minimum_consecutive_frames,
            )
        except TypeError:
            return sv.ByteTrack(**kwargs)

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
                frame_tracks = self._process_frame(result)
                for key in tracks:
                    tracks[key].append(frame_tracks[key])
        self.add_positions(tracks)
        tracks["ball"] = self.interpolate_ball(tracks["ball"])
        self.add_positions(tracks)
        return tracks

    def _process_frame(self, result):
        names = {int(k): str(v).lower() for k, v in result.names.items()}
        detections = sv.Detections.from_ultralytics(result)
        if len(detections) > 0:
            keep = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                class_name = names[int(class_id)]
                threshold = (
                    self.ball_confidence
                    if class_name in BALL_CLASSES
                    else self.detection_confidence
                )
                keep.append(float(confidence) >= threshold)
            detections = detections[np.asarray(keep, dtype=bool)]

        # ByteTrack follows athletes/referees. Ball is selected separately.
        tracked_input = detections
        if len(tracked_input) > 0:
            keep_track = [
                names[int(class_id)] not in BALL_CLASSES
                for class_id in tracked_input.class_id
            ]
            tracked_input = tracked_input[np.asarray(keep_track, dtype=bool)]

        tracked = self.tracker.update_with_detections(tracked_input)
        output = {"players": {}, "referees": {}, "ball": {}}
        for item in tracked:
            bbox = item[0].tolist()
            class_id = int(item[3])
            track_id = int(item[4])
            class_name = names[class_id]
            if class_name in PLAYER_CLASSES:
                role = "goalkeeper" if class_name == "goalkeeper" else "player"
                output["players"][track_id] = {"bbox": bbox, "role_detection": role}
            elif class_name in REFEREE_CLASSES:
                output["referees"][track_id] = {"bbox": bbox, "role_detection": "referee"}

        ball_candidates = []
        frame_height, frame_width = result.orig_shape
        frame_area = float(frame_height * frame_width)
        for bbox, class_id, confidence in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            if names[int(class_id)] in BALL_CLASSES:
                ball_candidates.append((bbox.tolist(), float(confidence)))
        selected_ball = self.select_ball(ball_candidates, frame_area)
        if selected_ball:
            output["ball"][1] = selected_ball
        return output

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

