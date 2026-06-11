import os
from contextlib import nullcontext

import numpy as np
import pandas as pd

from ft.utils.geometry import bbox_center, bbox_foot
from ft.tracking.ball_kalman import BallKalmanTracker

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import supervision as sv
except ImportError:
    sv = None

try:
    import torch
except ImportError:
    torch = None


PLAYER_CLASSES = {"person", "player", "goalkeeper"}
REFEREE_CLASSES = {"referee"}
BALL_CLASSES = {"ball", "sports ball"}


class YoloByteTracker:
    """YOLO detector plus ByteTrack association.

    Output format is a plain dictionary so downstream modules stay independent
    from Ultralytics and Supervision internals. The tracker produces only weak
    roles from detector classes; later modules are responsible for semantic
    corrections such as referee and goalkeeper kit colours.
    """

    def __init__(
        self,
        model_path,
        detection_confidence=0.05,
        ball_confidence=0.002,
        ball_max_area_ratio=0.0015,
        ball_size_penalty=0.5,
        ball_temporal_consistency=False,
        ball_temporal_max_distance=120.0,
        ball_temporal_max_distance_cap=120.0,
        ball_temporal_distance_penalty=0.35,
        ball_temporal_reject_outliers=True,
        ball_min_acquisition_confidence=0.05,
        ball_low_confidence_max_distance=30.0,
        ball_temporal_min_confidence_after_miss=0.05,
        ball_temporal_miss_reset=12,
        ball_kalman_enabled=False,
        ball_kalman_max_lost_frames=8,
        ball_kalman_process_noise_scale=50.0,
        ball_kalman_measurement_noise_scale=5.0,
        ball_kalman_high_speed_threshold=30.0,
        ball_kalman_high_speed_area_multiplier=3.0,
        track_activation_threshold=0.10,
        lost_track_buffer=150,
        minimum_matching_threshold=0.95,
        frame_rate=25,
        minimum_consecutive_frames=2,
        progress_every=250,
        inference_mode=True,
        half_precision=False,
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
        self.ball_temporal_consistency = bool(ball_temporal_consistency)
        self.ball_temporal_max_distance = float(ball_temporal_max_distance)
        self.ball_temporal_max_distance_cap = float(ball_temporal_max_distance_cap)
        self.ball_temporal_distance_penalty = float(ball_temporal_distance_penalty)
        self.ball_temporal_reject_outliers = bool(ball_temporal_reject_outliers)
        self.ball_min_acquisition_confidence = float(ball_min_acquisition_confidence)
        self.ball_low_confidence_max_distance = float(ball_low_confidence_max_distance)
        self.ball_temporal_min_confidence_after_miss = float(ball_temporal_min_confidence_after_miss)
        self.ball_temporal_miss_reset = max(0, int(ball_temporal_miss_reset or 0))
        self.ball_kalman_enabled = bool(ball_kalman_enabled)
        self.ball_kalman_high_speed_threshold = float(ball_kalman_high_speed_threshold)
        self.ball_kalman_high_speed_area_multiplier = max(1.0, float(ball_kalman_high_speed_area_multiplier))
        self.ball_kalman = (
            BallKalmanTracker(
                max_lost_frames=ball_kalman_max_lost_frames,
                process_noise_scale=ball_kalman_process_noise_scale,
                measurement_noise_scale=ball_kalman_measurement_noise_scale,
            )
            if self.ball_kalman_enabled
            else None
        )
        self._last_ball_bbox_size = None
        self._last_ball_center = None
        self._last_ball_frame = None
        self.tracker_params = {
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
            "frame_rate": frame_rate,
            "minimum_consecutive_frames": minimum_consecutive_frames,
        }
        self.tracker = self._build_tracker(
            track_activation_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
            frame_rate,
            minimum_consecutive_frames,
        )
        self.progress_every = int(progress_every or 0)
        self.inference_mode = bool(inference_mode)
        self.half_precision = bool(half_precision)

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

    def reset_tracker(self):
        """Reset ByteTrack state while keeping the detector warm."""
        self.tracker = self._build_tracker(**self.tracker_params)

    def reset_ball_state(self):
        """Forget temporal ball context after shot boundaries or new runs."""
        self._last_ball_center = None
        self._last_ball_frame = None
        self._last_ball_bbox_size = None
        if self.ball_kalman is not None:
            self.ball_kalman.reset()

    def run(self, frames, batch_size=20, scene_cut_frames=None):
        """Track frames sequentially, optionally resetting at shot boundaries."""
        tracks = {"players": [], "referees": [], "ball": []}
        scene_cut_frames = {int(frame) for frame in (scene_cut_frames or []) if int(frame) > 0}
        current_id_offset = 0
        max_seen_track_id = 0
        scene_segment_id = 0
        self.reset_ball_state()
        context = torch.inference_mode() if torch is not None and self.inference_mode else nullcontext()
        with context:
            for start in range(0, len(frames), batch_size):
                # Detection is batched for GPU throughput, but ByteTrack still
                # receives frames in order so its temporal state remains valid.
                results = self.model.predict(
                    frames[start : start + batch_size],
                    conf=self.ball_confidence,
                    verbose=False,
                    half=self.half_precision,
                )
                for local_index, result in enumerate(results):
                    frame_num = start + local_index
                    scene_cut_reset = frame_num in scene_cut_frames
                    if scene_cut_reset:
                        self.reset_tracker()
                        self.reset_ball_state()
                        current_id_offset = max_seen_track_id
                        scene_segment_id += 1
                    frame_tracks = self._process_frame(
                        result,
                        frame_num=frame_num,
                        id_offset=current_id_offset,
                        scene_segment_id=scene_segment_id,
                        scene_cut_reset=scene_cut_reset,
                    )
                    for key in tracks:
                        tracks[key].append(frame_tracks[key])
                    max_seen_track_id = max(max_seen_track_id, max_frame_track_id(frame_tracks))
                    if self.progress_every > 0 and (frame_num + 1) % self.progress_every == 0:
                        print(
                            "FT bytetrack:"
                            f" processed_frames={frame_num + 1}/{len(frames)}",
                            flush=True,
                        )
        self.add_positions(tracks)
        tracks["ball"] = self.interpolate_ball(tracks["ball"])
        self.add_positions(tracks)
        return tracks

    def _process_frame(self, result, frame_num=None, id_offset=0, scene_segment_id=0, scene_cut_reset=False):
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

        # ByteTrack follows athletes/referees. Ball detections are too small and
        # intermittent for the same association logic, so they are selected
        # separately and interpolated after the pass.
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
            local_track_id = int(item[4])
            track_id = local_track_id + int(id_offset)
            class_name = names[class_id]
            payload = {
                "bbox": bbox,
                "role_detection": "goalkeeper" if class_name == "goalkeeper" else "player",
                "raw_track_id": local_track_id,
                "scene_segment_id": int(scene_segment_id),
            }
            if scene_cut_reset:
                payload["scene_cut_boundary"] = True
            if class_name in PLAYER_CLASSES:
                output["players"][track_id] = payload
            elif class_name in REFEREE_CLASSES:
                payload["role_detection"] = "referee"
                output["referees"][track_id] = payload

        ball_candidates = []
        frame_height, frame_width = result.orig_shape
        frame_area = float(frame_height * frame_width)
        for bbox, class_id, confidence in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            if names[int(class_id)] in BALL_CLASSES:
                ball_candidates.append((bbox.tolist(), float(confidence)))
        predicted_ball = self._predict_ball_kalman()
        dynamic_area_ratio = self.ball_max_area_ratio
        if (
            predicted_ball is not None
            and predicted_ball.get("speed_px_per_frame", 0.0) > self.ball_kalman_high_speed_threshold
        ):
            dynamic_area_ratio = self.ball_max_area_ratio * self.ball_kalman_high_speed_area_multiplier
        selected_ball = self.select_ball(
            ball_candidates,
            frame_area,
            frame_num=frame_num,
            max_area_ratio=dynamic_area_ratio,
            predicted_center=predicted_ball.get("center") if predicted_ball else None,
            predicted_gap=predicted_ball.get("kalman_lost_frames") if predicted_ball else None,
        )
        if selected_ball:
            selected_ball["scene_segment_id"] = int(scene_segment_id)
            if dynamic_area_ratio != self.ball_max_area_ratio:
                selected_ball["dynamic_area_ratio"] = dynamic_area_ratio
            if scene_cut_reset:
                selected_ball["scene_cut_boundary"] = True
            self._remember_ball(selected_ball, frame_num)
            output["ball"][1] = selected_ball
        elif predicted_ball is not None:
            predicted_ball["scene_segment_id"] = int(scene_segment_id)
            if scene_cut_reset:
                predicted_ball["scene_cut_boundary"] = True
            output["ball"][1] = predicted_ball
        return output

    def select_ball(
        self,
        ball_candidates,
        frame_area,
        frame_num=None,
        max_area_ratio=None,
        predicted_center=None,
        predicted_gap=None,
    ):
        best = None
        best_score = -1e9
        temporal_candidates_seen = False
        temporal_candidates_kept = False
        max_area_ratio = self.ball_max_area_ratio if max_area_ratio is None else float(max_area_ratio)
        for bbox, confidence in ball_candidates:
            if (
                self.ball_temporal_consistency
                and not self._has_recent_ball_context(frame_num)
                and confidence < self.ball_min_acquisition_confidence
            ):
                continue
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            area_ratio = (width * height) / frame_area if frame_area > 0 else 1.0
            if area_ratio > max_area_ratio:
                continue
            # Low ball confidence is tolerated, but large false positives are
            # strongly penalized because logos/boots can otherwise win.
            size_penalty = (
                area_ratio / max_area_ratio
                if max_area_ratio > 0
                else 0.0
            )
            score = confidence - self.ball_size_penalty * size_penalty
            base_score = score
            temporal_distance = None
            temporal_gap = None
            if predicted_center is not None:
                temporal_candidates_seen = True
                center = bbox_center(bbox)
                temporal_gap = max(1, int(predicted_gap or 1))
                temporal_distance = float(np.linalg.norm(np.asarray(center) - np.asarray(predicted_center)))
                max_distance = self._ball_temporal_max_distance(temporal_gap)
                if confidence < self.ball_min_acquisition_confidence:
                    max_distance = min(max_distance, max(1.0, self.ball_low_confidence_max_distance))
                distance_ratio = temporal_distance / max_distance
                if self.ball_temporal_reject_outliers and distance_ratio > 1.0:
                    continue
                if (
                    temporal_gap > 1
                    and confidence < self.ball_temporal_min_confidence_after_miss
                ):
                    continue
                temporal_candidates_kept = True
                score -= self.ball_temporal_distance_penalty * distance_ratio
            elif self._can_score_ball_temporally(frame_num):
                temporal_candidates_seen = True
                center = bbox_center(bbox)
                temporal_gap = max(1, int(frame_num) - int(self._last_ball_frame))
                temporal_distance = float(np.linalg.norm(np.asarray(center) - np.asarray(self._last_ball_center)))
                max_distance = self._ball_temporal_max_distance(temporal_gap)
                if confidence < self.ball_min_acquisition_confidence:
                    max_distance = min(max_distance, max(1.0, self.ball_low_confidence_max_distance))
                distance_ratio = temporal_distance / max_distance
                if self.ball_temporal_reject_outliers and distance_ratio > 1.0:
                    continue
                if (
                    temporal_gap > 1
                    and confidence < self.ball_temporal_min_confidence_after_miss
                ):
                    continue
                temporal_candidates_kept = True
                score -= self.ball_temporal_distance_penalty * distance_ratio
            if score > best_score:
                best_score = score
                best = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                    "score": score,
                    "base_score": base_score,
                    "temporal_distance": temporal_distance,
                    "temporal_gap": temporal_gap,
                }
        if best is None and temporal_candidates_seen and not temporal_candidates_kept:
            return self._handle_missing_temporal_ball(frame_num)
        if best is None and self._last_ball_frame is not None and frame_num is not None:
            self._handle_missing_temporal_ball(frame_num)
        return best

    def _handle_missing_temporal_ball(self, frame_num):
        if self._last_ball_frame is not None and frame_num is not None:
            if int(frame_num) - int(self._last_ball_frame) > self.ball_temporal_miss_reset:
                self.reset_ball_state()
        return None

    def _can_score_ball_temporally(self, frame_num):
        if not self.ball_temporal_consistency:
            return False
        if not self._has_recent_ball_context(frame_num):
            return False
        gap = int(frame_num) - int(self._last_ball_frame)
        return 0 < gap <= self.ball_temporal_miss_reset

    def _has_recent_ball_context(self, frame_num):
        if frame_num is None or self._last_ball_center is None or self._last_ball_frame is None:
            return False
        gap = int(frame_num) - int(self._last_ball_frame)
        return 0 < gap <= self.ball_temporal_miss_reset

    def _ball_temporal_max_distance(self, temporal_gap):
        scaled = self.ball_temporal_max_distance * max(1, int(temporal_gap))
        cap = self.ball_temporal_max_distance_cap
        if cap and cap > 0:
            scaled = min(scaled, cap)
        return max(1.0, scaled)

    def _remember_ball(self, selected_ball, frame_num):
        if frame_num is None:
            return
        bbox = selected_ball["bbox"]
        self._last_ball_center = bbox_center(bbox)
        self._last_ball_frame = int(frame_num)
        self._last_ball_bbox_size = [
            max(1.0, float(bbox[2]) - float(bbox[0])),
            max(1.0, float(bbox[3]) - float(bbox[1])),
        ]
        if self.ball_kalman is not None:
            selected_ball["kalman_update"] = True
            self.ball_kalman.update(self._last_ball_center)

    def _predict_ball_kalman(self):
        if self.ball_kalman is None or self._last_ball_bbox_size is None:
            return None
        previous = None if self.ball_kalman.x is None else self.ball_kalman.x[:2].astype(float)
        center = self.ball_kalman.predict()
        if center is None or not self.ball_kalman.is_valid():
            return None
        speed = 0.0
        if previous is not None:
            speed = float(np.linalg.norm(np.asarray(center, dtype=np.float32) - previous))
        width, height = self._last_ball_bbox_size
        bbox = [
            float(center[0] - width / 2.0),
            float(center[1] - height / 2.0),
            float(center[0] + width / 2.0),
            float(center[1] + height / 2.0),
        ]
        return {
            "bbox": bbox,
            "center": [float(center[0]), float(center[1])],
            "interpolated": True,
            "kalman_predicted": True,
            "kalman_lost_frames": int(self.ball_kalman.lost_frames),
            "speed_px_per_frame": speed,
        }

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
        detected = [frame.get(1, {}) for frame in ball_frames]
        values = [track.get("bbox", []) for track in detected]
        if not any(len(value) == 4 for value in values):
            return [{} for _ in values]
        valid = [len(value) == 4 for value in values]
        values = [
            value if len(value) == 4 else [np.nan, np.nan, np.nan, np.nan]
            for value in values
        ]
        df = pd.DataFrame(values, columns=["x1", "y1", "x2", "y2"])
        # Ball position is only a contextual cue for visualization/analysis;
        # interpolation should not feed back into player identity.
        df = df.interpolate().bfill().ffill()
        output = []
        for index, (_, row) in enumerate(df.iterrows()):
            if row.isna().any():
                output.append({})
                continue
            if valid[index]:
                track = dict(detected[index])
                track["bbox"] = row.tolist()
                track["interpolated"] = bool(track.get("interpolated", False))
            else:
                track = {"bbox": row.tolist(), "interpolated": True}
            output.append({1: track})
        return output


def max_frame_track_id(frame_tracks):
    max_id = 0
    for group in ("players", "referees"):
        if frame_tracks.get(group):
            max_id = max(max_id, max(int(track_id) for track_id in frame_tracks[group]))
    return max_id
