import json
from pathlib import Path

import cv2
import numpy as np

from ft.calibration.tvcalib_adapter import load_tvcalib_homography, load_tvcalib_homography_map


class PitchTransform:
    """Image-to-pitch homography with manual and automatic fallback modes."""

    def __init__(
        self,
        homography=None,
        homographies_by_frame=None,
        calibration_points=None,
        enabled=True,
        source="disabled",
        pitch_length=105.0,
        pitch_width=68.0,
        nearest_frame=True,
        max_frame_gap=None,
    ):
        self.homography = homography
        self.homographies_by_frame = {
            int(frame): np.asarray(h, dtype=np.float32)
            for frame, h in (homographies_by_frame or {}).items()
        }
        self.calibration_points = calibration_points or []
        self.enabled = bool(enabled and (homography is not None or self.homographies_by_frame))
        self.source = source
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.nearest_frame = bool(nearest_frame)
        self.max_frame_gap = None if max_frame_gap is None else int(max_frame_gap)
        self._sorted_frames = sorted(self.homographies_by_frame)
        self._selected_frame_counts = {}

    @classmethod
    def from_config(cls, config, frames):
        if not config.get("enabled", True):
            return cls(enabled=False, source="disabled:config")
        tvcalib_cfg = config.get("tvcalib") or {}
        if tvcalib_cfg.get("enabled", False):
            return cls.from_tvcalib(
                tvcalib_cfg,
                pitch_length=float(config.get("pitch_length", 105.0)),
                pitch_width=float(config.get("pitch_width", 68.0)),
            )
        path = config.get("path")
        if path:
            return cls.from_file(path)
        if config.get("auto", False) and frames:
            frame_index = max(0, min(int(config.get("auto_frame", 0)), len(frames) - 1))
            return cls.from_frame(
                frames[frame_index],
                pitch_length=float(config.get("pitch_length", 105.0)),
                pitch_width=float(config.get("pitch_width", 68.0)),
            )
        return cls(enabled=False, source="disabled:no_calibration")

    @classmethod
    def from_tvcalib(cls, config, pitch_length=105.0, pitch_width=68.0):
        path = config.get("path")
        if not path:
            raise ValueError("calibration.tvcalib.enabled is true but calibration.tvcalib.path is empty")

        common = {
            "coordinate_system": config.get("coordinate_system", "tvcalib_centered"),
            "invert": bool(config.get("invert", False)),
            "pitch_length": pitch_length,
            "pitch_width": pitch_width,
            "temporal_index": int(config.get("temporal_index", 0)),
        }
        per_frame = bool(config.get("per_frame", True))
        if per_frame:
            homographies = load_tvcalib_homography_map(
                path,
                frame_offset=int(config.get("frame_offset", 0)),
                **common,
            )
            return cls(
                homographies_by_frame=homographies,
                source=f"tvcalib:{path}",
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                nearest_frame=config.get("nearest_frame", True),
                max_frame_gap=config.get("max_frame_gap"),
            )

        h = load_tvcalib_homography(
            path,
            image_id=config.get("image_id"),
            frame=config.get("frame"),
            **common,
        )
        return cls(
            homography=h,
            source=f"tvcalib:{path}",
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        pitch_length = float(payload.get("pitch_length", 105.0))
        pitch_width = float(payload.get("pitch_width", 68.0))
        if "homography" in payload:
            h = np.asarray(payload["homography"], dtype=np.float32)
            if h.shape != (3, 3):
                raise ValueError("homography must be a 3x3 matrix")
            return cls(homography=h, source=str(path), pitch_length=pitch_length, pitch_width=pitch_width)
        if "point_correspondences" in payload:
            pixels = []
            pitch = []
            for item in payload["point_correspondences"]:
                pixels.append(item["pixel"])
                pitch.append(item["pitch"])
            return cls.from_points(pixels, pitch, source=str(path), pitch_length=pitch_length, pitch_width=pitch_width)
        if "pixel_vertices" in payload:
            pixels = payload["pixel_vertices"]
            pitch = payload.get(
                "target_vertices",
                [[0, pitch_width], [0, 0], [pitch_length, 0], [pitch_length, pitch_width]],
            )
            return cls.from_points(pixels, pitch, source=str(path), pitch_length=pitch_length, pitch_width=pitch_width)
        raise ValueError(f"Calibration file has no supported calibration fields: {path}")

    @classmethod
    def from_points(cls, pixel_points, pitch_points, source="manual", pitch_length=105.0, pitch_width=68.0):
        pixel_points = np.asarray(pixel_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        if len(pixel_points) < 4 or len(pitch_points) < 4:
            raise ValueError("At least four point correspondences are required")
        h, _ = cv2.findHomography(pixel_points, pitch_points, method=0)
        if h is None:
            return cls(enabled=False, source=f"disabled:{source}:fit_failed")
        points = [
            {"pixel": pixel.tolist(), "pitch": target.tolist()}
            for pixel, target in zip(pixel_points, pitch_points)
        ]
        return cls(
            homography=h.astype(np.float32),
            calibration_points=points,
            source=source,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    @classmethod
    def from_frame(cls, frame, pitch_length=105.0, pitch_width=68.0, min_area_ratio=0.12):
        quad = estimate_field_quad(frame, min_area_ratio=min_area_ratio)
        if quad is None:
            return cls(enabled=False, source="disabled:auto_field_quad_failed")
        pitch = np.asarray(
            [[0, pitch_width], [0, 0], [pitch_length, 0], [pitch_length, pitch_width]],
            dtype=np.float32,
        )
        return cls.from_points(
            quad,
            pitch,
            source="auto:field_quad",
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    def transform_point(self, point, frame_index=None):
        if not self.enabled:
            return None
        homography = self._homography_for_frame(frame_index)
        if homography is None:
            return None
        point = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, homography)
        return transformed.reshape(2).astype(float).tolist()

    def apply_tracks(self, tracks):
        for key in ("players", "referees", "ball"):
            for frame_num, frame_tracks in enumerate(tracks.get(key, [])):
                for track in frame_tracks.values():
                    position = track.get("position")
                    track["position_pitch"] = (
                        self.transform_point(position, frame_index=frame_num)
                        if position is not None
                        else None
                    )

    def diagnostics(self):
        return {
            "enabled": self.enabled,
            "source": self.source,
            "points": self.calibration_points,
            "mode": "per_frame" if self.homographies_by_frame else "static",
            "num_homographies": (
                len(self.homographies_by_frame)
                if self.homographies_by_frame
                else int(self.homography is not None)
            ),
            "homography_frames": self._frame_summary(),
            "nearest_frame": self.nearest_frame,
            "max_frame_gap": self.max_frame_gap,
            "selected_frame_counts": {
                str(frame): count
                for frame, count in sorted(self._selected_frame_counts.items())
            },
        }

    def _homography_for_frame(self, frame_index=None):
        if not self.homographies_by_frame:
            return self.homography
        if frame_index is None:
            return self.homographies_by_frame[self._sorted_frames[0]]
        frame_index = int(frame_index)
        selected = frame_index if frame_index in self.homographies_by_frame else None
        if selected is None and self.nearest_frame:
            selected = min(self._sorted_frames, key=lambda frame: abs(frame - frame_index))
            if self.max_frame_gap is not None and abs(selected - frame_index) > self.max_frame_gap:
                selected = None
        if selected is None:
            return None
        self._selected_frame_counts[selected] = self._selected_frame_counts.get(selected, 0) + 1
        return self.homographies_by_frame[selected]

    def _frame_summary(self):
        if not self._sorted_frames:
            return []
        if len(self._sorted_frames) <= 20:
            return self._sorted_frames
        return {
            "first": self._sorted_frames[:5],
            "last": self._sorted_frames[-5:],
            "min": self._sorted_frames[0],
            "max": self._sorted_frames[-1],
        }


def estimate_field_quad(frame, min_area_ratio=0.12):
    if frame is None or frame.size == 0:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 35, 35]), np.array([95, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    height, width = frame.shape[:2]
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) / float(height * width) < min_area_ratio:
        return None
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float32)
    return order_quad(quad)


def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]
    return np.asarray([bottom_left, top_left, top_right, bottom_right], dtype=np.float32)
