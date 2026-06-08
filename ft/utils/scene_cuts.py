from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass(frozen=True)
class SceneCut:
    """One detected video discontinuity between two adjacent frames."""

    frame: int
    previous_frame: int
    score: float
    correlation: float
    cut_type: str

    def as_dict(self):
        return {
            "frame": int(self.frame),
            "previous_frame": int(self.previous_frame),
            "score": float(self.score),
            "correlation": float(self.correlation),
            "type": self.cut_type,
        }


def detect_scene_cuts(
    frames,
    enabled=False,
    threshold=0.65,
    min_gap=12,
    crop_top_fraction=0.12,
    crop_bottom_fraction=0.12,
    crop_left_fraction=0.04,
    crop_right_fraction=0.04,
    resize_width=320,
    h_bins=32,
    s_bins=32,
    hard_cut_threshold=None,
    max_cuts=None,
    top_scores=25,
):
    """Detect broadcast shot boundaries from HSV histogram discontinuities.

    The detector compares consecutive frames after cropping out broadcast
    graphics near the borders. It is intentionally cheap because it runs before
    YOLO/OCR and should be usable on full videos.
    """
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "cuts": [],
            "cut_frames": [],
            "segments": [],
        }
    frame_count = len(frames or [])
    if frame_count < 2:
        return {
            "enabled": True,
            "status": "too_few_frames",
            "frame_count": frame_count,
            "cuts": [],
            "cut_frames": [],
            "segments": scene_segments(frame_count, []),
        }

    threshold = float(threshold)
    hard_cut_threshold = float(hard_cut_threshold if hard_cut_threshold is not None else threshold * 1.35)
    min_gap = max(1, int(min_gap or 1))
    max_cuts = int(max_cuts) if max_cuts is not None else None
    prev_hist = hsv_histogram(
        frames[0],
        crop_top_fraction=crop_top_fraction,
        crop_bottom_fraction=crop_bottom_fraction,
        crop_left_fraction=crop_left_fraction,
        crop_right_fraction=crop_right_fraction,
        resize_width=resize_width,
        h_bins=h_bins,
        s_bins=s_bins,
    )
    cuts = []
    score_rows = []
    last_cut_frame = -min_gap
    for frame_num in range(1, frame_count):
        hist = hsv_histogram(
            frames[frame_num],
            crop_top_fraction=crop_top_fraction,
            crop_bottom_fraction=crop_bottom_fraction,
            crop_left_fraction=crop_left_fraction,
            crop_right_fraction=crop_right_fraction,
            resize_width=resize_width,
            h_bins=h_bins,
            s_bins=s_bins,
        )
        score = chi_square_distance(prev_hist, hist)
        corr = histogram_correlation(prev_hist, hist)
        score_rows.append(
            {
                "frame": int(frame_num),
                "previous_frame": int(frame_num - 1),
                "score": float(score),
                "correlation": float(corr),
            }
        )
        if score >= threshold and frame_num - last_cut_frame >= min_gap:
            cut_type = "hard_cut" if score >= hard_cut_threshold else "discontinuity"
            cuts.append(SceneCut(frame_num, frame_num - 1, score, corr, cut_type))
            last_cut_frame = frame_num
            if max_cuts is not None and len(cuts) >= max_cuts:
                prev_hist = hist
                break
        prev_hist = hist

    cut_frames = [cut.frame for cut in cuts]
    return {
        "enabled": True,
        "status": "ok",
        "method": "hsv_histogram_chi_square",
        "frame_count": int(frame_count),
        "threshold": float(threshold),
        "hard_cut_threshold": float(hard_cut_threshold),
        "min_gap": int(min_gap),
        "crop": {
            "top_fraction": float(crop_top_fraction),
            "bottom_fraction": float(crop_bottom_fraction),
            "left_fraction": float(crop_left_fraction),
            "right_fraction": float(crop_right_fraction),
        },
        "histogram": {
            "h_bins": int(h_bins),
            "s_bins": int(s_bins),
            "resize_width": int(resize_width or 0),
        },
        "cuts": [cut.as_dict() for cut in cuts],
        "cut_frames": [int(frame) for frame in cut_frames],
        "segments": scene_segments(frame_count, cut_frames),
        "top_scores": sorted(score_rows, key=lambda row: row["score"], reverse=True)[: int(top_scores or 0)],
    }


def hsv_histogram(
    frame,
    crop_top_fraction=0.12,
    crop_bottom_fraction=0.12,
    crop_left_fraction=0.04,
    crop_right_fraction=0.04,
    resize_width=320,
    h_bins=32,
    s_bins=32,
):
    """Return a normalized HSV H/S histogram for one frame."""
    crop = central_crop(
        frame,
        top_fraction=crop_top_fraction,
        bottom_fraction=crop_bottom_fraction,
        left_fraction=crop_left_fraction,
        right_fraction=crop_right_fraction,
    )
    if cv2 is None:
        return rgb_histogram(crop, bins=max(int(h_bins), int(s_bins)))
    if resize_width and crop.shape[1] > int(resize_width):
        scale = float(resize_width) / float(crop.shape[1])
        crop = cv2.resize(crop, (int(resize_width), max(1, int(crop.shape[0] * scale))))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [int(h_bins), int(s_bins)],
        [0, 180, 0, 256],
    )
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    if total <= 0.0:
        return hist
    return hist / total


def rgb_histogram(frame, bins=32):
    """Fallback normalized colour histogram used when OpenCV is unavailable."""
    bins = max(2, int(bins))
    hist, _edges = np.histogramdd(
        frame.reshape(-1, frame.shape[-1]).astype(np.float32),
        bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    if total <= 0.0:
        return hist
    return hist / total


def central_crop(
    frame,
    top_fraction=0.12,
    bottom_fraction=0.12,
    left_fraction=0.04,
    right_fraction=0.04,
):
    """Crop out borders where broadcast overlays often create false cuts."""
    height, width = frame.shape[:2]
    top = int(max(0.0, min(0.45, float(top_fraction))) * height)
    bottom = int(height - max(0.0, min(0.45, float(bottom_fraction))) * height)
    left = int(max(0.0, min(0.45, float(left_fraction))) * width)
    right = int(width - max(0.0, min(0.45, float(right_fraction))) * width)
    if bottom <= top or right <= left:
        return frame
    return frame[top:bottom, left:right]


def chi_square_distance(a, b, eps=1e-8):
    """Return the symmetric chi-square distance between two histograms."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(0.5 * np.sum(((a - b) ** 2) / (a + b + float(eps))))


def histogram_correlation(a, b):
    """Return Pearson correlation for diagnostic ranking only."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    if float(a.std()) <= 1e-8 or float(b.std()) <= 1e-8:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def scene_segments(frame_count, cut_frames):
    """Convert cut frames into inclusive segment ranges."""
    if frame_count <= 0:
        return []
    cuts = sorted({int(frame) for frame in cut_frames or [] if 0 < int(frame) < int(frame_count)})
    starts = [0] + cuts
    ends = [frame - 1 for frame in cuts] + [int(frame_count) - 1]
    return [
        {
            "segment_index": int(index),
            "start_frame": int(start),
            "end_frame": int(end),
            "num_frames": int(end - start + 1),
        }
        for index, (start, end) in enumerate(zip(starts, ends))
        if end >= start
    ]


def annotate_tracks_with_scene_segments(tracks, scene_cut_frames, scene_cut_lookup=None):
    """Attach segment metadata to every tracked row for downstream audits."""
    scene_cut_lookup = scene_cut_lookup or {}
    cut_frames = sorted({int(frame) for frame in scene_cut_frames or [] if int(frame) > 0})
    cut_set = set(cut_frames)
    segment_index = 0
    for frame_num in range(max_track_frames(tracks)):
        if frame_num in cut_set:
            segment_index += 1
        for group in ("players", "referees", "ball"):
            frame_groups = tracks.get(group, [])
            if frame_num >= len(frame_groups):
                continue
            for track in frame_groups[frame_num].values():
                track["scene_segment_id"] = int(segment_index)
                if frame_num in cut_set:
                    track["scene_cut_boundary"] = True
                    cut = scene_cut_lookup.get(frame_num, {})
                    if cut:
                        track["scene_cut_score"] = cut.get("score")
                        track["scene_cut_type"] = cut.get("type")


def max_track_frames(tracks):
    return max((len(tracks.get(group, [])) for group in ("players", "referees", "ball")), default=0)


def scene_cut_rows(diagnostics):
    """Flatten detected cuts for CSV export."""
    return [
        {
            "frame": row.get("frame"),
            "previous_frame": row.get("previous_frame"),
            "score": row.get("score"),
            "correlation": row.get("correlation"),
            "type": row.get("type"),
        }
        for row in diagnostics.get("cuts", [])
    ]
