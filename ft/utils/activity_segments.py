from __future__ import annotations

from statistics import median

from ft.utils.scene_cuts import max_track_frames, scene_segments


def detect_activity_segments(
    tracks,
    *,
    enabled=False,
    hard_boundary_frames=None,
    soft_boundary_frames=None,
    smoothing_window=15,
    count_change_threshold=3,
    persistence_frames=12,
    persistence_ratio=0.75,
    min_segment_frames=30,
    include_referees=True,
):
    """Split a tracked shot on persistent changes in visible athlete count.

    Scene cuts are hard boundaries. Count changes are deliberately soft: they
    create analysis units but never imply that tracker state should be reset.
    """
    frame_count = max_track_frames(tracks)
    if not enabled:
        return _disabled(frame_count)
    if frame_count <= 0:
        return {
            "enabled": True,
            "status": "too_few_frames",
            "frame_count": 0,
            "boundary_frames": [],
            "boundaries": [],
            "segments": [],
            "frame_stats": [],
        }

    smoothing_window = _positive_odd(smoothing_window)
    count_change_threshold = max(1, int(count_change_threshold))
    persistence_frames = max(1, int(persistence_frames))
    persistence_ratio = min(1.0, max(0.5, float(persistence_ratio)))
    min_segment_frames = max(1, int(min_segment_frames))
    hard_frames = sorted({
        int(frame)
        for frame in (hard_boundary_frames or [])
        if 0 < int(frame) < frame_count
    })
    soft_scene_frames = sorted({
        int(frame)
        for frame in (soft_boundary_frames or [])
        if 0 < int(frame) < frame_count and int(frame) not in hard_frames
    })

    player_counts = _group_counts(tracks, "players", frame_count)
    referee_counts = _group_counts(tracks, "referees", frame_count)
    ball_available = [
        bool(frame < len(tracks.get("ball", [])) and tracks["ball"][frame])
        for frame in range(frame_count)
    ]
    ball_detected = [
        any(
            not bool(track.get("interpolated", False))
            for track in (
                tracks.get("ball", [])[frame].values()
                if frame < len(tracks.get("ball", []))
                else []
            )
        )
        for frame in range(frame_count)
    ]
    athlete_counts = [
        player_counts[frame] + (referee_counts[frame] if include_referees else 0)
        for frame in range(frame_count)
    ]
    smoothed_counts = rolling_median(athlete_counts, smoothing_window)

    boundaries = [
        {
            "frame": frame,
            "type": "hard",
            "reason": "scene_cut",
            "previous_count_level": None,
            "new_count_level": None,
            "count_delta": None,
        }
        for frame in hard_frames
    ]
    boundaries.extend({
        "frame": frame,
        "type": "soft",
        "reason": "scene_discontinuity",
        "previous_count_level": None,
        "new_count_level": None,
        "count_delta": None,
    } for frame in soft_scene_frames)
    shot_ranges = scene_segments(frame_count, hard_frames + soft_scene_frames)
    for shot in shot_ranges:
        boundaries.extend(
            _persistent_count_boundaries(
                smoothed_counts,
                start=shot["start_frame"],
                end=shot["end_frame"],
                threshold=count_change_threshold,
                persistence_frames=persistence_frames,
                persistence_ratio=persistence_ratio,
                min_segment_frames=min_segment_frames,
            )
        )

    boundaries.sort(key=lambda row: int(row["frame"]))
    boundary_by_frame = {int(row["frame"]): row for row in boundaries}
    boundary_frames = sorted(boundary_by_frame)
    segments = []
    for segment in scene_segments(frame_count, boundary_frames):
        start = segment["start_frame"]
        end = segment["end_frame"]
        counts = athlete_counts[start : end + 1]
        player_slice = player_counts[start : end + 1]
        referee_slice = referee_counts[start : end + 1]
        ball_available_slice = ball_available[start : end + 1]
        ball_detected_slice = ball_detected[start : end + 1]
        boundary = boundary_by_frame.get(start)
        segments.append({
            **segment,
            "boundary_type": boundary.get("type") if boundary else "video_start",
            "boundary_reason": boundary.get("reason") if boundary else "video_start",
            "count_level": float(median(counts)) if counts else 0.0,
            "mean_athlete_count": _mean(counts),
            "min_athlete_count": min(counts) if counts else 0,
            "max_athlete_count": max(counts) if counts else 0,
            "mean_player_count": _mean(player_slice),
            "mean_referee_count": _mean(referee_slice),
            "ball_visible_fraction": _mean([int(value) for value in ball_detected_slice]),
            "ball_available_fraction": _mean([int(value) for value in ball_available_slice]),
        })

    frame_stats = []
    segment_index = 0
    for frame in range(frame_count):
        if frame in boundary_by_frame:
            segment_index += 1
        boundary = boundary_by_frame.get(frame)
        frame_stats.append({
            "frame": frame,
            "activity_segment_id": segment_index,
            "player_count": player_counts[frame],
            "referee_count": referee_counts[frame],
            "athlete_count": athlete_counts[frame],
            "smoothed_athlete_count": smoothed_counts[frame],
            "ball_visible": ball_detected[frame],
            "ball_available": ball_available[frame],
            "activity_boundary": boundary is not None,
            "activity_boundary_type": boundary.get("type") if boundary else None,
            "activity_boundary_reason": boundary.get("reason") if boundary else None,
        })

    return {
        "enabled": True,
        "status": "ok",
        "frame_count": frame_count,
        "method": "persistent_tracked_athlete_count",
        "config": {
            "smoothing_window": smoothing_window,
            "count_change_threshold": count_change_threshold,
            "persistence_frames": persistence_frames,
            "persistence_ratio": persistence_ratio,
            "min_segment_frames": min_segment_frames,
            "include_referees": bool(include_referees),
        },
        "hard_boundary_count": sum(row["type"] == "hard" for row in boundaries),
        "soft_boundary_count": sum(row["type"] == "soft" for row in boundaries),
        "boundary_frames": boundary_frames,
        "boundaries": boundaries,
        "segments": segments,
        "frame_stats": frame_stats,
    }


def annotate_tracks_with_activity_segments(tracks, diagnostics):
    """Copy activity segment metadata onto tracked objects for later exports."""
    stats = {
        int(row["frame"]): row
        for row in (diagnostics or {}).get("frame_stats", [])
    }
    for frame in range(max_track_frames(tracks)):
        row = stats.get(frame)
        if row is None:
            continue
        for group in ("players", "referees", "ball"):
            frame_groups = tracks.get(group, [])
            if frame >= len(frame_groups):
                continue
            for track in frame_groups[frame].values():
                track["activity_segment_id"] = int(row["activity_segment_id"])
                track["frame_athlete_count"] = int(row["athlete_count"])
                track["smoothed_athlete_count"] = float(row["smoothed_athlete_count"])
                if row["activity_boundary"]:
                    track["activity_boundary"] = True
                    track["activity_boundary_type"] = row["activity_boundary_type"]
                    track["activity_boundary_reason"] = row["activity_boundary_reason"]


def activity_segment_rows(diagnostics):
    return list((diagnostics or {}).get("segments", []))


def rolling_median(values, window):
    window = _positive_odd(window)
    radius = window // 2
    output = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        output.append(float(median(values[start:end])))
    return output


def _persistent_count_boundaries(
    values,
    *,
    start,
    end,
    threshold,
    persistence_frames,
    persistence_ratio,
    min_segment_frames,
):
    if end - start + 1 < min_segment_frames * 2:
        return []
    initial_end = min(end + 1, start + persistence_frames)
    level = float(median(values[start:initial_end]))
    last_boundary = start
    boundaries = []
    index = start + min_segment_frames
    while index <= end - min_segment_frames + 1:
        window_end = min(end + 1, index + persistence_frames)
        window = values[index:window_end]
        if len(window) < persistence_frames:
            break
        new_level = float(median(window))
        direction = 1.0 if new_level > level else -1.0
        persistent = sum(
            (float(value) - level) * direction >= threshold
            for value in window
        )
        if (
            abs(new_level - level) >= threshold
            and persistent / len(window) >= persistence_ratio
            and index - last_boundary >= min_segment_frames
            and end - index + 1 >= min_segment_frames
        ):
            boundaries.append({
                "frame": int(index),
                "type": "soft",
                "reason": "persistent_count_change",
                "previous_count_level": level,
                "new_count_level": new_level,
                "count_delta": new_level - level,
            })
            last_boundary = index
            level = new_level
            index += min_segment_frames
            continue
        index += 1
    return boundaries


def _group_counts(tracks, group, frame_count):
    frames = tracks.get(group, [])
    return [
        len(frames[frame]) if frame < len(frames) else 0
        for frame in range(frame_count)
    ]


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _positive_odd(value):
    value = max(1, int(value or 1))
    return value if value % 2 else value + 1


def _disabled(frame_count):
    return {
        "enabled": False,
        "status": "disabled",
        "frame_count": int(frame_count),
        "boundary_frames": [],
        "boundaries": [],
        "segments": [],
        "frame_stats": [],
    }
