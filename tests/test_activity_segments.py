from ft.utils.activity_segments import (
    annotate_tracks_with_activity_segments,
    detect_activity_segments,
    rolling_median,
)


def test_persistent_count_change_creates_soft_segment_without_hard_reset():
    tracks = tracks_with_player_counts([2] * 6 + [5] * 6)

    diagnostics = detect_activity_segments(
        tracks,
        enabled=True,
        smoothing_window=1,
        count_change_threshold=2,
        persistence_frames=2,
        persistence_ratio=1.0,
        min_segment_frames=3,
        include_referees=False,
    )

    assert diagnostics["boundary_frames"] == [6]
    assert diagnostics["boundaries"][0]["type"] == "soft"
    assert diagnostics["boundaries"][0]["reason"] == "persistent_count_change"
    assert [row["num_frames"] for row in diagnostics["segments"]] == [6, 6]


def test_transient_count_change_does_not_create_segment():
    tracks = tracks_with_player_counts([3] * 5 + [7] + [3] * 6)

    diagnostics = detect_activity_segments(
        tracks,
        enabled=True,
        smoothing_window=1,
        count_change_threshold=2,
        persistence_frames=3,
        persistence_ratio=1.0,
        min_segment_frames=3,
        include_referees=False,
    )

    assert diagnostics["soft_boundary_count"] == 0
    assert len(diagnostics["segments"]) == 1


def test_scene_cut_is_hard_boundary_and_activity_annotation_is_exportable():
    tracks = tracks_with_player_counts([2] * 6 + [5] * 6)

    diagnostics = detect_activity_segments(
        tracks,
        enabled=True,
        hard_boundary_frames=[6],
        smoothing_window=1,
        count_change_threshold=2,
        persistence_frames=2,
        min_segment_frames=3,
        include_referees=False,
    )
    annotate_tracks_with_activity_segments(tracks, diagnostics)

    assert diagnostics["hard_boundary_count"] == 1
    assert diagnostics["soft_boundary_count"] == 0
    assert tracks["players"][5][1]["activity_segment_id"] == 0
    assert tracks["players"][6][1]["activity_segment_id"] == 1
    assert tracks["players"][6][1]["activity_boundary"] is True
    assert tracks["players"][6][1]["activity_boundary_type"] == "hard"


def test_weak_scene_discontinuity_segments_without_becoming_hard_boundary():
    tracks = tracks_with_player_counts([3] * 12)

    diagnostics = detect_activity_segments(
        tracks,
        enabled=True,
        soft_boundary_frames=[6],
        smoothing_window=1,
        count_change_threshold=2,
        persistence_frames=2,
        min_segment_frames=3,
        include_referees=False,
    )

    assert diagnostics["hard_boundary_count"] == 0
    assert diagnostics["soft_boundary_count"] == 1
    assert diagnostics["boundaries"][0]["reason"] == "scene_discontinuity"


def test_ball_visibility_excludes_interpolated_positions():
    tracks = tracks_with_player_counts([3] * 6)
    tracks["ball"] = [
        {1: {"bbox": [0, 0, 2, 2], "interpolated": index != 2}}
        for index in range(6)
    ]

    diagnostics = detect_activity_segments(
        tracks,
        enabled=True,
        smoothing_window=1,
        min_segment_frames=3,
    )

    segment = diagnostics["segments"][0]
    assert segment["ball_available_fraction"] == 1.0
    assert abs(segment["ball_visible_fraction"] - 1 / 6) < 1e-9


def test_rolling_median_uses_odd_window():
    assert rolling_median([1, 10, 1], 2) == [5.5, 1.0, 5.5]


def tracks_with_player_counts(counts):
    return {
        "players": [
            {
                track_id: {"bbox": [0, 0, 10, 10]}
                for track_id in range(1, count + 1)
            }
            for count in counts
        ],
        "referees": [{} for _ in counts],
        "ball": [{} for _ in counts],
    }
