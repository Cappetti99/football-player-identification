from contextlib import contextmanager
import sys
from unittest.mock import patch

# The phase-contract tests mock all image operations. Allow them to run in a
# lightweight development environment where optional OpenCV is not installed;
# production and integration tests still import the real dependency.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    from unittest.mock import MagicMock

    sys.modules["cv2"] = MagicMock()

from ft.pipeline import (
    TemporalAnalysis,
    analyze_temporal_structure,
    render_output_phase,
    tracking_phase,
)


class FakeDiagnostics:
    def __init__(self):
        self.stages = []

    @contextmanager
    def stage(self, name):
        self.stages.append(name)
        yield


def test_temporal_phase_separates_hard_and_soft_cuts():
    diagnostics = FakeDiagnostics()
    detected = {
        "enabled": True,
        "status": "ok",
        "cut_frames": [10, 20],
        "cuts": [
            {"frame": 10, "type": "discontinuity"},
            {"frame": 20, "type": "hard_cut"},
        ],
        "segments": [],
    }
    with patch("ft.pipeline.detect_scene_cuts", return_value=detected):
        result = analyze_temporal_structure(
            {
                "scene_cuts": {
                    "enabled": True,
                    "tracking_reset_enabled": True,
                    "tracking_reset_hard_only": True,
                }
            },
            frames=[object()],
            run_diagnostics=diagnostics,
        )

    assert result.scene_cut_frames == [10, 20]
    assert result.tracking_reset_frames == [20]
    assert result.hard_scene_cut_frames == [20]
    assert result.soft_scene_cut_frames == [10]
    assert diagnostics.stages == ["scene_cuts"]


def test_tracking_phase_forwards_boundaries_and_annotates_activity():
    diagnostics = FakeDiagnostics()
    temporal = TemporalAnalysis(
        diagnostics={"cuts": [{"frame": 2, "type": "hard_cut"}]},
        scene_cut_frames=[2],
        tracking_reset_frames=[2],
        hard_scene_cut_frames=[2],
        soft_scene_cut_frames=[],
    )
    tracks = {"players": [{}, {}, {}], "referees": [{}, {}, {}], "ball": [{}, {}, {}]}
    calls = {}

    def fake_run(tracker, frames, scene_cut_frames, reset_enabled):
        calls["tracking"] = (tracker, frames, scene_cut_frames, reset_enabled)
        return tracks

    def fake_activity(value, **kwargs):
        return {
            "segments": [{"segment_index": 0}],
            "hard_boundary_count": 1,
            "soft_boundary_count": 0,
        }

    def fake_annotation(value, activity):
        calls["activity"] = (value, activity)

    with (
        patch("ft.pipeline.build_tracker", return_value="tracker"),
        patch("ft.pipeline.run_tracker_with_scene_cuts", side_effect=fake_run),
        patch("ft.pipeline.detect_activity_segments", side_effect=fake_activity),
        patch(
            "ft.pipeline.annotate_tracks_with_activity_segments",
            side_effect=fake_annotation,
        ),
    ):
        result_tracks, activity = tracking_phase(
            {
                "scene_cuts": {"tracking_reset_enabled": True},
                "activity_segmentation": {"enabled": True},
            },
            frames=["f0", "f1", "f2"],
            model_path="model.pt",
            temporal=temporal,
            run_diagnostics=diagnostics,
        )

    assert result_tracks is tracks
    assert calls["tracking"] == (
        "tracker",
        ["f0", "f1", "f2"],
        [2],
        True,
    )
    assert calls["activity"][0] is tracks
    assert activity["hard_boundary_count"] == 1
    assert diagnostics.stages == ["tracking", "activity_segmentation"]


def test_render_phase_preserves_overlay_and_fps_configuration():
    diagnostics = FakeDiagnostics()
    calls = {}
    output_frames = ["annotated"]

    def fake_overlay(frames, tracks, config):
        calls["overlay"] = (frames, tracks, config)
        return output_frames

    def fake_save(frames, path, fps):
        calls["save"] = (frames, path, fps)

    with (
        patch("ft.pipeline.draw_overlay", side_effect=fake_overlay),
        patch("ft.pipeline.save_video", side_effect=fake_save),
    ):
        render_output_phase(
            {"overlay": {"enabled": True}, "tracking": {"frame_rate": 30}},
            frames=["raw"],
            tracks={"players": []},
            output_path="out.mp4",
            run_diagnostics=diagnostics,
        )

    assert calls["overlay"] == (["raw"], {"players": []}, {"enabled": True})
    assert calls["save"] == (output_frames, "out.mp4", 30)
    assert diagnostics.stages == ["overlay", "save_video"]


if __name__ == "__main__":
    test_temporal_phase_separates_hard_and_soft_cuts()
    test_tracking_phase_forwards_boundaries_and_annotates_activity()
    test_render_phase_preserves_overlay_and_fps_configuration()
    print("pipeline phase tests passed")
