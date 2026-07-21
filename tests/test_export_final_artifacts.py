"""Contract test for ft.pipeline.export_final_artifacts.

This locks in the argument wiring between `_run_pipeline_impl` and the
extracted artifact-writing phase (2026-07-21 refactor). It exists because a
mismatch here (a name computed in one function but only used in the other)
is a plain NameError that pyflakes already catches statically — this test is
a second, behavioral line of defense: it actually calls the function with a
minimal but complete set of inputs and checks the write succeeds and a couple
of key artifacts contain what was passed in.
"""
import json
from pathlib import Path

from ft.pipeline import export_final_artifacts


class FakeDiagnosticsSource:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def diagnostics(self):
        return self._payload


def test_export_final_artifacts_writes_expected_files_without_raising(tmp_path):
    artifacts_dir = tmp_path
    video_id = "TEST-VIDEO"

    export_final_artifacts(
        config={},
        artifacts_dir=artifacts_dir,
        video_id=video_id,
        tracks={},
        final_rows=[],
        calibrator=FakeDiagnosticsSource({"source": "auto:field_quad"}),
        scene_cut_diagnostics={"cut_frames": [], "cuts": []},
        activity_diagnostics={"segments": [], "frame_stats": []},
        compact_activity_diagnostics={"segments": []},
        linking_diagnostics={"enabled": False},
        prtreid_tracklet_features=[],
        prtreid_linking_diagnostics={},
        prtreid_bridge_diagnostics={"applied_rows": 0},
        team_assignments={},
        referee_diagnostics={},
        goalkeeper_diagnostics={},
        semantic_groups={},
        jersey_diagnostics={"status": "ok", "frame_selection": {}},
        secondary_ocr_diagnostics={},
        region_ctc_diagnostics={},
        jersey_linking_diagnostics={},
        constraints_diagnostics={},
        identity_propagation_diagnostics={},
        summaries=[],
        assignments={},
        candidate_scores=[],
        identity_candidates={},
        segment_jersey_candidates=[],
        segment_jersey_diagnostics={},
        segment_candidate_diagnostics={},
        identity_evidence=[],
        constraint_actions=[],
        identity_gate_audit={"summary": {"strong_visual_gate_available": True}, "tracklets": []},
        number_region_diagnostics={},
        number_region_detections=[],
        number_region_evidence=[],
        exporter=FakeDiagnosticsSource({"rows_written": 0}),
        visual_extractor=FakeDiagnosticsSource({"backend": "hsv"}),
        ocr=None,
    )

    metadata = artifacts_dir / "metadata"
    assert (metadata / f"{video_id}_identity_assignments.json").is_file()
    assert (metadata / f"{video_id}_export.json").is_file()

    assignments_payload = json.loads(
        (metadata / f"{video_id}_identity_assignments.json").read_text()
    )
    assert assignments_payload["jersey_ocr"]["status"] == "ok"

    export_payload = json.loads((metadata / f"{video_id}_export.json").read_text())
    assert export_payload == {"rows_written": 0}


def test_export_final_artifacts_handles_missing_ocr_selector_attributes(tmp_path):
    """ocr=None must not raise: frame_selector/subject_filter/prefix_consolidator
    are all read via getattr(ocr, ..., None) so a disabled jersey_ocr stage
    (where `ocr` is never constructed) must still export cleanly."""
    export_final_artifacts(
        config={},
        artifacts_dir=tmp_path,
        video_id="NO-OCR",
        tracks={},
        final_rows=[],
        calibrator=FakeDiagnosticsSource(),
        scene_cut_diagnostics={"cut_frames": [], "cuts": []},
        activity_diagnostics={"segments": [], "frame_stats": []},
        compact_activity_diagnostics={"segments": []},
        linking_diagnostics={},
        prtreid_tracklet_features=[],
        prtreid_linking_diagnostics={},
        prtreid_bridge_diagnostics={"applied_rows": 0},
        team_assignments={},
        referee_diagnostics={},
        goalkeeper_diagnostics={},
        semantic_groups={},
        jersey_diagnostics={"status": "disabled"},
        secondary_ocr_diagnostics={},
        region_ctc_diagnostics={},
        jersey_linking_diagnostics={},
        constraints_diagnostics={},
        identity_propagation_diagnostics={},
        summaries=[],
        assignments={},
        candidate_scores=[],
        identity_candidates={},
        segment_jersey_candidates=[],
        segment_jersey_diagnostics={},
        segment_candidate_diagnostics={},
        identity_evidence=[],
        constraint_actions=[],
        identity_gate_audit={"summary": {"strong_visual_gate_available": True}, "tracklets": []},
        number_region_diagnostics={},
        number_region_detections=[],
        number_region_evidence=[],
        exporter=FakeDiagnosticsSource(),
        visual_extractor=FakeDiagnosticsSource(),
        ocr=None,
    )

    # write_table skips creating a file for empty rows without fieldnames;
    # write_json always writes, so it is the reliable "the loop ran" signal.
    assert (Path(tmp_path) / "metadata" / "NO-OCR_jersey_frame_scores.json").is_file()
