from ft.features.jersey_region_ctc_audit import (
    collect_selected_crops,
    conservative_preview,
)
from ft.validation import _validate_jersey_region_ctc_audit


def test_collect_selected_crops_is_deduplicated(tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"x")
    diagnostics = {
        "tracklets": {
            "one": {
                "display_track_id": 7,
                "selected_crops": [
                    {"crop_path": str(crop), "frame": 10},
                    {"crop_path": str(crop), "frame": 10},
                ],
            }
        }
    }
    rows = collect_selected_crops(diagnostics)
    assert len(rows) == 1
    assert rows[0]["display_track_id"] == 7


def test_collect_selected_crops_enforces_topk_and_temporal_gap(tmp_path):
    crops = []
    scores = []
    for frame, score in ((1, 0.8), (2, 0.9), (10, 0.7), (20, 0.6)):
        path = tmp_path / f"{frame}.jpg"
        path.write_bytes(b"x")
        crops.append({"crop_path": str(path), "frame": frame})
        scores.append({
            "crop_path": str(path), "selection_score": score,
            "selection_reason": "selected", "selection_rank": frame,
        })
    diagnostics = {"tracklets": {"one": {"display_track_id": 7, "selected_crops": crops}}}
    rows = collect_selected_crops(
        diagnostics,
        frame_selection_rows=scores,
        max_crops_per_tracklet=2,
        min_frame_gap=5,
    )
    assert [row["frame"] for row in rows] == [2, 10]
    assert rows[0]["selection_reason"] == "selected"
    assert rows[0]["selection_rank"] == 2


def test_conservative_preview_never_fills_primary_abstention():
    assert conservative_preview(None, 10, 0.99, 0.90) == (
        None,
        "primary_abstention_preserved",
    )
    assert conservative_preview(11, 10, 0.89, 0.90) == (
        11,
        "candidate_below_threshold",
    )
    assert conservative_preview(11, 10, 0.90, 0.90) == (
        10,
        "high_confidence_override",
    )


def test_region_ctc_validation_enforces_audit_and_hashes():
    errors = []
    _validate_jersey_region_ctc_audit(
        {
            "enabled": True,
            "mode": "apply",
            "audit_only": False,
            "ctc_checkpoint": "ctc.pth",
            "ctc_checkpoint_sha256": "bad",
            "detector_checkpoint": "detector.pt",
            "detector_checkpoint_sha256": "bad",
            "detector_confidence": 1.1,
            "min_override_confidence": 0.9,
            "box_padding": 0.1,
            "fusion_preview_enabled": "false",
            "batch_size": 1,
            "detector_batch_size": 1,
        },
        errors,
    )
    assert "jersey_region_ctc_audit supports only audit" in errors
    assert "jersey_region_ctc_audit.audit_only must remain true" in errors
    assert any("ctc_checkpoint_sha256" in error for error in errors)
    assert any("detector_checkpoint_sha256" in error for error in errors)
    assert any("fusion_preview_enabled" in error for error in errors)
