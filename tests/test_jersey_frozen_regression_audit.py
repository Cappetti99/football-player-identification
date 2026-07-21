from scripts.build_jersey_frozen_regression_audit import selected_crop_paths


def test_selected_crop_paths_uses_nested_tracklets():
    diagnostics = {
        "enabled": True,
        "tracklets": {
            "one": {
                "display_track_id": 7,
                "selected_crops": [
                    {"crop_path": "a.jpg"},
                    {"crop_path": "b.jpg"},
                ],
            }
        },
    }
    assert selected_crop_paths(diagnostics, "7") == ["a.jpg", "b.jpg"]
    assert selected_crop_paths(diagnostics, "8") == []
