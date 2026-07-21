from scripts.evaluate_jersey_number_region_ctc_ocr_run import selected_crops, summarize


def test_all_track_summary_counts_abstentions():
    rows = [
        {"assigned": True, "correct": True, "gt_in_top5": True},
        {"assigned": True, "correct": False, "gt_in_top5": False},
        {"assigned": False, "correct": False, "gt_in_top5": False},
    ]
    result = summarize(rows, selected=[1, 2], regions=[1])
    assert result["tracklets"] == 3
    assert result["coverage"] == 2 / 3
    assert result["accuracy_all"] == 1 / 3


def test_selected_crops_ignores_global_diagnostic_flags(tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"x")
    diagnostics = {
        "enabled": True,
        "track_1": {
            "display_track_id": 1,
            "selected_crops": [{"crop_path": str(crop), "frame": 7}],
        },
    }
    predictions = {
        ("S", "T"): {"eval_track_id": "1", "gt": 10},
    }
    rows = selected_crops(diagnostics, predictions)
    assert len(rows) == 1
    assert rows[0]["frame"] == 7


def test_selected_crops_reads_tracklets_container(tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"x")
    diagnostics = {
        "enabled": True,
        "tracklets": {
            "1": {
                "display_track_id": 1,
                "selected_crops": [{"crop_path": str(crop), "frame": 9}],
            },
        },
    }
    predictions = {
        ("S", "T"): {"eval_track_id": "1", "gt": 10},
    }
    rows = selected_crops(diagnostics, predictions)
    assert len(rows) == 1
    assert rows[0]["frame"] == 9
