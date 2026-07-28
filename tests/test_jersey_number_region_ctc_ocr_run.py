import json

from scripts.evaluate_jersey_number_region_ctc_ocr_run import dump_raw_scores, selected_crops, summarize


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


def test_dump_raw_scores_writes_one_entry_per_track(tmp_path):
    tracks = {
        ("S", "T1"): {
            "frames": [1, 40],
            "weights": [0.9, 0.6],
            "scores": [{"7": -0.1, "17": -2.0}, {"7": -0.3, "17": -1.5}],
        },
    }
    predictions = {("S", "T1"): {"eval_track_id": "3", "gt": 7}}
    out_path = tmp_path / "raw_scores.json"

    dump_raw_scores(out_path, tracks, predictions)

    payload = json.loads(out_path.read_text())
    assert list(payload) == ["S::T1"]
    entry = payload["S::T1"]
    assert entry["eval_track_id"] == "3"
    assert entry["gt_jersey_number"] == 7
    assert entry["frames"] == [1, 40]
    assert entry["weights"] == [0.9, 0.6]
    assert entry["scores"][0]["7"] == -0.1
