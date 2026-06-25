import importlib.util
from pathlib import Path


CALIBRATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_prtreid_linker.py"
CALIBRATE_SPEC = importlib.util.spec_from_file_location("calibrate_prtreid_linker", CALIBRATE_PATH)
calibrate = importlib.util.module_from_spec(CALIBRATE_SPEC)
CALIBRATE_SPEC.loader.exec_module(calibrate)

EXPORT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_prtreid_dataset.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_prtreid_dataset", EXPORT_PATH)
export_dataset = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(export_dataset)


def test_calibration_selects_zero_false_positive_threshold():
    rows = [
        {"label": "same", "similarity": 0.998, "margin": 0.04},
        {"label": "same", "similarity": 0.996, "margin": 0.03},
        {"label": "different", "similarity": 0.997, "margin": 0.01},
        {"label": "different", "similarity": 0.990, "margin": 0.05},
    ]
    threshold = calibrate.select_zero_false_positive_threshold(rows)
    metrics = calibrate.evaluate(rows, threshold)
    assert metrics["false_positives"] == 0
    assert metrics["true_positives"] >= 1


def test_manifest_split_has_no_sequence_leakage():
    rows = []
    for source_split, video, identity in (("train", "train_video", "train:p1"), ("valid", "valid_video", "valid:p1")):
        for frame in range(6):
            rows.append(
                {
                    "img_path": f"/{video}_{frame}.jpg",
                    "source_split": source_split,
                    "identity_key": identity,
                    "pid": None,
                    "camid": video,
                    "video_id": video,
                    "team": "left",
                    "role": "player",
                    "jersey_number": 7,
                    "frame": frame,
                    "split": source_split,
                    "label_source": "soccernet_gsr_ground_truth",
                }
            )
    manifest = export_dataset.finalize_splits(rows, min_samples=4, max_samples=6, query_ratio=0.2)
    export_dataset.validate_manifest(manifest)
    assert {row["video_id"] for row in manifest if row["split"] == "train"} == {"train_video"}
    assert {row["video_id"] for row in manifest if row["split"] in {"query", "gallery"}} == {"valid_video"}
    train_pids = sorted({row["pid"] for row in manifest if row["split"] == "train"})
    valid_pids = sorted({row["pid"] for row in manifest if row["split"] in {"query", "gallery"}})
    assert train_pids == list(range(len(train_pids)))
    assert valid_pids == list(range(len(valid_pids)))
