import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_identity_nearmiss.py"
SPEC = importlib.util.spec_from_file_location("calibrate_identity_nearmiss", SCRIPT)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def test_nearmiss_calibration_selects_zero_false_positive_threshold():
    rows = [
        {"score": 0.19, "label": "same"},
        {"score": 0.17, "label": "same"},
        {"score": 0.16, "label": "different"},
        {"score": 0.11, "label": "different"},
    ]
    threshold, metrics = calibration.select_threshold(rows)

    assert threshold == 0.17
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
