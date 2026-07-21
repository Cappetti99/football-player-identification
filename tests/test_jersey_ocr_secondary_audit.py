import sys
from unittest.mock import MagicMock

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = MagicMock()

from ft.pipeline import compare_ocr_assignments, secondary_ocr_proposals
from ft.validation import _validate_jersey_ocr_secondary_audit


def test_secondary_proposals_apply_frozen_votes_and_margin():
    assignments = {
        1: {"jersey_number": 10, "votes": 4, "winner_margin": 0.08},
        2: {"jersey_number": 24, "votes": 3, "winner_margin": 0.50},
        3: {"jersey_number": 14, "votes": 5, "winner_margin": 0.079},
    }

    proposals = secondary_ocr_proposals(assignments, 4, 0.08)

    assert set(proposals) == {"1"}
    assert proposals["1"]["applied"] is False


def test_secondary_comparison_is_diagnostic_only():
    primary = {1: {"jersey_number": 10}, 2: {"jersey_number": 20}}
    secondary = {1: {"jersey_number": 10}, 3: {"jersey_number": 30}}

    result = compare_ocr_assignments(primary, secondary)

    assert result["agreement"] == 1
    assert result["primary_only"] == 1
    assert result["secondary_only"] == 1
    assert result["disagreement"] == 0


def test_secondary_validation_rejects_apply_and_requires_weights():
    errors = []
    _validate_jersey_ocr_secondary_audit(
        {"enabled": True, "mode": "apply", "backend": "mmocr_rec"},
        errors,
        [],
    )

    assert "jersey_ocr_secondary_audit supports only audit/propose" in errors
    assert "jersey_ocr_secondary_audit.mmocr_rec_weights is required" in errors
