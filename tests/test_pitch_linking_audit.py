import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_pitch_linking.py"
SPEC = importlib.util.spec_from_file_location("audit_pitch_linking", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_pitch_gate_flags_impossible_current_link_without_using_gt_for_gate():
    tracks = {
        1: {
            "raw_id": 1, "display_id": 1, "start": 0, "end": 10, "frames": 11,
            "first_pixel": [0, 0], "last_pixel": [100, 100],
            "first_pitch": [10, 10], "last_pitch": [10, 10],
        },
        2: {
            "raw_id": 2, "display_id": 1, "start": 15, "end": 20, "frames": 6,
            "first_pixel": [105, 100], "last_pixel": [110, 100],
            "first_pitch": [30, 10], "last_pitch": [31, 10],
        },
    }
    rows = audit.build_audit(
        tracks, gt_by_raw={1: "a", 2: "b"}, fps=25, max_gap=90,
        max_speed_mps=12, calibration_source="tvcalib:test",
    )

    assert len(rows) == 1
    assert rows[0]["required_speed_mps"] == 100.0
    assert rows[0]["pitch_gate_pass"] is False
    assert rows[0]["would_block_current_link"] is True
    assert rows[0]["gt_same_identity_offline"] is False


def test_pitch_gate_abstains_when_endpoint_is_outside_pitch():
    tracks = {
        1: {
            "raw_id": 1, "display_id": 1, "start": 0, "end": 10, "frames": 11,
            "first_pixel": [0, 0], "last_pixel": [100, 100],
            "first_pitch": [10, 10], "last_pitch": [200, 10],
        },
        2: {
            "raw_id": 2, "display_id": 1, "start": 15, "end": 20, "frames": 6,
            "first_pixel": [105, 100], "last_pixel": [110, 100],
            "first_pitch": [30, 10], "last_pitch": [31, 10],
        },
    }

    row = audit.build_audit(tracks, fps=25)[0]
    assert row["pitch_usable"] is False
    assert row["pitch_gate_pass"] is None
    assert row["would_block_current_link"] is False
