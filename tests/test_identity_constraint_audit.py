import importlib.util
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_identity_constraints.py"
SPEC = importlib.util.spec_from_file_location("audit_identity_constraints", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_audit_deduplicates_constraint_rows_by_frame_raw_id_and_reason():
    action = {
        "action_type": "global_team_jersey_owners",
        "reason": "global_duplicate_team_jersey_owner",
        "team_id": "1",
        "jersey_number": "2",
        "cleared_player_id": "team1_02",
        "kept_player_id": "team1_alt02",
        "cleared_display_track_ids": "[9]",
        "kept_display_track_ids": "[49]",
        "cleared_row_keys": '[{"frame": 10, "raw_track_id": 90}, {"frame": 11, "raw_track_id": 90}]',
    }

    rows = audit.normalize_actions([action, dict(action)], {9: {10, 11}, 49: {20, 21}})

    assert len(rows) == 1
    assert rows[0]["row_keys"] == [(10, 90), (11, 90)]
    assert rows[0]["classification"] == "competing_known_owners"


def test_audit_classifies_same_player_global_clear_as_bug():
    assert audit.classify_action(
        "global_duplicate_team_jersey_owner",
        "team1_02",
        "team1_02",
    ) == "same_player_global_clear_bug"


def test_audit_keeps_identity_and_jersey_loss_separate_for_unknown_owner():
    rows = audit.normalize_actions(
        [
            {
                "reason": "global_duplicate_team_jersey_owner",
                "team_id": "1",
                "jersey_number": "2",
                "cleared_player_id": "unknown",
                "kept_player_id": "team1_02",
                "cleared_num_rows": "3",
            }
        ],
        {},
    )
    groups = audit.aggregate_actions(rows, {}, {9: {"player_id": "unknown", "identity_status": "unknown"}})

    assert groups[0]["unique_rows_cleared"] == 3
    assert groups[0]["identity_rows_cleared"] == 0
    assert groups[0]["jersey_rows_cleared"] == 3
    assert groups[0]["classification"] == "unknown_owner_vs_known_owner"
    assert groups[0]["preconstraint_assignment_player_ids"] == []
    assert groups[0]["preconstraint_assignment_statuses"] == []


def test_audit_writes_csv_header_when_no_constraints_fire():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit.csv"
        audit.write_csv([], path, audit.GROUP_FIELDS)

        assert path.read_text(encoding="utf-8").splitlines()[0].startswith("action_type,team_id,jersey_number")
