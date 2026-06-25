import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_prtreid_embeddings.py"
SPEC = importlib.util.spec_from_file_location("audit_prtreid_embeddings", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_embedding_audit_separates_reliable_tracklets():
    rows = [
        row(1, "p1", 1, [1.0, 0.0], confidence=0.90),
        row(1, "p1", 1, [0.9, 0.1], confidence=0.90),
        row(2, "p1", 1, [0.95, 0.05], confidence=0.88),
        row(2, "p1", 1, [1.0, 0.0], confidence=0.88),
        row(3, "p2", 1, [0.0, 1.0], confidence=0.92),
        row(3, "p2", 1, [0.1, 0.9], confidence=0.92),
        row(4, "p3", 2, [-1.0, 0.0], confidence=0.93),
        row(4, "p3", 2, [-0.9, 0.1], confidence=0.93),
    ]

    tracklets = audit.aggregate_tracklets(rows, min_frames=2, min_identity_confidence=0.75)
    report = audit.build_report(rows, tracklets)

    assert report["tracklet_summary"]["embedded_tracklets"] == 4
    assert report["tracklet_summary"]["identity_labeled_tracklets"] == 4
    assert report["pairwise"]["same_player"]["count"] == 1
    assert report["pairwise"]["same_player"]["p50"] > 0.99
    assert report["nearest_neighbor"]["player_top1_accuracy"] == 1.0
    assert report["nearest_neighbor"]["team_top1_accuracy"] == 0.75


def test_embedding_audit_filters_weak_identity_labels():
    rows = [
        row(1, "p1", 1, [1.0, 0.0], confidence=0.90),
        row(1, "p1", 1, [1.0, 0.0], confidence=0.90),
        row(2, "p2", 1, [0.0, 1.0], confidence=0.20),
        row(2, "p2", 1, [0.0, 1.0], confidence=0.20),
    ]

    tracklets = audit.aggregate_tracklets(rows, min_frames=2, min_identity_confidence=0.75)

    assert [item["player_id"] for item in tracklets] == ["p1", None]


def row(display_id, player_id, team_id, embedding, confidence=0.9):
    return {
        "track_group": "players",
        "display_track_id": str(display_id),
        "player_id": player_id,
        "team_id": str(team_id),
        "identity_confidence": str(confidence),
        "identity_status": "assigned",
        "visual_embedding": str(embedding),
        "reid_model": "prtreid",
        "role_detection": "player",
        "reid_role_detection": "player",
        "crop_quality": "0.3",
    }
