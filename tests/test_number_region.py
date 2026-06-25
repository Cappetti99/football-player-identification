import numpy as np

from ft.features.number_region import NumberRegionAuditor, build_tracklet_evidence, normalize_partial_number


class FakeRecognizer:
    def recognize(self, rows):
        tracklets = {}
        for row in rows:
            key = str(row["display_track_id"])
            tracklets.setdefault(key, {"detections": []})["detections"].append(
                {
                    "crop_path": row["crop_path"],
                    "frame": row["frame"],
                    "text": "?7",
                    "confidence": 0.8,
                    "source": "fake",
                    "variant": "raw",
                }
            )
        return {}, {"status": "ok", "backend": "fake", "tracklets": tracklets}


def test_number_region_audit_is_separate_and_preserves_partial_evidence(tmp_path):
    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    rows = [
        {
            "frame": 0,
            "raw_track_id": 4,
            "display_track_id": 9,
            "track_group": "players",
            "role_detection": "player",
            "bbox": [10, 10, 50, 90],
            "crop_quality": 0.7,
            "player_id": "known-must-not-change",
        }
    ]
    detections, evidence, diagnostics = NumberRegionAuditor(tmp_path, frame_interval=5).run(
        [frame], rows, FakeRecognizer()
    )

    assert detections[0]["recognition_normalized"] == "?7"
    assert evidence[0]["winner"] == "?7"
    assert diagnostics["identity_mutations"] == 0
    assert rows[0]["player_id"] == "known-must-not-change"


def test_tracklet_evidence_counts_consecutive_support():
    rows = [
        {"display_track_id": 1, "frame": frame, "recognition_normalized": value, "recognition_confidence": 0.8}
        for frame, value in [(0, "2"), (5, "2"), (10, "7"), (15, "2")]
    ]
    evidence = build_tracklet_evidence(rows, frame_interval=5)
    assert evidence[0]["winner"] == "2"
    assert evidence[0]["winner_votes"] == 3
    assert evidence[0]["max_consecutive_support"] == 2


def test_partial_number_normalization_does_not_complete_digits():
    assert normalize_partial_number("?7") == "?7"
    assert normalize_partial_number("1?") == "1?"
    assert normalize_partial_number("007") is None
