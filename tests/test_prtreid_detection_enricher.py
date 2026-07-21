import numpy as np
import pytest

from ft.tracking.observations import DetectionObservation, FrameDetections
from ft.tracking.prtreid_enricher import PRTReIDDetectionEnricher, crop_bbox


class FakeExtractor:
    def extract_crops(self, crops):
        assert [crop.shape[:2] for crop in crops] == [(4, 5)]
        return [{
            "visual_embedding": [0.6, 0.8],
            "reid_role_detection": "player",
            "reid_role_confidence": 0.9,
            "reid_visibility_scores": [1.0],
            "reid_model": "prtreid",
        }]


def observation(class_name="player", bbox=None):
    return DetectionObservation(
        frame=3,
        bbox=bbox or [1, 2, 6, 6],
        confidence=0.8,
        class_id=0,
        class_name=class_name,
        role="player" if class_name == "player" else "ball",
    )


def test_prtreid_enrichment_attaches_evidence_without_overwriting_detector_role():
    player = observation()
    ball = observation("ball")
    records = [FrameDetections(3, [player, ball], (8, 8))]
    enricher = PRTReIDDetectionEnricher(extractor=FakeExtractor())

    returned = enricher.enrich_batch([np.zeros((8, 8, 3), dtype=np.uint8)], records)

    assert returned is records
    assert player.embedding == [0.6, 0.8]
    assert player.role == "player"
    assert player.role_scores == {"player": 0.9}
    assert player.metadata["reid_role_detection"] == "player"
    assert player.athlete_payload()["visual_embedding"] == [0.6, 0.8]
    assert ball.embedding is None
    diagnostics = enricher.diagnostics()
    assert diagnostics["computed"] == 1
    assert diagnostics["coverage"] == 1.0
    assert diagnostics["embedding_dim"] == 2
    assert diagnostics["embedding_norm_mean"] == 1.0
    assert diagnostics["role_counts"] == {"player": 1}
    assert diagnostics["role_confidence_mean"] == 0.9
    assert diagnostics["visibility_mean"] == 1.0


def test_crop_bbox_clips_to_image_and_rejects_empty_boxes():
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    assert crop_bbox(image, [-2, -3, 4.2, 5.1]).shape[:2] == (6, 5)
    assert crop_bbox(image, [20, 20, 21, 21]) is None


def test_enrichment_rejects_misaligned_batches():
    enricher = PRTReIDDetectionEnricher(extractor=FakeExtractor())
    with pytest.raises(ValueError, match="one detection record per frame"):
        enricher.enrich_batch([], [FrameDetections(0, [], (8, 8))])
