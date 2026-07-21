import numpy as np

import ft.features.prtreid as prtreid_module
from ft.features.prtreid import PRTReIDFeatureExtractor, role_confidence
from ft.features.visual import VisualFeatureExtractor, normalize_embedding_mode


def test_visual_mode_accepts_prtreid():
    assert normalize_embedding_mode("prtreid") == "prtreid"
    assert normalize_embedding_mode("prt-reid") == "prtreid"


def test_prtreid_role_confidence_is_softmax_probability():
    logits = np.asarray([[0.0, 1.0, 3.0]], dtype=np.float32)
    confidence = role_confidence(logits, 0)
    expected = np.exp(3.0) / np.exp(logits[0]).sum()

    assert 0.0 <= confidence <= 1.0
    assert np.isclose(confidence, expected)


def test_visual_extractor_reuses_injected_prtreid_model():
    class ExistingExtractor:
        def add_row_features(self, rows):
            rows[0]["visual_embedding"] = [1.0, 0.0]

        def diagnostics(self):
            return {"backend": "prtreid", "computed": 1}

    existing = ExistingExtractor()
    extractor = VisualFeatureExtractor(
        embedding_mode="prtreid",
        prtreid_extractor=existing,
    )
    rows = [{}]

    extractor.add_row_features(rows)

    assert extractor.prtreid is existing
    assert rows[0]["visual_embedding"] == [1.0, 0.0]


def test_prtreid_extractor_populates_embedding_and_role(monkeypatch):
    monkeypatch.setattr(prtreid_module, "read_crop", lambda path: np.zeros((16, 8, 3), dtype=np.uint8))

    def fake_factory(_extractor):
        def run(crops, external_parts_masks=None):
            return [
                {
                    "visual_embedding": [3.0, 4.0],
                    "reid_visibility_scores": [1.0],
                    "reid_body_masks": [[[1.0]]],
                    "reid_role_detection": "player",
                    "reid_role_confidence": 0.91,
                }
                for _crop in crops
            ]

        return run

    extractor = PRTReIDFeatureExtractor(enabled=True, extractor_factory=fake_factory)
    rows = [{"crop_path": "fake.jpg"}]

    extractor.add_row_features(rows)

    assert rows[0]["reid_model"] == "prtreid"
    assert rows[0]["reid_role_detection"] == "player"
    assert rows[0]["reid_role_confidence"] == 0.91
    assert np.allclose(rows[0]["visual_embedding"], [0.6, 0.8])
    assert extractor.diagnostics()["embedding_dim"] == 2


def test_prtreid_missing_crop_records_failure(monkeypatch):
    monkeypatch.setattr(prtreid_module, "read_crop", lambda path: None)
    extractor = PRTReIDFeatureExtractor(enabled=True, extractor_factory=lambda _extractor: lambda crops, external_parts_masks=None: [])
    rows = [{"crop_path": "missing.jpg"}]

    extractor.add_row_features(rows)

    assert rows[0]["visual_embedding"] is None
    assert rows[0]["reid_model"] == "prtreid"
    assert rows[0]["reid_error"] == "missing_or_unreadable_crop"
    assert extractor.diagnostics()["failed"] == 1


def test_prtreid_reuses_existing_tracklet_prototype_without_inference():
    extractor = PRTReIDFeatureExtractor(
        enabled=True,
        extractor_factory=lambda _extractor: (_ for _ in ()).throw(AssertionError("must not initialize")),
    )
    rows = [{"visual_embedding": [1.0, 0.0], "reid_model": "prtreid"}]

    extractor.add_row_features(rows)

    assert extractor.diagnostics()["computed"] == 0
    assert extractor.diagnostics()["reused"] == 1


def test_visual_prtreid_filters_runtime_config_keys(monkeypatch):
    monkeypatch.setattr(
        "ft.features.visual.PRTReIDFeatureExtractor",
        lambda **kwargs: kwargs,
    )

    extractor = VisualFeatureExtractor(
        embedding_mode="prtreid",
        prtreid_config={
            "enabled": True,
            "role_enabled": True,
            "role_min_confidence": 0.6,
            "role_protect_existing": True,
        },
    )

    assert extractor.prtreid["enabled"] is True
    assert extractor.prtreid["role_enabled"] is True
    assert "role_min_confidence" not in extractor.prtreid
    assert "role_protect_existing" not in extractor.prtreid
