import hashlib

import pytest
from PIL import Image

from ft.features.jersey_frame_selector import (
    JerseyFrameSelector,
    composite_selection_score,
)


class FakeSelector(JerseyFrameSelector):
    def _load_model(self):
        return None

    def _score(self, display_track_id, rows):
        output = []
        for index, row in enumerate(rows):
            item = {
                "display_track_id": int(display_track_id),
                "row_index": index,
                "frame": int(row["frame"]),
                "crop_path": row.get("crop_path", f"{index}.jpg"),
                "pred_role": row.get("role_detection", "unknown"),
                "crop_quality": float(row.get("crop_quality", 1.0)),
                "legibility_score": float(row["score"]),
                "score_status": "ok",
            }
            if self.model_type in self.YOLO_MODEL_TYPES:
                item["selection_score"] = float(row["score"])
                if self.model_type == self.YOLO_READABILITY_MODEL_TYPE:
                    item["number_readability_score"] = float(row["score"])
                else:
                    item["clean_back_score"] = float(row["score"])
            output.append(item)
            self.score_rows.append(dict(item))
        return output


def make_selector(tmp_path, **overrides):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "device": "cpu",
        "top_k": 3,
        "min_legibility_score": 0.5,
        "min_frame_gap": 5,
    }
    config.update(overrides)
    return FakeSelector(**config)


def test_selector_is_deterministic_and_temporally_diverse(tmp_path):
    selector = make_selector(tmp_path)
    rows = [
        {"frame": 10, "score": 0.90, "crop_path": "b.jpg", "role_detection": "player"},
        {"frame": 12, "score": 0.95, "crop_path": "a.jpg", "role_detection": "player"},
        {"frame": 20, "score": 0.80, "crop_path": "c.jpg", "role_detection": "player"},
        {"frame": 30, "score": 0.40, "crop_path": "d.jpg", "role_detection": "player"},
        {"frame": 40, "score": 0.70, "crop_path": "e.jpg", "role_detection": "referee"},
    ]
    selected = selector.select(7, rows, min_crop_quality=0.08)
    assert [row["frame"] for row in selected] == [12, 20]
    reasons = {row["frame"]: row["selection_reason"] for row in selector.selection_rows}
    assert reasons[10] == "temporal_near_duplicate"
    assert reasons[30] == "below_legibility_threshold"
    assert reasons[40] == "role_not_allowed"


def test_selector_scores_but_rejects_low_quality(tmp_path):
    selector = make_selector(tmp_path)
    rows = [
        {"frame": 1, "score": 0.99, "crop_quality": 0.07, "role_detection": "player"},
        {"frame": 8, "score": 0.90, "crop_quality": 0.08, "role_detection": "player"},
    ]
    assert [row["frame"] for row in selector.select(4, rows, 0.08)] == [8]


def test_selector_decision_enforces_votes_and_margin(tmp_path):
    selector = make_selector(tmp_path, mode="propose", min_winner_votes=2, min_margin=0.10)
    assert not selector.accept_decision(1, {"jersey_number": 8, "votes": 1, "winner_margin": 0.9}, 2)
    assert not selector.accept_decision(2, {"jersey_number": 8, "votes": 2, "winner_margin": 0.09}, 2)
    assert selector.accept_decision(3, {"jersey_number": 8, "votes": 2, "winner_margin": 0.10}, 2)


def test_selector_rejects_missing_or_wrong_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        FakeSelector(checkpoint=tmp_path / "missing", checkpoint_sha256="0" * 64)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        FakeSelector(checkpoint=checkpoint, checkpoint_sha256="0" * 64)


def test_yolo_back_selector_ranks_composite_score(tmp_path):
    selector = make_selector(
        tmp_path, model_type="jersey_back_yolo11s_cls", mode="propose",
        top_k=2, min_selection_score=0.0, use_torso_crop=False,
    )
    rows = [
        {"frame": 10, "score": 0.60, "role_detection": "player"},
        {"frame": 12, "score": 0.95, "role_detection": "player"},
        {"frame": 20, "score": 0.80, "role_detection": "player"},
    ]
    assert [row["frame"] for row in selector.select(4, rows)] == [12, 20]


def test_yolo_back_selector_rejects_apply_mode(tmp_path):
    with pytest.raises(ValueError, match="audit/propose"):
        make_selector(tmp_path, model_type="jersey_back_yolo11s_cls", mode="apply")


def test_yolo_readability_selector_ranks_model_probability(tmp_path):
    selector = make_selector(
        tmp_path,
        model_type="jersey_number_readability_yolo26s_cls",
        mode="propose",
        top_k=2,
        min_selection_score=0.0,
        use_torso_crop=False,
    )
    rows = [
        {"frame": 10, "score": 0.60, "role_detection": "player"},
        {"frame": 12, "score": 0.95, "role_detection": "player"},
        {"frame": 20, "score": 0.80, "role_detection": "player"},
    ]
    assert [row["frame"] for row in selector.select(4, rows)] == [12, 20]


def test_yolo_readability_selector_rejects_apply_mode(tmp_path):
    with pytest.raises(ValueError, match="audit/propose"):
        make_selector(
            tmp_path,
            model_type="jersey_number_readability_yolo26s_cls",
            mode="apply",
        )


def test_composite_score_is_weighted_and_clamped():
    weights = {"clean_back": 0.7, "sharpness": 0.2, "size": 0.1, "crop_quality": 0.0}
    assert composite_selection_score(1.2, 0.5, -1.0, 0.5, weights) == pytest.approx(0.8)


def test_propose_materializes_torso_with_provenance(tmp_path):
    source = tmp_path / "player.jpg"
    Image.new("RGB", (100, 200), color=(20, 40, 60)).save(source)
    selector = make_selector(
        tmp_path, model_type="jersey_back_yolo11s_cls", mode="propose",
        top_k=1, min_selection_score=0.0,
    )
    rows = [{
        "frame": 10, "score": 0.9, "crop_path": str(source),
        "role_detection": "player", "crop_quality": 1.0,
    }]
    selected = selector.select(1, rows)
    torso = Image.open(selected[0]["crop_path"])
    assert torso.size == (70, 120)
    assert selected[0]["selector_original_crop_path"] == str(source)
    assert selected[0]["selector_crop_variant"] == "torso"
    audit = selector.selection_rows[0]
    assert audit["torso_crop_status"] == "ok"
    assert audit["selector_torso_crop_path"] == selected[0]["crop_path"]
