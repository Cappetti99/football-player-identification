import numpy as np

from ft.features.prtreid_tracklets import build_prtreid_tracklet_features, temporal_quality_sample
from ft.linking.prtreid_tracklet_linker import PRTReIDTrackletLinker


class FakeExtractor:
    def extract_crops(self, crops):
        return [
            {"visual_embedding": [1.0, 0.0] if float(crop.mean()) < 100 else [0.0, 1.0]}
            for crop in crops
        ]


def test_temporal_quality_sampling_is_deterministic():
    items = [
        {"frame": index, "raw_track_id": 1, "crop_quality": float(index % 3), "crop": None}
        for index in range(12)
    ]
    first = temporal_quality_sample(items, 4)
    second = temporal_quality_sample(list(reversed(items)), 4)
    assert [row["frame"] for row in first] == [row["frame"] for row in second]
    assert len(first) == 4


def test_tracklet_prototype_is_attached_and_reused():
    frames = [np.full((100, 100, 3), 20, dtype=np.uint8) for _ in range(4)]
    tracks = {
        "players": [
            {1: {"bbox": [10, 10, 50, 90], "display_track_id": 1}}
            for _ in frames
        ]
    }
    features = build_prtreid_tracklet_features(
        frames,
        tracks,
        FakeExtractor(),
        max_samples_per_tracklet=3,
        min_crop_quality=0.0,
        min_samples=2,
    )
    assert len(features) == 1
    assert features[0]["sample_count"] == 3
    assert features[0]["eligible"] is True
    for frame_tracks in tracks["players"]:
        assert frame_tracks[1]["prtreid_tracklet_embedding"] == [1.0, 0.0]
        assert frame_tracks[1]["prtreid_tracklet_sample_count"] == 3
        assert "visual_embedding" not in frame_tracks[1]


def test_same_scene_link_requires_similarity_margin_and_mutual_nearest():
    tracks = make_tracks(
        [
            (0, 1, [1.0, 0.0], 0, [10.0, 10.0]),
            (1, 1, [1.0, 0.0], 0, [12.0, 10.0]),
            (4, 2, [0.999, 0.02], 0, [15.0, 10.0]),
            (5, 2, [0.999, 0.02], 0, [17.0, 10.0]),
        ]
    )
    linker = PRTReIDTrackletLinker(
        same_scene={"min_similarity": 0.99, "min_margin": 0.0, "max_gap": 20, "max_distance": 50},
        cross_scene={"enabled": False},
    )
    display_map = linker.apply(tracks)
    assert display_map[2] == 1
    assert len(linker.diagnostics["accepted_links"]) == 1
    assert linker.diagnostics["accepted_links"][0]["mutual_nearest"] is True


def test_cross_scene_policy_is_independent():
    rows = [
        (0, 1, [1.0, 0.0], 0, [10.0, 10.0]),
        (1, 1, [1.0, 0.0], 0, [12.0, 10.0]),
        (3, 2, [1.0, 0.0], 1, [400.0, 400.0]),
        (4, 2, [1.0, 0.0], 1, [410.0, 400.0]),
    ]
    disabled_tracks = make_tracks(rows)
    disabled = PRTReIDTrackletLinker(cross_scene={"enabled": False})
    disabled_map = disabled.apply(disabled_tracks)
    assert disabled_map[2] == 2
    assert disabled.diagnostics["candidates"][0]["link_type"] == "cross_scene"
    assert disabled.diagnostics["rejection_counts"]["policy_disabled"] == 1

    enabled_tracks = make_tracks(rows)
    enabled = PRTReIDTrackletLinker(
        same_scene={"enabled": False},
        cross_scene={"enabled": True, "max_segment_gap": 1, "max_gap": 10, "min_similarity": 0.99, "min_margin": 0.0},
    )
    enabled_map = enabled.apply(enabled_tracks)
    assert enabled_map[2] == 1
    assert enabled.diagnostics["accepted_links"][0]["link_type"] == "cross_scene"


def test_missing_or_unreliable_prototype_never_merges():
    tracks = make_tracks(
        [
            (0, 1, [1.0, 0.0], 0, [10.0, 10.0]),
            (2, 2, None, 0, [12.0, 10.0]),
        ],
        sample_count=1,
    )
    linker = PRTReIDTrackletLinker(min_samples=3)
    display_map = linker.apply(tracks)
    assert display_map[2] == 2
    assert not linker.diagnostics["accepted_links"]


def make_tracks(rows, sample_count=4):
    frame_count = max(row[0] for row in rows) + 1
    frames = [{} for _ in range(frame_count)]
    for frame, track_id, embedding, segment, position in rows:
        frames[frame][track_id] = {
            "display_track_id": track_id,
            "prtreid_tracklet_embedding": embedding,
            "prtreid_tracklet_sample_count": sample_count,
            "prtreid_tracklet_consistency": 0.99 if embedding is not None else None,
            "team": 1,
            "team_confidence": 0.9,
            "scene_segment_id": segment,
            "position": position,
        }
    return {"players": frames}
