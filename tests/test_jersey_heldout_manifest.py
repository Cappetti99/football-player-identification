import importlib.util
import json
from pathlib import Path



spec = importlib.util.spec_from_file_location(
    "build_jersey_heldout_manifest",
    Path(__file__).resolve().parents[1] / "scripts" / "build_jersey_heldout_manifest.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_exclusions_are_derived_with_their_evidence(tmp_path):
    (tmp_path / "handoff.md").write_text("frozen: SNGS-067 SNGS-068\n")
    (tmp_path / "notes.json").write_text(json.dumps({"pilot": ["SNGS-068", "SNGS-088"]}))

    excluded, scanned = builder.scan_exclusions([tmp_path])

    assert scanned == 2
    assert set(excluded) == {"SNGS-067", "SNGS-068", "SNGS-088"}
    # A sequence used twice records both sources, so the reason is auditable.
    assert len(excluded["SNGS-068"]) == 2


def test_binary_and_ignored_directories_are_skipped(tmp_path):
    (tmp_path / "a.md").write_text("SNGS-001")
    (tmp_path / "model.pt").write_bytes(b"SNGS-999")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "b.md").write_text("SNGS-998")

    excluded, scanned = builder.scan_exclusions([tmp_path])

    assert set(excluded) == {"SNGS-001"}
    assert scanned == 1


class FakeRunEval:
    """Minimal stand-in for the evaluator's GT helpers."""

    @staticmethod
    def read_json(path):
        return json.loads(Path(path).read_text())

    @staticmethod
    def resolve_role(annotation, categories):
        return (annotation.get("attributes") or {}).get("role")

    @staticmethod
    def jersey_number_from_attributes(attributes):
        value = attributes.get("jersey_number")
        return None if value in (None, "") else int(value)

    @staticmethod
    def infer_split(path):
        return "train" if "train" in Path(path).parts else "val"


def labels(annotations):
    return {"categories": [], "annotations": annotations}


def annotation(track, role="player", jersey=None):
    return {"track_id": track, "attributes": {"role": role, "jersey_number": jersey}}


def write_sequence(root, split, name, annotations):
    directory = root / split / name
    directory.mkdir(parents=True)
    (directory / "Labels-GameState.json").write_text(json.dumps(labels(annotations)))


def test_stats_count_tracks_not_annotations(tmp_path):
    write_sequence(tmp_path, "train", "SNGS-500", [
        annotation("1", jersey=10), annotation("1", jersey=10),
        annotation("2", jersey=7),
        annotation("3"),                       # no jersey GT
        annotation("4", role="referee", jersey=5),  # role filtered out
    ])
    stats = builder.sequence_jersey_stats(
        tmp_path / "train" / "SNGS-500" / "Labels-GameState.json",
        FakeRunEval, {"player", "goalkeeper"},
    )
    assert stats["tracks"] == 3
    assert stats["tracks_with_jersey"] == 2
    assert stats["annotations_with_jersey"] == 3
    assert stats["distinct_numbers"] == 2


def test_pool_reports_why_each_sequence_was_dropped(tmp_path):
    write_sequence(tmp_path, "train", "SNGS-500", [annotation("1", jersey=10)])
    write_sequence(tmp_path, "train", "SNGS-501", [annotation("1", jersey=9)])
    write_sequence(tmp_path, "val", "SNGS-502", [annotation("1", jersey=8)])

    pool, skipped = builder.build_pool(
        tmp_path, "train", FakeRunEval, {"player"}, {"SNGS-501": ["handoff.md"]}
    )

    assert [entry["sequence"] for entry in pool] == ["SNGS-500"]
    assert skipped == [
        {"sequence": "SNGS-501", "reason": "already_used", "evidence": ["handoff.md"]}
    ]


def test_split_filter_excludes_other_splits(tmp_path):
    write_sequence(tmp_path, "val", "SNGS-502", [annotation("1", jersey=8)])
    pool, _ = builder.build_pool(tmp_path, "train", FakeRunEval, {"player"}, {})
    assert pool == []


def test_selection_is_deterministic_for_a_seed():
    import random

    pool = sorted(f"SNGS-{index:03d}" for index in range(500, 540))
    first = sorted(random.Random(20260726).sample(pool, 15))
    second = sorted(random.Random(20260726).sample(pool, 15))
    third = sorted(random.Random(1).sample(pool, 15))
    assert first == second
    assert first != third
