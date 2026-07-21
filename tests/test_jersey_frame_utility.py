import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_gsr_jersey_utility_manifest import repair_class_coverage, stratified_track_limit


def test_manifest_builder_is_sequence_disjoint_and_deterministic(tmp_path):
    root = tmp_path / "GSR"
    for sequence_index in range(3):
        build_sequence(root, f"SNGS-{sequence_index:03d}", jersey=7)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        run_builder(root, output)
    manifest = json.loads((first / "manifest.json").read_text())
    assert set(manifest["train_sequences"]).isdisjoint(manifest["validation_sequences"])
    assert (first / "train.jsonl").read_text() == (second / "train.jsonl").read_text()
    assert (first / "validation.jsonl").read_text() == (second / "validation.jsonl").read_text()
    assert json.loads((first / "summary.json").read_text())["load_stats"]["accepted_tracks"] == 3


def test_manifest_builder_refuses_test_without_unlock(tmp_path):
    root = tmp_path / "GSR"
    for sequence_index in range(2):
        build_sequence(root, f"SNGS-{sequence_index:03d}", split="test", jersey=7)
    command = builder_command(root, tmp_path / "output", split="test")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "test is frozen" in result.stderr


def test_tracklet_cap_preserves_imbalanced_partitions():
    parts = {
        "train": [{"sequence": "train", "track_id": str(index)} for index in range(500)],
        "validation": [{"sequence": "validation", "track_id": "1"}],
    }
    limited = stratified_track_limit(parts, limit=100, seed=7)
    assert sum(len(rows) for rows in limited.values()) == 100
    assert len(limited["validation"]) == 1
    assert len(limited["train"]) == 99


def test_sequence_split_repairs_validation_only_class():
    tracks = [
        {"sequence": "train_a", "track_id": "1", "jersey": 7},
        {"sequence": "train_b", "track_id": "2", "jersey": 7},
        {"sequence": "validation_common", "track_id": "3", "jersey": 7},
        {"sequence": "validation_unique", "track_id": "4", "jersey": 38},
    ]
    train, validation, repairs = repair_class_coverage(
        tracks,
        train_sequences={"train_a", "train_b"},
        validation_sequences={"validation_common", "validation_unique"},
        seed=7,
    )
    train_classes = {row["jersey"] for row in tracks if row["sequence"] in train}
    validation_classes = {row["jersey"] for row in tracks if row["sequence"] in validation}
    assert validation_classes <= train_classes
    assert "validation_unique" in train
    assert len(validation) == 2
    assert repairs[0]["classes"] == [38]


def build_sequence(root, sequence, split="train", jersey=7):
    sequence_dir = root / split / sequence
    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True)
    images = []
    annotations = []
    for frame in range(1, 9):
        filename = f"{frame:06d}.jpg"
        (image_dir / filename).write_bytes(b"fake")
        images.append({"image_id": frame, "file_name": filename, "frame": frame})
        annotations.append({
            "id": frame,
            "image_id": frame,
            "track_id": 11,
            "bbox_image": {"x": 10, "y": 20, "w": 30, "h": 60},
            "attributes": {"role": "player", "jersey": jersey},
        })
    payload = {"info": {"version": "1.3"}, "images": images, "annotations": annotations}
    (sequence_dir / "Labels-GameState.json").write_text(json.dumps(payload))


def run_builder(root, output):
    subprocess.run(builder_command(root, output), check=True, capture_output=True, text=True)


def builder_command(root, output, split="train"):
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_gsr_jersey_utility_manifest.py"
    return [
        sys.executable, str(script),
        "--gsr-dir", str(root),
        "--split", split,
        "--output-dir", str(output),
        "--seed", "20260716",
        "--min-track-frames", "8",
        "--max-frames-per-track", "8",
    ]
