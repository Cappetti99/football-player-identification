import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproject_tracklets_tvcalib.py"
SPEC = importlib.util.spec_from_file_location("reproject_tracklets_tvcalib", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_bbox_bottom_middle():
    assert module.bbox_bottom_middle([10, 20, 30, 60]) == [20.0, 60]


def test_vector_accepts_exported_json_and_empty_values():
    assert module.vector("[1.5, 2.5]") == [1.5, 2.5]
    assert module.vector("") is None
