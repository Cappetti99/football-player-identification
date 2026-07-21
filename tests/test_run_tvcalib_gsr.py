from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tvcalib_gsr.py"
SPEC = spec_from_file_location("run_tvcalib_gsr", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sample_evenly_matches_historical_750_frame_sampling() -> None:
    frames = [Path(f"{index:06d}.jpg") for index in range(1, 751)]

    selected = MODULE.sample_evenly(frames, 30)

    assert len(selected) == 30
    assert [int(path.stem) for path in selected] == list(range(1, 727, 25))


def test_sample_evenly_keeps_short_inputs() -> None:
    frames = [Path("000001.jpg"), Path("000002.jpg")]

    assert MODULE.sample_evenly(frames, 30) == frames
