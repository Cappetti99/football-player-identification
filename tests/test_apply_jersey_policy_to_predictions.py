import csv
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "apply_jersey_policy_to_predictions",
    Path(__file__).resolve().parents[1] / "scripts" / "apply_jersey_policy_to_predictions.py",
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

from ft.decision.jersey_policy import normalize_policy, pick_winner  # noqa: E402


BASELINE_FIELDS = [
    "eval_track_id", "sequence", "gt_track_id", "role", "team", "num_gt_frames",
    "gt_jersey_number", "pred_jersey_number", "assigned", "correct", "confidence",
    "head_confidence", "winner_margin", "votes", "total_detections", "candidates",
]
CTC_FIELDS = [
    "eval_track_id", "sequence", "gt_track_id", "gt_jersey_number",
    "pred_jersey_number", "assigned", "correct", "confidence", "winner_margin",
    "recognized_frames", "gt_in_top5",
]


def row(sequence, track, gt, pred, confidence=0.7, fields=BASELINE_FIELDS):
    base = {field: "" for field in fields}
    base.update({
        "eval_track_id": track, "sequence": sequence, "gt_track_id": track,
        "gt_jersey_number": gt,
        "pred_jersey_number": "" if pred is None else pred,
        "assigned": pred is not None,
        "correct": pred is not None and pred == gt,
        "confidence": confidence,
    })
    if "role" in base:
        base["role"] = "player"
        base["num_gt_frames"] = 40
    return base


def write(directory, rows, fields):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return directory


def surfaces(tmp_path, baseline_rows, ctc_rows):
    primary = write(tmp_path / "sar", baseline_rows, BASELINE_FIELDS)
    ctc = write(tmp_path / "ctc", ctc_rows, CTC_FIELDS)
    return {"jersey_ocr_primary": primary / "predictions.csv",
            "jersey_region_ctc": ctc / "predictions.csv"}


def run(tmp_path, baseline_rows, ctc_rows, on_abstain="fallback"):
    paths = surfaces(tmp_path, baseline_rows, ctc_rows)
    sources = {name: adapter.load_predictions(path) for name, path in paths.items()}
    adapter.validate_surfaces(sources)
    policy = normalize_policy(
        {"sources": ["jersey_ocr_primary", "jersey_region_ctc"], "on_abstain": on_abstain}
    )
    from ft.decision.jersey_policy import resolve_jersey_assignments
    _, diagnostics = resolve_jersey_assignments(adapter.build_evidence(sources, "h"), policy)
    fused, provenance = adapter.fuse(sources, "jersey_ocr_primary", diagnostics["decisions"])
    metrics = adapter.summarize(fused, sources["jersey_ocr_primary"], "jersey_ocr_primary")
    return fused, provenance, metrics


# --- the rule is the shipped one -------------------------------------------

def test_fallback_only_fills_abstentions(tmp_path):
    baseline = [row("S1", "1", 10, 10), row("S1", "2", 7, None), row("S1", "3", 4, 9)]
    ctc = [row("S1", "1", 10, 99, fields=CTC_FIELDS),
           row("S1", "2", 7, 7, fields=CTC_FIELDS),
           row("S1", "3", 4, 4, fields=CTC_FIELDS)]

    fused, _, metrics = run(tmp_path, baseline, ctc)
    by_track = {r["gt_track_id"]: r for r in fused}

    # decided by the primary: untouched, even though the CTC disagrees
    assert adapter.integer(by_track["1"]["pred_jersey_number"]) == 10
    assert by_track["1"]["decision_source"] == "jersey_ocr_primary"
    # abstained: the CTC fills it
    assert adapter.integer(by_track["2"]["pred_jersey_number"]) == 7
    assert by_track["2"]["decision_source"] == "jersey_region_ctc"
    # the primary was wrong but it decided, so the CTC does not rescue it:
    # a fallback policy adds coverage, it never corrects a wrong decision.
    assert adapter.integer(by_track["3"]["pred_jersey_number"]) == 9
    assert metrics["transitions"]["wrong->wrong"] == 1
    assert metrics["transitions"]["abstain->correct"] == 1


def test_no_correct_to_wrong_is_structural(tmp_path):
    """A fallback policy cannot regress a decided track, by construction."""
    baseline = [row("S1", str(i), 10, 10) for i in range(1, 6)]
    ctc = [row("S1", str(i), 10, 99, fields=CTC_FIELDS) for i in range(1, 6)]

    _, _, metrics = run(tmp_path, baseline, ctc)

    assert metrics["transitions"].get("correct->wrong", 0) == 0
    assert metrics["transitions"]["correct->correct"] == 5


def test_abstain_mode_changes_nothing(tmp_path):
    baseline = [row("S1", "1", 10, None)]
    ctc = [row("S1", "1", 10, 10, fields=CTC_FIELDS)]

    fused, _, metrics = run(tmp_path, baseline, ctc, on_abstain="abstain")

    assert fused[0]["pred_jersey_number"] == ""
    assert metrics["assigned"] == 0


def test_adapter_matches_pick_winner_directly(tmp_path):
    """The fused output must agree with the rule applied on its own."""
    baseline = [row("S1", "1", 10, None), row("S1", "2", 7, 7)]
    ctc = [row("S1", "1", 10, 5, fields=CTC_FIELDS),
           row("S1", "2", 7, 9, fields=CTC_FIELDS)]
    paths = surfaces(tmp_path, baseline, ctc)
    sources = {name: adapter.load_predictions(path) for name, path in paths.items()}
    policy = normalize_policy(
        {"sources": ["jersey_ocr_primary", "jersey_region_ctc"], "on_abstain": "fallback"}
    )
    evidence = adapter.build_evidence(sources, "h")

    by_subject = {}
    for item in evidence:
        by_subject.setdefault(item.subject_id, {})[item.produced_by] = item
    expected = {
        subject: pick_winner(candidates, policy["sources"], True)[0].value
        for subject, candidates in by_subject.items()
    }

    fused, _, _ = run(tmp_path / "again", baseline, ctc)
    actual = {
        adapter.subject_of(r["sequence"], r["gt_track_id"]): str(r["pred_jersey_number"])
        for r in fused
    }
    assert actual == {k: str(v) for k, v in expected.items()}


# --- surface hygiene --------------------------------------------------------

def test_surfaces_must_match(tmp_path):
    paths = surfaces(
        tmp_path,
        [row("S1", "1", 10, 10), row("S1", "2", 7, 7)],
        [row("S1", "1", 10, 10, fields=CTC_FIELDS)],
    )
    sources = {name: adapter.load_predictions(path) for name, path in paths.items()}
    with pytest.raises(SystemExit, match="surfaces differ"):
        adapter.validate_surfaces(sources)


def test_ground_truth_mismatch_is_fatal(tmp_path):
    paths = surfaces(
        tmp_path,
        [row("S1", "1", 10, 10)],
        [row("S1", "1", 11, 10, fields=CTC_FIELDS)],
    )
    sources = {name: adapter.load_predictions(path) for name, path in paths.items()}
    with pytest.raises(SystemExit, match="ground truth differs"):
        adapter.validate_surfaces(sources)


def test_provenance_records_every_source_prediction(tmp_path):
    baseline = [row("S1", "1", 10, None)]
    ctc = [row("S1", "1", 10, 5, fields=CTC_FIELDS)]

    _, provenance, _ = run(tmp_path, baseline, ctc)

    entry = provenance[0]
    assert entry["chosen_source"] == "jersey_region_ctc"
    assert entry["reason"] == "fallback_after_abstain"
    assert entry["pred_jersey_ocr_primary"] == ""
    assert adapter.integer(entry["pred_jersey_region_ctc"]) == 5


def test_baseline_only_columns_survive_a_ctc_win(tmp_path):
    """Surface metadata describes the track, not the prediction: keep it."""
    baseline = [row("S1", "1", 10, None)]
    ctc = [row("S1", "1", 10, 5, fields=CTC_FIELDS)]

    fused, _, _ = run(tmp_path, baseline, ctc)

    assert fused[0]["role"] == "player"
    assert adapter.integer(fused[0]["num_gt_frames"]) == 40


# --- the zero-regression guarantee is about ordering, not on_abstain --------

def test_guarantee_holds_only_when_baseline_is_the_first_source(tmp_path):
    """Ordering, not on_abstain, is what forbids correct->wrong."""
    baseline = [row("S1", "1", 10, 10)]
    ctc = [row("S1", "1", 10, 99, fields=CTC_FIELDS)]
    paths = surfaces(tmp_path, baseline, ctc)
    sources = {name: adapter.load_predictions(path) for name, path in paths.items()}
    from ft.decision.jersey_policy import resolve_jersey_assignments

    # baseline first: it keeps the track, no regression possible
    policy_a = normalize_policy(
        {"sources": ["jersey_ocr_primary", "jersey_region_ctc"], "on_abstain": "fallback"}
    )
    _, diag_a = resolve_jersey_assignments(adapter.build_evidence(sources, "h"), policy_a)
    fused_a, _ = adapter.fuse(sources, "jersey_ocr_primary", diag_a["decisions"])
    metrics_a = adapter.summarize(fused_a, sources["jersey_ocr_primary"], "jersey_ocr_primary")
    assert metrics_a["transitions"].get("correct->wrong", 0) == 0

    # baseline second: the same on_abstain now allows a regression
    policy_b = normalize_policy(
        {"sources": ["jersey_region_ctc", "jersey_ocr_primary"], "on_abstain": "fallback"}
    )
    _, diag_b = resolve_jersey_assignments(adapter.build_evidence(sources, "h"), policy_b)
    fused_b, _ = adapter.fuse(sources, "jersey_ocr_primary", diag_b["decisions"])
    metrics_b = adapter.summarize(fused_b, sources["jersey_ocr_primary"], "jersey_ocr_primary")
    assert metrics_b["transitions"]["correct->wrong"] == 1


def test_coverage_does_not_depend_on_ordering(tmp_path):
    """Whether a track is assigned depends on any source deciding, not on order."""
    baseline = [row("S1", "1", 10, None), row("S1", "2", 7, 7), row("S1", "3", 4, None)]
    ctc = [row("S1", "1", 10, 5, fields=CTC_FIELDS),
           row("S1", "2", 7, 9, fields=CTC_FIELDS),
           row("S1", "3", 4, None, fields=CTC_FIELDS)]
    paths = surfaces(tmp_path, baseline, ctc)
    sources = {name: adapter.load_predictions(path) for name, path in paths.items()}
    from ft.decision.jersey_policy import resolve_jersey_assignments

    coverages = []
    for order in (["jersey_ocr_primary", "jersey_region_ctc"],
                  ["jersey_region_ctc", "jersey_ocr_primary"]):
        policy = normalize_policy({"sources": order, "on_abstain": "fallback"})
        _, diagnostics = resolve_jersey_assignments(adapter.build_evidence(sources, "h"), policy)
        fused, _ = adapter.fuse(sources, "jersey_ocr_primary", diagnostics["decisions"])
        coverages.append(
            adapter.summarize(fused, sources["jersey_ocr_primary"], "jersey_ocr_primary")["coverage"]
        )
    assert coverages[0] == coverages[1]
