"""Regression tests for the GS-HOTA oracle ablation harness.

The harness attributes end-to-end error to individual architecture modules by
replacing them with a ground-truth oracle. Its own correctness is what makes
the resulting table meaningful, so these tests pin the properties that would
silently corrupt every reported number if they broke.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ft.evaluation.gsr_detection_tracking import gs_hota_summary


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_ablation():
    path = REPO_ROOT / "scripts" / "ablate_gs_hota_oracles.py"
    spec = importlib.util.spec_from_file_location("ablate_gs_hota_oracles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ablation = load_ablation()


def gt_row(track_id, x, y, team="left", role="player", jersey=7):
    return {
        "gt_track_id": track_id,
        "bbox": [x, y, x + 10, y + 20],
        "gt_position_pitch": [float(x), float(y)],
        "gt_team": team,
        "gt_role": role,
        "gt_jersey": jersey,
    }


def pred_row(track_id, x, y, team="left", role="player", jersey=7, position=None):
    identity = f"players:{track_id}"
    return {
        "pred_identity_id": identity,
        "raw_pred_identity_id": identity,
        "bbox": [x, y, x + 10, y + 20],
        "pred_position_pitch": position if position is not None else [float(x), float(y)],
        "pred_team": team,
        "pred_role": role,
        "pred_jersey": jersey,
    }


def scenario(frames=3, jersey_b=9, position_b=None):
    """Two players tracked over several frames; player 'b' carries the injected
    error so each test can attribute a known cause to a known module."""
    gt, pred = {}, {}
    for frame in range(frames):
        gt[frame] = [
            gt_row("a", 10, 20, team="left", jersey=7),
            gt_row("b", 50, 20, team="right", jersey=9),
        ]
        pred[frame] = [
            pred_row("a", 10, 20, team="left", jersey=7),
            pred_row("b", 50, 20, team="right", jersey=jersey_b, position=position_b),
        ]
    return gt, pred


def run(gt, pred, spec_name):
    lookups = ablation.matched_lookups(gt, pred, 0.50)
    configurations = ablation.build_configurations()
    candidate = ablation.apply_oracles(gt, pred, *lookups, configurations[spec_name])
    return gs_hota_summary(gt, candidate)["gs_hota"]


def test_full_oracle_reproduces_ground_truth_exactly():
    """The load-bearing sanity check: with every module oracle, predictions are
    the ground truth, so GS-HOTA must be exactly 1. If this drifts, no other
    row in the ablation table can be trusted."""
    gt, pred = scenario(jersey_b=4, position_b=[52.0, 20.0])
    assert run(gt, pred, "full_oracle") == pytest.approx(1.0, abs=1e-9)


def test_full_oracle_holds_even_when_the_detector_missed_a_player():
    """A ground-truth box the real system never produced must still be
    reconstructed by the detection oracle, not left as a false negative."""
    gt, pred = scenario()
    for frame in pred:
        pred[frame] = [row for row in pred[frame] if row["pred_identity_id"] != "players:b"]
    assert run(gt, pred, "full_oracle") == pytest.approx(1.0, abs=1e-9)


def test_oracle_repairs_only_the_module_it_owns():
    gt, pred = scenario(jersey_b=4)
    baseline = run(gt, pred, "baseline")
    assert run(gt, pred, "oracle_jersey") > baseline
    # Team and role were already correct; their oracles change nothing.
    assert run(gt, pred, "oracle_team") == pytest.approx(baseline)
    assert run(gt, pred, "oracle_role") == pytest.approx(baseline)


def test_leaving_the_broken_module_real_keeps_the_baseline_score():
    """When one module holds the only error, making every *other* module
    perfect cannot help: the isolated-cost column must stay at baseline."""
    gt, pred = scenario(jersey_b=4)
    assert run(gt, pred, "oracle_all_but_jersey") == pytest.approx(
        run(gt, pred, "baseline")
    )


def test_leaving_a_correct_module_real_costs_nothing():
    gt, pred = scenario(jersey_b=4)
    assert run(gt, pred, "oracle_all_but_team") == pytest.approx(1.0, abs=1e-9)
    assert run(gt, pred, "oracle_all_but_role") == pytest.approx(1.0, abs=1e-9)


def test_position_error_is_attributed_to_calibration():
    """A 2 m offset with no attribute errors must show up as a calibration
    cost, and nowhere else."""
    gt, pred = scenario(jersey_b=9, position_b=[52.0, 20.0])
    assert run(gt, pred, "oracle_all_but_calibration") < 1.0
    assert run(gt, pred, "oracle_calibration") == pytest.approx(1.0, abs=1e-9)


def test_identity_similarity_is_multiplicative_not_additive():
    """GS-HOTA multiplies LocSim by IdSim, so a wrong jersey zeroes the pair
    however good the localisation is. Fixing calibration alone must therefore
    not rescue a tracklet whose jersey is still wrong."""
    gt, pred = scenario(jersey_b=4, position_b=[52.0, 20.0])
    baseline = run(gt, pred, "baseline")
    assert run(gt, pred, "oracle_calibration") == pytest.approx(baseline)


def test_unmatched_prediction_keeps_its_real_attributes():
    """A false positive corresponds to no ground-truth player, so no oracle can
    exist for it; it must survive substitution untouched."""
    gt, pred = scenario()
    ghost = pred_row("ghost", 200, 200, team="left", jersey=99)
    for frame in pred:
        pred[frame] = pred[frame] + [dict(ghost)]
    lookups = ablation.matched_lookups(gt, pred, 0.50)
    configurations = ablation.build_configurations()
    candidate = ablation.apply_oracles(
        gt, pred, *lookups, configurations["oracle_all_attributes"]
    )
    ghosts = [row for row in candidate[0] if row["pred_identity_id"] == "players:ghost"]
    assert len(ghosts) == 1
    assert ghosts[0]["pred_jersey"] == 99


def test_detection_oracle_abstains_where_the_real_system_saw_nothing():
    """With detection made perfect but attributes left real, a player the
    detector missed has no predicted attributes at all. Those must stay absent
    rather than being quietly filled in from ground truth."""
    gt, pred = scenario()
    for frame in pred:
        pred[frame] = [row for row in pred[frame] if row["pred_identity_id"] != "players:b"]
    lookups = ablation.matched_lookups(gt, pred, 0.50)
    configurations = ablation.build_configurations()
    candidate = ablation.apply_oracles(
        gt, pred, *lookups, configurations["oracle_detection_tracking"]
    )
    missed = [row for row in candidate[0] if row["pred_identity_id"] == "gt:b"]
    assert len(missed) == 1
    assert missed[0]["pred_jersey"] is None
    assert missed[0]["pred_team"] is None


def test_attribute_rows_stay_on_the_real_detection_surface():
    """Regression guard. These rows once used the detection oracle, which
    injects ground-truth boxes the system never detected as attribute-less
    predictions -- turning one false negative into a false-negative/
    false-positive pair and inflating every attribute's apparent cost. They
    must stay on the real detection surface so that a module's cost is exactly
    oracle_all_attributes minus oracle_all_but_<module>."""
    configurations = ablation.build_configurations()
    for module in ablation.ATTRIBUTE_MODULES:
        assert not configurations[f"oracle_all_but_{module}"]["oracle_detection_tracking"]
        assert not configurations[f"oracle_{module}"]["oracle_detection_tracking"]
    assert not configurations["oracle_all_attributes"]["oracle_detection_tracking"]
    # Only these two legitimately replace detection with ground truth.
    assert configurations["full_oracle"]["oracle_detection_tracking"]
    assert configurations["oracle_detection_tracking"]["oracle_detection_tracking"]


def test_a_missed_player_does_not_inflate_attribute_costs():
    """With a player the detector never saw, the attribute rows must be
    unaffected by that miss: it is a detection failure, not a team or jersey
    failure, and must not be charged to them."""
    gt, pred = scenario(jersey_b=4)
    for frame in pred:
        pred[frame] = [row for row in pred[frame] if row["pred_identity_id"] != "players:b"]
    ceiling = run(gt, pred, "oracle_all_attributes")
    # 'b' is missing entirely, so no attribute oracle can reach it; every
    # attribute row must sit at the same ceiling rather than below it.
    for module in ablation.ATTRIBUTE_MODULES:
        assert run(gt, pred, f"oracle_all_but_{module}") == pytest.approx(ceiling)


def test_module_attribution_measures_costs_against_a_single_ceiling():
    results = {
        "baseline": {"gs_hota": 0.30, "delta_vs_baseline": 0.0},
        "oracle_all_attributes": {"gs_hota": 0.80, "delta_vs_baseline": 0.50},
        "full_oracle": {"gs_hota": 1.0, "delta_vs_baseline": 0.70},
    }
    for module in ablation.ATTRIBUTE_MODULES:
        results[f"oracle_{module}"] = {"gs_hota": 0.40, "delta_vs_baseline": 0.10}
        results[f"oracle_all_but_{module}"] = {"gs_hota": 0.70, "delta_vs_baseline": 0.40}

    attribution = ablation.module_attribution(results)
    for module in ablation.ATTRIBUTE_MODULES:
        assert attribution[module]["isolated_cost"] == pytest.approx(0.80 - 0.70)
        assert attribution[module]["marginal_gain"] == pytest.approx(0.10)
    # Detection and tracking are what separates the attribute ceiling from 1.
    assert attribution["detection_tracking"]["isolated_cost"] == pytest.approx(0.20)


def test_annotations_without_pitch_coordinates_are_dropped_with_their_matches():
    """GS-HOTA scores pitch positions, so an annotation with no bbox_pitch can
    never match anything. Keeping it costs a false negative *and* a false
    positive for whatever correctly detected it, penalising every system by a
    constant unrelated to its quality."""
    gt, pred = scenario()
    for frame in gt:
        gt[frame][1]["gt_position_pitch"] = None

    clean_gt, clean_pred, dropped = ablation.drop_unscoreable_subjects(gt, pred, 0.50)

    assert dropped["ground_truth_boxes"] == 3
    assert dropped["matched_predictions"] == 3
    assert all(row["gt_track_id"] != "b" for rows in clean_gt.values() for row in rows)
    assert all(
        row["pred_identity_id"] != "players:b"
        for rows in clean_pred.values()
        for row in rows
    )


def test_dropping_unscoreable_annotations_restores_the_exact_sanity_check():
    """Without the drop, a perfect oracle scores below 1 purely because of an
    annotation gap, which would silently disarm the harness's own correctness
    check. This is the regression that motivated the drop."""
    gt, pred = scenario()
    for frame in gt:
        gt[frame][1]["gt_position_pitch"] = None

    contaminated = run(gt, pred, "full_oracle")
    assert contaminated < 1.0

    clean_gt, clean_pred, _ = ablation.drop_unscoreable_subjects(gt, pred, 0.50)
    assert run(clean_gt, clean_pred, "full_oracle") == pytest.approx(1.0, abs=1e-9)


def test_nothing_is_dropped_when_every_annotation_is_scoreable():
    gt, pred = scenario()
    clean_gt, clean_pred, dropped = ablation.drop_unscoreable_subjects(gt, pred, 0.50)
    assert dropped == {"ground_truth_boxes": 0, "matched_predictions": 0}
    assert clean_gt == gt and clean_pred == pred


def test_an_undetected_unscoreable_annotation_drops_no_prediction():
    """Only the prediction that actually matched the gap is removed; unrelated
    detections must survive untouched."""
    gt, pred = scenario()
    for frame in gt:
        gt[frame][1]["gt_position_pitch"] = None
        pred[frame] = [row for row in pred[frame] if row["pred_identity_id"] != "players:b"]

    clean_gt, clean_pred, dropped = ablation.drop_unscoreable_subjects(gt, pred, 0.50)
    assert dropped["ground_truth_boxes"] == 3
    assert dropped["matched_predictions"] == 0
    assert all(len(rows) == 1 for rows in clean_pred.values())


def load_aggregator():
    path = REPO_ROOT / "scripts" / "aggregate_gs_hota_ablation.py"
    spec = importlib.util.spec_from_file_location("aggregate_gs_hota_ablation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregator_refuses_the_frozen_test_split(tmp_path):
    """The GSR test split is spendable once. An ablation is a diagnostic, so it
    must never consume it by accident -- only behind an explicit flag."""
    import subprocess

    aggregator = REPO_ROOT / "scripts" / "aggregate_gs_hota_ablation.py"
    manifest = tmp_path / "test_split.json"
    manifest.write_text(json.dumps({"split": "test", "sequences": []}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(aggregator),
         "--manifest", str(manifest),
         "--artifacts-root", str(tmp_path),
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "frozen" in (result.stdout + result.stderr).lower()


def test_aggregator_accepts_the_test_split_only_when_asked(tmp_path):
    import subprocess

    aggregator = REPO_ROOT / "scripts" / "aggregate_gs_hota_ablation.py"
    manifest = tmp_path / "test_split.json"
    manifest.write_text(json.dumps({"split": "test", "sequences": []}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(aggregator),
         "--manifest", str(manifest),
         "--artifacts-root", str(tmp_path),
         "--output-dir", str(tmp_path / "out"),
         "--allow-test"],
        capture_output=True, text=True,
    )
    # No sequences to ablate, so it gets past the guard and finishes.
    assert "frozen" not in (result.stdout + result.stderr).lower()


def test_reported_tables_exclude_the_unclean_counterfactual():
    """oracle_detection_tracking is not a valid counterfactual, so it must not
    reach a thesis table where it would read as 'perfect detection hurts'."""
    aggregator = load_aggregator()
    assert "oracle_detection_tracking" in aggregator.EXCLUDED_FROM_TABLES
    assert "oracle_detection_tracking" not in aggregator.CONFIGURATION_LABELS

    def distribution(mean):
        return {"n": 12, "mean": mean, "median": mean,
                "ci95_low": mean - 0.05, "ci95_high": mean + 0.05}

    aggregate = {
        "sequence_count": 12, "split": "valid", "tau_metres": 5.0,
        "arms": {"a": "SAR primary"},
        "configurations": {
            name: {"a": distribution(0.3)} for name in aggregator.CONFIGURATION_LABELS
        },
        "attribution": {
            "jersey": {"a": {"isolated_cost": distribution(0.25),
                             "marginal_gain": distribution(0.1)}},
        },
    }
    latex = aggregator.latex_tables(aggregate)
    assert "oracle_detection_tracking" not in latex
    assert latex.count("\\begin{table}") == 2
    assert "\\toprule" in latex and "\\bottomrule" in latex


def test_sanity_check_reports_failure_when_full_oracle_is_wrong():
    assert ablation.check_full_oracle({"full_oracle": {"gs_hota": 1.0}})["passed"]
    assert not ablation.check_full_oracle({"full_oracle": {"gs_hota": 0.93}})["passed"]
    assert not ablation.check_full_oracle({})["passed"]
