#!/usr/bin/env python3
"""Oracle ablation of the FT architecture, measured with GS-HOTA.

Reproduces the ablation methodology of Somers et al. (CVPR24, Table 2) on our
own pipeline: measure GS-HOTA while replacing individual modules with a
ground-truth oracle, to attribute the end-to-end error to specific stages.

Two complementary directions are reported for every module:

* ``oracle_<module>``     -- baseline plus that one module made perfect.
  How much would we gain by fixing this module alone? (marginal benefit)
* ``oracle_all_but_<m>``  -- every other attribute made perfect, this one real,
  on the real detection/tracking surface. How much does this module cost when
  nothing else is broken? (isolated cost) The cost of a module is exactly
  ``oracle_all_attributes - oracle_all_but_<module>``.

A module that is cheap on both counts is not a bottleneck; one that is
expensive on both is. The two can disagree, which is itself informative:
a module can be individually harmless yet dominate once the others are fixed.

Modules covered, and the GS-HOTA component each controls:

    calibration -> pitch position   -> LocSim (Gaussian on metres, tau=5m)
    team        -> team attribute   -> IdSim
    role        -> role attribute   -> IdSim
    jersey      -> jersey attribute -> IdSim
    detection   -> which subjects exist       -> DetA (FP/FN)
    tracking    -> how they link across time  -> AssA

Scope and honest limitations:

* This is an **evaluation-surface** ablation, not a re-run. Oracle values are
  substituted after inference, so downstream interactions inside the pipeline
  are not reproduced: e.g. a perfect team label here does not also change what
  the roster-aware filter would have done to the jersey number during the run.
  It answers "how much does this module's error cost at the output", not
  "what would the system do if this module were perfect". The latter needs GT
  injected into the pipeline itself.
* Attribute oracles apply only to detections the real system actually
  produced and that matched a ground-truth box. A player the detector missed
  cannot be repaired by a perfect team classifier, and is left as a false
  negative -- which is the point of separating the detection row.
* The ``oracle_detection_tracking`` row is the one figure here that is **not**
  a clean counterfactual, and it can even score below baseline. Injecting a
  ground-truth box the system never detected gives a subject whose attributes
  were never computed; leaving them absent turns one false negative into a
  false-negative/false-positive pair. Producing real attributes for those
  boxes would mean running the recognizers on crops that were never extracted,
  which no offline substitution can do. Read the detection/tracking cost off
  ``oracle_all_attributes`` instead, which is well defined: real detection and
  tracking, every attribute correct.
* GS-HOTA scores position + team + role + jersey. It does **not** score the
  final named-player assignment (roster filter, Hungarian); that stage is
  measured by ft/evaluation/identity_benchmark.py instead.
* Ground-truth annotations with no ``bbox_pitch`` are removed before scoring,
  together with whatever detection matched them; see
  ``drop_unscoreable_subjects``. This makes the reported baseline differ very
  slightly from the headline GS-HOTA produced by scripts/evaluate_ft_gsr.py,
  which keeps them: at most 0.9% on the worst SoccerNet-GSR valid sequence and
  under 0.1% averaged over the 12-sequence pilot, far below the effects being
  measured here.

    python3 scripts/ablate_gs_hota_oracles.py \\
        --labels /path/to/SNGS-082/Labels-GameState.json \\
        --tracklets artifacts/.../SNGS-082_tracklets.csv \\
        --output-dir evaluation_outputs/gs_hota_ablation/SNGS-082
"""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ft.evaluation.gsr_detection_tracking import (  # noqa: E402
    evaluate_frames,
    gs_hota_summary,
)


ATTRIBUTE_MODULES = ("calibration", "team", "role", "jersey")
# Which prediction field each attribute module owns, and the ground-truth
# field the oracle copies from.
ORACLE_FIELDS = {
    "calibration": ("pred_position_pitch", "gt_position_pitch"),
    "team": ("pred_team", "gt_team"),
    "role": ("pred_role", "gt_role"),
    "jersey": ("pred_jersey", "gt_jersey"),
}


def load_evaluator():
    """Reuse evaluate_ft_gsr's own loaders so the ablation reads the artifacts
    exactly as the real measurement does."""
    path = REPO_ROOT / "scripts" / "evaluate_ft_gsr.py"
    spec = importlib.util.spec_from_file_location("evaluate_ft_gsr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--tau-metres", type=float, default=5.0)
    parser.add_argument(
        "--gt-pitch-coordinate-system",
        choices=["soccernet_centered", "ft"],
        default="soccernet_centered",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--gt-roles", nargs="+", default=["goalkeeper", "player", "referee"]
    )
    args = parser.parse_args()

    payload = run_ablation(
        labels=args.labels,
        tracklets=args.tracklets,
        iou_threshold=args.iou_threshold,
        tau_metres=args.tau_metres,
        gt_pitch_coordinate_system=args.gt_pitch_coordinate_system,
        max_frames=args.max_frames,
        gt_roles=args.gt_roles,
    )
    results = payload["configurations"]
    attribution = payload["module_attribution"]
    sanity = payload["sanity_check"]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gs_hota_ablation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "gs_hota_ablation.csv", results)
    print_table(results, attribution, sanity)
    print(f"\nwritten: {output / 'gs_hota_ablation.json'}")
    return 0 if sanity["passed"] else 1


def run_ablation(
    labels,
    tracklets,
    iou_threshold=0.50,
    tau_metres=5.0,
    gt_pitch_coordinate_system="soccernet_centered",
    max_frames=None,
    gt_roles=("goalkeeper", "player", "referee"),
):
    """Ablate one sequence and return the full result payload.

    Exposed separately from main() so the multi-sequence aggregator can call it
    directly, guaranteeing both use identical loading, matching and oracle
    logic rather than drifting apart.
    """
    evaluator = load_evaluator()
    gt, metadata = evaluator.load_ground_truth(
        Path(labels),
        allowed_roles=list(gt_roles),
        pitch_coordinate_system=gt_pitch_coordinate_system,
    )
    pred = evaluator.load_predictions(Path(tracklets))
    if max_frames is not None:
        gt = {f: rows for f, rows in gt.items() if f < max_frames}
        pred = {f: rows for f, rows in pred.items() if f < max_frames}

    gt, pred, unscoreable = drop_unscoreable_subjects(gt, pred, iou_threshold)

    # Same left/right team-id reconciliation the real evaluator applies before
    # GS-HOTA: GT teams are "left"/"right", predictions carry FT's numeric
    # team_id. Oracles then operate in the same (mapped) space.
    matches = evaluate_frames(gt, pred, iou_threshold)["matches"]
    team_mapping = evaluator.choose_team_mapping(matches)
    pred = {
        frame: [
            {**row, "pred_team": team_mapping.get(row["pred_team"], row["pred_team"])}
            for row in rows
        ]
        for frame, rows in pred.items()
    }

    gt_by_matched_pred, matched_pred_by_gt = matched_lookups(gt, pred, iou_threshold)

    results = {}
    for name, configuration in build_configurations().items():
        candidate = apply_oracles(
            gt, pred, gt_by_matched_pred, matched_pred_by_gt, configuration
        )
        summary = gs_hota_summary(gt, candidate, tau=tau_metres)
        results[name] = {
            "oracle_modules": sorted(configuration["oracle_attributes"]),
            "oracle_detection_tracking": configuration["oracle_detection_tracking"],
            "gs_hota": summary["gs_hota"],
            "gs_deta": summary["gs_deta"],
            "gs_assa": summary["gs_assa"],
            "gs_loca": summary["loca"],
        }

    baseline = results["baseline"]["gs_hota"]
    for row in results.values():
        row["delta_vs_baseline"] = row["gs_hota"] - baseline

    return {
        "sequence": metadata.get("name") or Path(labels).parent.name,
        "iou_threshold": iou_threshold,
        "tau_metres": tau_metres,
        "frames": len(gt),
        "unscoreable_subjects": unscoreable,
        "configurations": results,
        "module_attribution": module_attribution(results),
        "sanity_check": check_full_oracle(results),
        "method": (
            "evaluation-surface oracle substitution on IoU-matched pairs; "
            "not a pipeline re-run, so downstream interactions are not modelled"
        ),
    }


def drop_unscoreable_subjects(gt, pred, iou_threshold):
    """Remove ground-truth boxes that GS-HOTA cannot score, and their matches.

    GS-HOTA's similarity is defined on pitch coordinates, so an annotation
    without ``bbox_pitch`` has no position and can never match anything --
    not even a perfect oracle. Keeping such a row costs two errors, a false
    negative for the annotation and a false positive for whatever correctly
    detected it, so it penalises every system by a constant that has nothing
    to do with system quality. Measured on SoccerNet-GSR valid, the resulting
    deficit is exactly ``2 * missing / total`` (0.43% of annotations in
    SNGS-025 depressed a perfect-oracle score to 0.9914).

    The matched prediction is dropped alongside the annotation: dropping only
    the ground-truth row would turn a correct detection into a false positive,
    punishing the system for an annotation gap it had no part in.
    """
    missing = {
        (frame, row["gt_track_id"])
        for frame, rows in gt.items()
        for row in rows
        if row.get("gt_position_pitch") is None
    }
    if not missing:
        return gt, pred, {"ground_truth_boxes": 0, "matched_predictions": 0}

    evaluation = evaluate_frames(gt, pred, iou_threshold)
    orphaned = {
        (match["frame"], match["pred_identity_id"])
        for match in evaluation["matches"]
        if (match["frame"], match["gt_track_id"]) in missing
    }
    clean_gt = {
        frame: [row for row in rows if (frame, row["gt_track_id"]) not in missing]
        for frame, rows in gt.items()
    }
    clean_pred = {
        frame: [row for row in rows if (frame, row["pred_identity_id"]) not in orphaned]
        for frame, rows in pred.items()
    }
    return clean_gt, clean_pred, {
        "ground_truth_boxes": len(missing),
        "matched_predictions": len(orphaned),
        "reason": "no bbox_pitch in the annotation; GS-HOTA similarity is undefined",
    }


def build_configurations():
    """baseline, one-oracle-added, one-module-left-real, and the two extremes."""
    configurations = {
        "baseline": spec(set(), False),
    }
    for module in ATTRIBUTE_MODULES:
        configurations[f"oracle_{module}"] = spec({module}, False)
    configurations["oracle_detection_tracking"] = spec(set(), True)

    # Deliberately on the REAL detection/tracking surface, not the oracle one.
    # With detection made oracle, ground-truth boxes the system never detected
    # enter as predictions carrying no attributes, which turns a single false
    # negative into a false-negative/false-positive pair and penalises those
    # players twice. Keeping detection real makes every attribute row directly
    # comparable to oracle_all_attributes below, so the cost of a module is
    # exactly oracle_all_attributes - oracle_all_but_<module>.
    for module in ATTRIBUTE_MODULES:
        others = set(ATTRIBUTE_MODULES) - {module}
        configurations[f"oracle_all_but_{module}"] = spec(others, False)
    # Everything perfect except detection/tracking: the cost of detection and
    # association alone, with every attribute correct.
    configurations["oracle_all_attributes"] = spec(set(ATTRIBUTE_MODULES), False)
    # Sanity: every module oracle must reproduce the ground truth exactly.
    configurations["full_oracle"] = spec(set(ATTRIBUTE_MODULES), True)
    return configurations


def spec(oracle_attributes, oracle_detection_tracking):
    return {
        "oracle_attributes": set(oracle_attributes),
        "oracle_detection_tracking": bool(oracle_detection_tracking),
    }


def matched_lookups(gt, pred, threshold):
    """Map matched prediction <-> ground-truth rows, keyed per frame.

    Keys are (frame, pred_identity_id) and (frame, gt_track_id). Both are
    one-box-per-frame by construction; a violation would silently corrupt the
    oracle substitution, so it is checked rather than assumed.
    """
    evaluation = evaluate_frames(gt, pred, threshold)
    gt_by_pred = {}
    pred_by_gt = {}
    for match in evaluation["matches"]:
        frame = match["frame"]
        pred_key = (frame, match["pred_identity_id"])
        gt_key = (frame, match["gt_track_id"])
        if pred_key in gt_by_pred or gt_key in pred_by_gt:
            raise SystemExit(
                f"duplicate match key at frame {frame}: the same identity appears "
                "twice in one frame, so oracle substitution would be ambiguous"
            )
        gt_by_pred[pred_key] = match
        pred_by_gt[gt_key] = match
    return gt_by_pred, pred_by_gt


def apply_oracles(gt, pred, gt_by_matched_pred, matched_pred_by_gt, spec):
    if spec["oracle_detection_tracking"]:
        return oracle_detection_tracking_rows(gt, matched_pred_by_gt, spec)
    return oracle_attribute_rows(pred, gt_by_matched_pred, spec)


def oracle_attribute_rows(pred, gt_by_matched_pred, spec):
    """Real detections and tracks; selected attributes copied from ground truth.

    Unmatched predictions (false positives) keep their real attributes: no
    oracle exists for a detection that corresponds to no real player.
    """
    output = {}
    for frame, rows in pred.items():
        new_rows = []
        for row in rows:
            row = copy.copy(row)
            match = gt_by_matched_pred.get((frame, row["pred_identity_id"]))
            if match is not None:
                for module in spec["oracle_attributes"]:
                    pred_field, gt_field = ORACLE_FIELDS[module]
                    row[pred_field] = match[gt_field]
            new_rows.append(row)
        output[frame] = new_rows
    return output


def oracle_detection_tracking_rows(gt, matched_pred_by_gt, spec):
    """Ground-truth boxes and identities; attributes from the real system.

    A ground-truth box the real system never detected has no predicted
    attributes at all, so its attributes stay None (an abstention) unless that
    module is itself under oracle. That keeps the detector's misses visible as
    identity failures instead of silently repairing them.
    """
    output = {}
    for frame, rows in gt.items():
        new_rows = []
        for gt_row in rows:
            match = matched_pred_by_gt.get((frame, gt_row["gt_track_id"]))
            identity = f"gt:{gt_row['gt_track_id']}"
            row = {
                "pred_identity_id": identity,
                "raw_pred_identity_id": identity,
                "bbox": gt_row["bbox"],
            }
            for module in ATTRIBUTE_MODULES:
                pred_field, gt_field = ORACLE_FIELDS[module]
                if module in spec["oracle_attributes"]:
                    row[pred_field] = gt_row[gt_field]
                else:
                    row[pred_field] = match[pred_field] if match else None
            new_rows.append(row)
        output[frame] = new_rows
    return output


def module_attribution(results):
    """Isolated cost of each module, on one consistent surface.

    Attribute modules are measured against ``oracle_all_attributes`` (real
    detection and tracking, every attribute perfect); detection and tracking
    together are measured against ``full_oracle``. Costs are not additive:
    GS-HOTA multiplies localisation by identity similarity, so two modules
    failing on the same subject overlap rather than sum.
    """
    ceiling = results["oracle_all_attributes"]["gs_hota"]
    attribution = {}
    for module in ATTRIBUTE_MODULES:
        attribution[module] = {
            "isolated_cost": ceiling - results[f"oracle_all_but_{module}"]["gs_hota"],
            "marginal_gain": results[f"oracle_{module}"]["delta_vs_baseline"],
            "measured_against": "oracle_all_attributes",
        }
    attribution["detection_tracking"] = {
        "isolated_cost": results["full_oracle"]["gs_hota"] - ceiling,
        "marginal_gain": None,
        "measured_against": "full_oracle",
        "note": (
            "oracle_detection_tracking is not a clean counterfactual and is "
            "excluded here; see the module docstring"
        ),
    }
    return attribution


def check_full_oracle(results, tolerance=1e-9):
    """With every module oracle the predictions are the ground truth, so
    GS-HOTA must be exactly 1. Anything else means the harness itself is
    wrong, and no other row in the table can be trusted."""
    value = results.get("full_oracle", {}).get("gs_hota")
    passed = value is not None and abs(value - 1.0) <= tolerance
    return {
        "name": "full_oracle == 1.0",
        "value": value,
        "passed": bool(passed),
        "detail": (
            "harness reproduces ground truth when every module is oracle"
            if passed
            else "FAILED: oracle substitution does not reproduce ground truth; "
            "the ablation numbers are not trustworthy until this is explained"
        ),
    }


def write_csv(path, results):
    import csv

    fields = [
        "configuration", "gs_hota", "delta_vs_baseline",
        "gs_deta", "gs_assa", "gs_loca",
        "oracle_modules", "oracle_detection_tracking",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in results.items():
            writer.writerow({
                "configuration": name,
                "gs_hota": row["gs_hota"],
                "delta_vs_baseline": row["delta_vs_baseline"],
                "gs_deta": row["gs_deta"],
                "gs_assa": row["gs_assa"],
                "gs_loca": row["gs_loca"],
                "oracle_modules": ";".join(row["oracle_modules"]),
                "oracle_detection_tracking": row["oracle_detection_tracking"],
            })


def print_table(results, attribution, sanity):
    print(f"{'configuration':<32}{'GS-HOTA':>9}{'delta':>9}{'GS-DetA':>9}{'GS-AssA':>9}")
    print("-" * 68)
    for name, row in results.items():
        flag = "  (*)" if name == "oracle_detection_tracking" else ""
        print(
            f"{name:<32}{row['gs_hota']:>9.4f}{row['delta_vs_baseline']:>+9.4f}"
            f"{row['gs_deta']:>9.4f}{row['gs_assa']:>9.4f}{flag}"
        )
    print("-" * 68)
    print("(*) not a clean counterfactual; read detection/tracking cost from")
    print("    oracle_all_attributes instead. See the module docstring.")
    print()
    print(f"{'module':<24}{'isolated cost':>15}{'marginal gain':>15}")
    print("-" * 54)
    for module, row in sorted(
        attribution.items(), key=lambda item: -item[1]["isolated_cost"]
    ):
        gain = row["marginal_gain"]
        gain_text = f"{gain:>+15.4f}" if gain is not None else f"{'n/a':>15}"
        print(f"{module:<24}{row['isolated_cost']:>15.4f}{gain_text}")
    print("-" * 54)
    print("costs are not additive: GS-HOTA multiplies LocSim by IdSim, so")
    print("failures on the same subject overlap rather than sum.")
    print()
    status = "PASS" if sanity["passed"] else "FAIL"
    print(f"sanity {status}: {sanity['name']} -> {sanity['value']}")
    if not sanity["passed"]:
        print(f"  {sanity['detail']}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
