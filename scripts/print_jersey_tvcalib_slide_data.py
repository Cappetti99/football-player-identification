#!/usr/bin/env python3
"""Print every number on the TVCalib slide, sourced from four real artifacts.

No plot: this slide is two text tables, so the script only prints numbers to
copy into the LaTeX tables, in the same order they appear on the slide.
Nothing here recomputes a statistic -- every value is read verbatim from an
aggregate.json or comparison.json produced upstream.

    python scripts/print_jersey_tvcalib_slide_data.py \
        --auto-calib evaluation_outputs/calibration_link_audit/gsr_valid_pilot12_conf012_v1_aggregate/aggregate.json \
        --tvcalib-calib evaluation_outputs/calibration_link_audit/gsr_valid_pilot12_tvcalib_v1_aggregate/aggregate.json \
        --rigid-comparison evaluation_outputs/detection_tracking/conf012_vs_tvcalib_pitch14_apply_v1/comparison.json \
        --soft-comparison evaluation_outputs/detection_tracking/conf012_vs_tvcalib_pitch_rerank_w010_v1/comparison.json \
        --candidate-ranking evaluation_outputs/detection_tracking/gsr_valid_pilot12_conf012_tvcalib_candidate_ranking_v1/summary.json
"""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto-calib", required=True)
    parser.add_argument("--tvcalib-calib", required=True)
    parser.add_argument("--rigid-comparison", required=True)
    parser.add_argument("--soft-comparison", required=True)
    parser.add_argument("--candidate-ranking", required=True)
    args = parser.parse_args()

    auto = load(args.auto_calib)
    tvcalib = load(args.tvcalib_calib)
    rigid = load(args.rigid_comparison)["metrics"]["display_hota"]
    soft = load(args.soft_comparison)["metrics"]["display_hota"]
    ranking = load(args.candidate_ranking)
    sweep_zero = next(row for row in ranking["weight_sweep"] if row["pitch_weight"] == 0.0)

    print("=== tabella 1: accuratezza geometrica ===")
    print(f"{'':24s}{'mediana':>10s}{'P90':>10s}")
    print(
        f"{'calibrazione automatica':24s}"
        f"{auto['mean_sequence_pitch_median_error_m']:9.2f}m"
        f"{auto['mean_sequence_pitch_p90_error_m']:9.2f}m"
    )
    print(
        f"{'TVCalib':24s}"
        f"{tvcalib['mean_sequence_pitch_median_error_m']:9.2f}m"
        f"{tvcalib['mean_sequence_pitch_p90_error_m']:9.2f}m"
    )

    print("\n=== tabella 2: ablation end-to-end (contro confidence=0.12) ===")
    print(f"{'':18s}{'deltaHOTA':>12s}{'seq. migliorate':>18s}")
    print(
        f"{'vincolo rigido':18s}{rigid['mean_improvement']:+12.5f}"
        f"{rigid['improved_sequences']:>13d} su {rigid['n']}"
    )
    print(
        f"{'costo morbido':18s}{soft['mean_improvement']:+12.5f}"
        f"{soft['improved_sequences']:>13d} su {soft['n']}"
    )

    print("\n=== candidati mancanti ===")
    evaluable = sweep_zero["evaluable_targets"]
    available = sweep_zero["correct_candidate_available"]
    print(f"candidato corretto assente: {evaluable - available} su {evaluable}")
    print(f"(available={available}, oracle_coverage={sweep_zero['oracle_coverage']})")


if __name__ == "__main__":
    main()
