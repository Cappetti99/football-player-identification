#!/usr/bin/env python3
import argparse
from pathlib import Path

from ft.evaluation.identity_benchmark import (
    adjudicate,
    canonical_hash,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)


FIELDS = [
    "item_id", "item_type", "video_id", "split", "reviewer",
    "second_review_required", "annotation_status", "gt_player_id",
    "gt_team_id", "gt_jersey_number", "jersey_visibility", "pair_label",
    "uncertainty_reason", "notes",
]


def main():
    parser = argparse.ArgumentParser(description="Measure reviewer agreement and finalize benchmark ground truth.")
    parser.add_argument("--benchmark-dir", default="evaluation_outputs/identity_benchmark_v1")
    parser.add_argument("--reviewer-a")
    parser.add_argument("--reviewer-b")
    parser.add_argument("--adjudication")
    parser.add_argument("--output-dir", default="evaluation/identity_benchmark_v1")
    args = parser.parse_args()

    benchmark = Path(args.benchmark_dir)
    reviewer_a_path = Path(args.reviewer_a or benchmark / "annotations" / "reviewer_a.csv")
    reviewer_b_path = Path(args.reviewer_b or benchmark / "annotations" / "reviewer_b.csv")
    adjudication_path = Path(args.adjudication or benchmark / "annotations" / "adjudication.csv")
    manifest_path = benchmark / "benchmark_manifest.json"
    manifest = read_json(manifest_path)
    reviewer_a = read_csv(reviewer_a_path)
    reviewer_b = read_csv(reviewer_b_path)
    adjudication_rows = read_csv(adjudication_path) if adjudication_path.is_file() and adjudication_path.stat().st_size else []
    final, disagreements, agreement = adjudicate(reviewer_a, reviewer_b, adjudication_rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    disagreement_rows = []
    for item in disagreements:
        left, right = item["reviewer_a"], item["reviewer_b"]
        disagreement_rows.append({
            **{key: left.get(key, "") for key in FIELDS},
            "reviewer": "ADJUDICATOR",
            "reviewer_a_signature": str(left),
            "reviewer_b_signature": str(right),
        })
    write_csv(disagreement_rows, output / "adjudication_required.csv", FIELDS)
    write_json(agreement, output / "agreement_report.json")

    unresolved = [row for row in disagreements if row["item_id"] not in {item["item_id"] for item in adjudication_rows}]
    if unresolved:
        print(f"unresolved_disagreements={len(unresolved)} template={output / 'adjudication_required.csv'}")
        raise SystemExit(2)
    expected = len(manifest.get("identity_units", [])) + len(manifest.get("pairs", []))
    if len(final) != expected:
        raise SystemExit(f"ground truth incomplete: finalized={len(final)} expected={expected}")
    write_csv(final, output / "ground_truth.csv", FIELDS)
    gt_meta = {
        "benchmark_sha256": manifest["benchmark_sha256"],
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "reviewer_a_sha256": sha256_file(reviewer_a_path),
        "reviewer_b_sha256": sha256_file(reviewer_b_path),
        "adjudication_sha256": sha256_file(adjudication_path) if adjudication_path.is_file() else None,
        "ground_truth_sha256": canonical_hash(final),
        "items": len(final),
        "agreement": agreement,
    }
    write_json(gt_meta, output / "ground_truth_manifest.json")
    print(f"ground_truth={output / 'ground_truth.csv'} items={len(final)}")


if __name__ == "__main__":
    main()
