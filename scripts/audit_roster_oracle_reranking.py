#!/usr/bin/env python3
"""Measure the theoretical ceiling of roster-constrained re-ranking for the
region-CTC recognizer, using an oracle roster built from GSR ground truth.

Motivation: the CTC sometimes confuses visually similar digits (documented
case: true number 95 read as 55) with high confidence, while the correct
number is present lower in its own top-5. The region-CTC audit
(`JerseyRegionCTCAuditor`) never consults the roster today -- it always takes
the raw top-1. On videos with a real roster (Int-Ata, ...) that is a concrete
gap to close. GSR has no real roster in production and this script must never
be read as "GSR now uses a roster" -- it only measures, using GT jersey
numbers as an oracle roster, how much headroom roster-constrained re-ranking
could realistically capture. That number is what decides whether building the
real thing for roster-equipped videos is worth it.

Oracle roster construction: for a given track's (sequence, team) pair, the
roster is the set of GT jersey numbers observed on every OTHER track of the
same team in the same sequence (the track itself is excluded to avoid trivial
leakage). This uses only information already in the GSR annotations, and nly
to bound the opportunity, not to declare a result usable on GSR itself.

Inputs:
  --raw-scores   : output of `evaluate_jersey_number_region_ctc_ocr_run.py
                   --dump-raw-scores ...` (per-frame CTC candidate log-probs)
  --team-source-csv : a predictions.csv from the underlying OCR run that has
                   `team` and `gt_jersey_number` columns (e.g.
                   `.../finetuned_sar/predictions.csv`)

Recomputes `aggregate_frames` locally (pure Python, no torch) from the raw
per-frame scores -- guaranteed to reproduce the original run's predictions
exactly, so this is a pure counterfactual re-ranking of the same evidence,
not a new inference pass.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft.features.jersey_number_ctc import aggregate_frames  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-scores", required=True)
    parser.add_argument("--team-source-csv", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    raw = json.loads(Path(args.raw_scores).read_text(encoding="utf-8"))
    team_rows = read_csv(args.team_source_csv)

    # A real roster lists every player on the team, including the one this
    # track belongs to -- it is an independent list, not derived from any
    # track's own prediction. So the track's own GT number is legitimately
    # part of its team's oracle roster here; excluding it would exclude
    # exactly the answer a real roster would provide (jersey numbers are
    # unique within a team, so no other track can ever "vouch" for this
    # track's own number).
    team_by_key = {}
    roster_members = defaultdict(set)
    for row in team_rows:
        key = (row["sequence"], row["gt_track_id"])
        team = row.get("team")
        gt = to_int(row.get("gt_jersey_number"))
        team_by_key[key] = team
        if team is not None and gt is not None:
            roster_members[(row["sequence"], team)].add(gt)

    rows = []
    missing_team = 0
    for track_key, data in raw.items():
        sequence = data["sequence"]
        gt_track_id = str(data["gt_track_id"])
        key = (sequence, gt_track_id)
        truth = data["gt_jersey_number"]
        team = team_by_key.get(key)
        if team is None:
            missing_team += 1
            continue

        result = aggregate_frames(data["scores"], data["weights"])
        original_pred = result["prediction"]
        if original_pred is None:
            continue  # unassigned originally; not part of accuracy_assigned, skip

        ranking = list(result["scores"].items())[: args.top_k]
        roster = roster_members[(sequence, team)]
        in_roster_ranked = [candidate for candidate, _score in ranking if int(candidate) in roster]
        reranked_pred = int(in_roster_ranked[0]) if in_roster_ranked else original_pred

        original_correct = original_pred == truth
        reranked_correct = reranked_pred == truth
        rows.append({
            "sequence": sequence,
            "gt_track_id": gt_track_id,
            "team": team,
            "gt_jersey_number": truth,
            "original_pred": original_pred,
            "reranked_pred": reranked_pred,
            "original_correct": original_correct,
            "reranked_correct": reranked_correct,
            "changed": reranked_pred != original_pred,
            "roster_size": len(roster),
            "gt_in_top_k": str(truth) in dict(ranking),
            "top_k_candidates": [c for c, _ in ranking],
        })

    transitions = {
        "correct_to_correct": sum(1 for r in rows if r["original_correct"] and r["reranked_correct"]),
        "correct_to_wrong": sum(1 for r in rows if r["original_correct"] and not r["reranked_correct"]),
        "wrong_to_correct": sum(1 for r in rows if not r["original_correct"] and r["reranked_correct"]),
        "wrong_to_wrong": sum(1 for r in rows if not r["original_correct"] and not r["reranked_correct"]),
    }
    summary = {
        "assigned_tracks_evaluated": len(rows),
        "missing_team_skipped": missing_team,
        "original_accuracy_assigned": ratio(sum(r["original_correct"] for r in rows), len(rows)),
        "reranked_accuracy_assigned": ratio(sum(r["reranked_correct"] for r in rows), len(rows)),
        "transitions": transitions,
        "net_gain": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
        "changed_rows": sum(1 for r in rows if r["changed"]),
        "top_k": args.top_k,
        "note": (
            "Oracle roster derived from GSR ground truth, for ceiling estimation only. "
            "correct_to_wrong must stay at or near zero for this to be a safe mechanism; "
            "any regression here means the roster constraint can override a genuinely "
            "correct read and would need a confidence gate before ever being used on real "
            "roster-equipped videos."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "roster_oracle_reranking.csv", rows)
    (output / "roster_oracle_reranking_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\ncsv={output / 'roster_oracle_reranking.csv'}")
    print(f"summary={output / 'roster_oracle_reranking_summary.json'}")


def to_int(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def ratio(a, b):
    return a / b if b else 0.0


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )


if __name__ == "__main__":
    main()
