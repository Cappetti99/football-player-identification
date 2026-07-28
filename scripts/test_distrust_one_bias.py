#!/usr/bin/env python3
"""Offline test: does distrusting a thin-margin "1" verdict improve jersey OCR
accuracy against real SoccerNet-GSR ground truth?

Motivation: on Usa-Bel, 8/23 tracklets had their winning jersey_number == 1,
mostly with modest winner_margin against varying runner-up candidates (7, 4,
3, 6...) -- consistent with the already-documented backend bias toward "1"
as a low-confidence default (see confusion.csv: many true numbers ->1).

This re-simulates each tracklet's *already computed* vote_numbers() output
(jersey_number, winner_margin, candidates) from ocr_diagnostics.json -- no
new inference -- under two candidate policies:

  - demote: if winner == 1 and winner_margin < threshold, treat as
    unassigned (drop the guess rather than keep an unreliable "1").
  - promote_runner_up: same trigger, but use the second-ranked candidate
    instead of dropping -- tests whether the runner-up is usually the real
    answer (as in a number_band truncation, e.g. "10" -> "1").

Run from ~/FT:
    python3 scripts/test_distrust_one_bias.py --run-dir evaluation_outputs/gsr_jersey_ocr_mmocr_50s/gsr_val_mmocr_easyocr_50s_5000t
    python3 scripts/test_distrust_one_bias.py --run-dir evaluation_outputs/gsr_jersey_ocr_mmocr_10s_v2/gsr_val_mmocr_easyocr_10s_1000t
"""
import argparse
import csv
import json
from pathlib import Path

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


def evaluate(tracks, gt_by_id, min_votes, mode, threshold):
    evaluated = assigned = correct = 0
    demoted = 0
    for eval_id, gt in gt_by_id.items():
        track = tracks.get(eval_id)
        if track is None:
            continue
        evaluated += 1
        voted = track.get("voted")
        if not voted or int(voted.get("votes", 0) or 0) < min_votes:
            continue
        number = voted.get("jersey_number")
        margin = float(voted.get("winner_margin", 0.0) or 0.0)
        candidates = voted.get("candidates") or []
        if number is not None and int(number) == 1 and margin < threshold:
            demoted += 1
            if mode == "demote":
                continue
            if mode == "promote_runner_up":
                runner_up = candidates[1] if len(candidates) > 1 else None
                if runner_up is None:
                    continue
                number = runner_up.get("jersey_number")
        if number is None:
            continue
        assigned += 1
        if str(number) == str(gt):
            correct += 1
    return {
        "evaluable_tracks": evaluated,
        "assigned": assigned,
        "correct": correct,
        "coverage": assigned / evaluated if evaluated else None,
        "accuracy_assigned": (correct / assigned) if assigned else None,
        "accuracy_all": (correct / evaluated) if evaluated else None,
        "demoted_thin_margin_ones": demoted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    root = Path(args.run_dir)
    diagnostics = json.loads((root / "ocr_diagnostics.json").read_text())
    tracks = diagnostics.get("tracklets") or {}
    min_votes = int(diagnostics.get("min_votes", 2))

    gt_by_id = {}
    with open(root / "predictions.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gt_by_id[row["eval_track_id"]] = row["gt_jersey_number"]

    print(f"tracklets: {len(tracks)}, gt rows: {len(gt_by_id)}, min_votes: {min_votes}")
    print()

    baseline = evaluate(tracks, gt_by_id, min_votes, mode="none", threshold=-1.0)
    print("baseline:", json.dumps(baseline))
    print()

    for mode in ("demote", "promote_runner_up"):
        for threshold in THRESHOLDS:
            result = evaluate(tracks, gt_by_id, min_votes, mode=mode, threshold=threshold)
            print(f"{mode} threshold={threshold}: {json.dumps(result)}")
        print()


if __name__ == "__main__":
    main()
