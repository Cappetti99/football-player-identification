#!/usr/bin/env python3
"""Compare two frozen jersey-decision runs against offline GSR frame matches."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {None, "", "None", "null", "-1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--artifact-root", default="artifacts/costume-video")
    parser.add_argument("--matches-root", required=True)
    parser.add_argument("--baseline-template", required=True)
    parser.add_argument("--candidate-template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    matches_root = Path(args.matches_root)
    summaries = []
    deltas = []

    for sequence in args.sequences:
        matches = normalize_matches(read_csv(matches_root / sequence / "frame_matches.csv"))
        canonical, gt_jersey, gt_role = gt_indices(matches)
        variants = {
            "baseline": load_decisions(
                artifact_root, args.baseline_template.format(sequence=sequence), sequence
            ),
            "candidate": load_decisions(
                artifact_root, args.candidate_template.format(sequence=sequence), sequence
            ),
        }
        for label, predictions in variants.items():
            summary = summarize(
                sequence, label, predictions, matches, canonical, gt_jersey, gt_role
            )
            summaries.append(summary)
            print(sequence, label, summary)
        deltas.extend(compare_decisions(sequence, variants, gt_jersey))

    pooled = {
        label: pool([row for row in summaries if row["variant"] == label])
        for label in ("baseline", "candidate")
    }
    result = {
        "sequences": args.sequences,
        "pooled": pooled,
        "per_sequence": summaries,
        "deltas": deltas,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nPOOLED")
    for label, row in pooled.items():
        print(label, row)
    print("\nDELTAS")
    for row in deltas:
        print(row)
    print("\nWROTE", output)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_matches(rows):
    output = []
    for row in rows:
        output.append({
            "display": text(
                row.get("pred_display_track_id")
                or row.get("display_track_id")
                or row.get("pred_track_id")
            ),
            "gt_track": text(row.get("gt_track_id")),
            "visible_jersey": integer(row.get("gt_jersey")),
            "role": text(row.get("gt_role") or row.get("gt_class")),
        })
    return output


def gt_indices(matches):
    jerseys_by_gt = defaultdict(list)
    gt_tracks_by_display = defaultdict(list)
    roles_by_display = defaultdict(list)
    for row in matches:
        if row["gt_track"] is not None:
            jerseys_by_gt[row["gt_track"]].append(row["visible_jersey"])
        if row["display"] is not None:
            gt_tracks_by_display[row["display"]].append(row["gt_track"])
            roles_by_display[row["display"]].append(row["role"])
    canonical = {track: mode(values) for track, values in jerseys_by_gt.items()}
    dominant_gt = {
        display: mode(values) for display, values in gt_tracks_by_display.items()
    }
    gt_jersey = {
        display: canonical.get(track) for display, track in dominant_gt.items()
    }
    gt_role = {display: mode(values) for display, values in roles_by_display.items()}
    return canonical, gt_jersey, gt_role


def load_decisions(root, run, sequence):
    path = root / run / "metadata" / f"{sequence}_jersey_frame_decisions.csv"
    output = {}
    for row in read_csv(path):
        display = str(row["display_track_id"])
        output[display] = integer(row.get("decision")) if row["accepted"] == "True" else None
    return output


def summarize(sequence, label, predictions, matches, canonical, gt_jersey, gt_role):
    evaluable = [
        display for display, jersey in gt_jersey.items()
        if jersey is not None and display in predictions
    ]
    emitted = [display for display in evaluable if predictions[display] is not None]
    correct = [display for display in emitted if predictions[display] == gt_jersey[display]]
    wrong = [display for display in emitted if predictions[display] != gt_jersey[display]]
    visible = [
        row for row in matches
        if row["display"] in predictions and row["visible_jersey"] is not None
    ]
    visible_emitted = [row for row in visible if predictions[row["display"]] is not None]
    visible_correct = [
        row for row in visible_emitted
        if predictions[row["display"]] == row["visible_jersey"]
    ]
    no_jersey_gt = {track for track, jersey in canonical.items() if jersey is None}
    false_no_jersey = [
        row for row in matches
        if row["gt_track"] in no_jersey_gt
        and row["display"] in predictions
        and predictions[row["display"]] is not None
    ]
    return {
        "sequence": sequence,
        "variant": label,
        "evaluable": len(evaluable),
        "emitted": len(emitted),
        "correct": len(correct),
        "wrong": len(wrong),
        "track_coverage": ratio(len(emitted), len(evaluable)),
        "track_accuracy_emitted": ratio(len(correct), len(emitted)),
        "track_accuracy_all": ratio(len(correct), len(evaluable)),
        "visible_frames": len(visible),
        "visible_emitted": len(visible_emitted),
        "visible_correct": len(visible_correct),
        "frame_accuracy_all": ratio(len(visible_correct), len(visible)),
        "false_no_jersey_tracks": len({row["display"] for row in false_no_jersey}),
        "false_no_jersey_frames": len(false_no_jersey),
        "role_contaminations": sum(
            gt_role.get(display) not in {None, "player"} for display in emitted
        ),
    }


def compare_decisions(sequence, variants, gt_jersey):
    baseline = variants["baseline"]
    candidate = variants["candidate"]
    output = []
    for display in sorted(set(baseline) | set(candidate), key=lambda value: int(value)):
        before = baseline.get(display)
        after = candidate.get(display)
        if before == after:
            continue
        truth = gt_jersey.get(display)
        output.append({
            "sequence": sequence,
            "display_track_id": display,
            "baseline": before,
            "candidate": after,
            "gt_jersey": truth,
            "baseline_correct": before is not None and before == truth,
            "candidate_correct": after is not None and after == truth,
            "new_emission": before is None and after is not None,
            "abstention": before is not None and after is None,
        })
    return output


def pool(rows):
    keys = [
        "evaluable", "emitted", "correct", "wrong", "visible_frames",
        "visible_emitted", "visible_correct", "false_no_jersey_tracks",
        "false_no_jersey_frames", "role_contaminations",
    ]
    totals = {key: sum(row[key] for row in rows) for key in keys}
    return {
        **totals,
        "track_coverage": ratio(totals["emitted"], totals["evaluable"]),
        "track_accuracy_emitted": ratio(totals["correct"], totals["emitted"]),
        "track_accuracy_all": ratio(totals["correct"], totals["evaluable"]),
        "frame_accuracy_all": ratio(totals["visible_correct"], totals["visible_frames"]),
    }


def mode(values):
    filtered = [value for value in values if value is not None]
    return Counter(filtered).most_common(1)[0][0] if filtered else None


def integer(value):
    return None if value in EMPTY else int(float(value))


def text(value):
    return None if value in EMPTY else str(value)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


if __name__ == "__main__":
    main()
