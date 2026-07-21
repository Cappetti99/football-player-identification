#!/usr/bin/env python3
"""Evaluate legibility-filtered OCR grouped by FT display tracks.

The script simulates a deployable decision path: crop selection, voting and
role gates use only FT outputs. Ground truth is joined *after* those decisions
and is used solely to report metrics. This prevents the earlier oracle mistake
of grouping observations by ``gt_track_id``.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--min-winner-votes", type=int, default=1)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument(
        "--allowed-roles", nargs="+", default=["player", "goalkeeper", "referee"]
    )
    parser.add_argument("--gate-each-frame-role", action="store_true")
    args = parser.parse_args()

    crops = [normalize_crop(row) for row in read_csv(Path(args.scores))]
    matches = [normalize_match(row) for row in read_csv(Path(args.matches))]
    crops = [row for row in crops if row["display_track_id"] is not None]
    matches = [
        row for row in matches
        if row["display_track_id"] is not None and row["gt_track_id"] is not None
    ]

    # Operational side: a running system knows display_track_id, legibility,
    # OCR candidates and predicted roles. It does not know any GT identifier.
    crop_groups = defaultdict(list)
    for row in crops:
        crop_groups[row["display_track_id"]].append(row)

    match_groups = defaultdict(list)
    for row in matches:
        match_groups[row["display_track_id"]].append(row)

    decisions = {}
    decision_rows = []
    for display_id in sorted(set(crop_groups) | set(match_groups), key=sort_key):
        selected = select_crops(
            crop_groups.get(display_id, []),
            args.top_k,
            args.min_score,
            args.min_frame_gap,
        )
        raw_prediction, margin, winner_votes = vote(selected)
        matched = match_groups.get(display_id, [])
        dominant_pred_role = mode(
            row["pred_role"] for row in matched if row["pred_role"] is not None
        )
        accepted = (
            raw_prediction is not None
            and winner_votes >= args.min_winner_votes
            and margin is not None
            and margin >= args.min_margin
            and dominant_pred_role in set(args.allowed_roles)
        )
        prediction = raw_prediction if accepted else None
        # Evaluation side begins here. Dominant GT association and purity
        # describe the already-made display decision; they never affect it.
        gt_counts = Counter(
            row["gt_track_id"] for row in matched if row["gt_track_id"] is not None
        )
        dominant_gt = gt_counts.most_common(1)[0][0] if gt_counts else None
        dominant_gt_jersey = mode(
            row["gt_jersey"] for row in matched
            if row["gt_track_id"] == dominant_gt and row["gt_jersey"] is not None
        )
        purity = ratio(gt_counts.get(dominant_gt, 0), sum(gt_counts.values()))
        decisions[display_id] = prediction
        decision_rows.append({
            "display_track_id": display_id,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "accepted": accepted,
            "dominant_pred_role": dominant_pred_role,
            "dominant_gt_track_id": dominant_gt,
            "dominant_gt_jersey": dominant_gt_jersey,
            "correct_on_dominant_gt": (
                prediction == dominant_gt_jersey
                if prediction is not None and dominant_gt_jersey is not None else None
            ),
            "association_purity": purity,
            "matched_frames": len(matched),
            "selected_crops": len(selected),
            "selected_frames": [row["frame"] for row in selected],
            "selected_winners": [row["winner"] for row in selected],
            "margin": margin,
            "winner_votes": winner_votes,
        })

    allowed_roles = set(args.allowed_roles)

    def frame_prediction(row):
        """Expose a stored track decision only on semantically valid frames."""
        prediction = decisions.get(row["display_track_id"])
        if args.gate_each_frame_role and row["pred_role"] not in allowed_roles:
            return None
        return prediction

    visible_matches = [row for row in matches if row["gt_jersey"] is not None]
    frame_emitted = [
        row for row in visible_matches if frame_prediction(row) is not None
    ]
    frame_correct = sum(
        frame_prediction(row) == row["gt_jersey"]
        for row in frame_emitted
    )
    not_visible_matches = [row for row in matches if row["gt_jersey"] is None]
    false_emitted_not_visible = sum(
        frame_prediction(row) is not None
        for row in not_visible_matches
    )
    # A missing per-frame jersey can mean either "not currently readable" or
    # "this GT identity has no jersey label anywhere". Report those cases
    # separately instead of treating all propagation as an OCR false positive.
    canonical_jerseys = canonical_jerseys_by_gt(matches)
    hidden_identity_evaluable = [
        row for row in not_visible_matches
        if canonical_jerseys.get(row["gt_track_id"]) is not None
    ]
    hidden_identity_emitted = [
        row for row in hidden_identity_evaluable
        if frame_prediction(row) is not None
    ]
    hidden_identity_correct = sum(
        frame_prediction(row)
        == canonical_jerseys.get(row["gt_track_id"])
        for row in hidden_identity_emitted
    )

    evaluable_display = [
        row for row in decision_rows if row["dominant_gt_jersey"] is not None
    ]
    emitted_display = [row for row in evaluable_display if row["prediction"] is not None]
    correct_display = sum(row["correct_on_dominant_gt"] is True for row in emitted_display)

    gt_rows = build_gt_summary(matches, decisions)
    summary = {
        "grouping": "display_track_id",
        "top_k": args.top_k,
        "min_legibility_score": args.min_score,
        "min_frame_gap": args.min_frame_gap,
        "min_winner_votes": args.min_winner_votes,
        "min_margin": args.min_margin,
        "allowed_roles": args.allowed_roles,
        "gate_each_frame_role": args.gate_each_frame_role,
        "display_tracks": {
            "total": len(decision_rows),
            "evaluable_visible": len(evaluable_display),
            "emitted": len(emitted_display),
            "coverage": ratio(len(emitted_display), len(evaluable_display)),
            "correct_on_dominant_gt": correct_display,
            "accuracy_on_emitted": ratio(correct_display, len(emitted_display)),
        },
        "matched_visible_frames": {
            "total": len(visible_matches),
            "emitted": len(frame_emitted),
            "coverage": ratio(len(frame_emitted), len(visible_matches)),
            "correct": frame_correct,
            "accuracy_on_emitted": ratio(frame_correct, len(frame_emitted)),
            "accuracy_all_visible": ratio(frame_correct, len(visible_matches)),
        },
        "matched_not_visible_frames": {
            "total": len(not_visible_matches),
            "false_emitted": false_emitted_not_visible,
            "false_emission_rate": ratio(
                false_emitted_not_visible, len(not_visible_matches)
            ),
            "interpretation": "strict per-frame OCR observability",
        },
        "hidden_frame_identity_propagation": {
            "interpretation": "identity correctness using the GT track canonical jersey",
            "evaluable": len(hidden_identity_evaluable),
            "emitted": len(hidden_identity_emitted),
            "coverage": ratio(
                len(hidden_identity_emitted), len(hidden_identity_evaluable)
            ),
            "correct": hidden_identity_correct,
            "accuracy_on_emitted": ratio(
                hidden_identity_correct, len(hidden_identity_emitted)
            ),
            "accuracy_all_evaluable": ratio(
                hidden_identity_correct, len(hidden_identity_evaluable)
            ),
        },
        "gt_tracklets_diagnostic": summarize_gt(gt_rows),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "display_decisions": decision_rows,
                    "gt_tracklets": gt_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_gt_summary(matches, decisions):
    """Produce tracklet diagnostics after operational decisions are frozen."""
    grouped = defaultdict(list)
    for row in matches:
        if row["gt_track_id"] is not None:
            grouped[row["gt_track_id"]].append(row)
    output = []
    for gt_track_id, rows in sorted(grouped.items(), key=lambda item: sort_key(item[0])):
        gt_jersey = mode(row["gt_jersey"] for row in rows if row["gt_jersey"] is not None)
        display_counts = Counter(row["display_track_id"] for row in rows)
        dominant_display = display_counts.most_common(1)[0][0] if display_counts else None
        dominant_prediction = decisions.get(dominant_display)
        per_frame_predictions = [
            decisions.get(row["display_track_id"])
            for row in rows if decisions.get(row["display_track_id"]) is not None
        ]
        frame_mode_prediction = mode(per_frame_predictions)
        output.append({
            "gt_track_id": gt_track_id,
            "gt_jersey": gt_jersey,
            "display_track_count": len(display_counts),
            "dominant_display_track_id": dominant_display,
            "dominant_display_prediction": dominant_prediction,
            "dominant_display_correct": (
                dominant_prediction == gt_jersey
                if dominant_prediction is not None and gt_jersey is not None else None
            ),
            "frame_mode_prediction": frame_mode_prediction,
            "frame_mode_correct": (
                frame_mode_prediction == gt_jersey
                if frame_mode_prediction is not None and gt_jersey is not None else None
            ),
        })
    return output


def canonical_jerseys_by_gt(matches):
    grouped = defaultdict(list)
    for row in matches:
        if row["gt_track_id"] is not None and row["gt_jersey"] is not None:
            grouped[row["gt_track_id"]].append(row["gt_jersey"])
    return {gt_track_id: mode(values) for gt_track_id, values in grouped.items()}


def summarize_gt(rows):
    visible = [row for row in rows if row["gt_jersey"] is not None]
    dominant_emitted = [
        row for row in visible if row["dominant_display_prediction"] is not None
    ]
    mode_emitted = [row for row in visible if row["frame_mode_prediction"] is not None]
    return {
        "visible": len(visible),
        "dominant_display": {
            "emitted": len(dominant_emitted),
            "correct": sum(row["dominant_display_correct"] is True for row in dominant_emitted),
            "accuracy": ratio(
                sum(row["dominant_display_correct"] is True for row in dominant_emitted),
                len(visible),
            ),
        },
        "frame_weighted_mode": {
            "emitted": len(mode_emitted),
            "correct": sum(row["frame_mode_correct"] is True for row in mode_emitted),
            "accuracy": ratio(
                sum(row["frame_mode_correct"] is True for row in mode_emitted),
                len(visible),
            ),
        },
    }


def select_crops(rows, limit, min_score, min_frame_gap):
    """Rank readable crops while suppressing near-duplicate video frames."""
    ranked = sorted(rows, key=lambda row: row["legibility_score"], reverse=True)
    selected = []
    for row in ranked:
        if row["legibility_score"] < min_score:
            continue
        if min_frame_gap and any(
            abs(row["frame"] - other["frame"]) < min_frame_gap for other in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def vote(rows):
    """Choose a jersey by summed OCR confidence across selected crops."""
    scores = defaultdict(float)
    counts = Counter()
    for row in rows:
        if row["winner"] is None:
            continue
        scores[row["winner"]] += max(0.001, row["winner_confidence"])
        counts[row["winner"]] += 1
    if not scores:
        return None, None, 0
    ranked = sorted(scores.items(), key=lambda item: (item[1], counts[item[0]]), reverse=True)
    winner, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(scores.values())
    return winner, (winner_score - runner_up) / total if total else None, counts[winner]


def normalize_crop(row):
    return {
        "display_track_id": text(row.get("display_track_id")),
        "frame": integer(row.get("frame")) or 0,
        "winner": jersey(row.get("winner")),
        "winner_confidence": floating(row.get("winner_confidence")),
        "legibility_score": floating(row.get("legibility_score")),
    }


def normalize_match(row):
    return {
        "display_track_id": text(row.get("pred_track_id")),
        "gt_track_id": text(row.get("gt_track_id")),
        "gt_jersey": jersey(row.get("gt_jersey")),
        "pred_role": text(row.get("pred_role")),
        "frame": integer(row.get("frame")) or 0,
    }


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def text(value):
    return None if value in {None, "", "None", "null", "unknown", "-1", -1} else str(value)


def integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def floating(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def jersey(value):
    value = integer(value)
    return value if value is not None and 1 <= value <= 99 else None


def mode(values):
    values = list(values)
    return Counter(values).most_common(1)[0][0] if values else None


def ratio(a, b):
    return float(a / b) if b else None


def sort_key(value):
    number = integer(value)
    return (0, number) if number is not None else (1, str(value))


if __name__ == "__main__":
    main()
