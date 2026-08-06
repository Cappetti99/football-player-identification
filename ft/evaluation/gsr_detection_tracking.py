"""Detection and multi-object tracking metrics for SoccerNet-GSR artifacts.

The functions in this module are deliberately independent from the FT runtime:
ground truth is consumed only after inference.  Tracking is reported on both the
raw tracker identity and the final FT display identity by choosing ``id_key``.
"""

from collections import Counter, defaultdict
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


HOTA_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
SIZE_BINS = (
    ("small", 0.0, 0.005),
    ("medium", 0.005, 0.02),
    ("large", 0.02, float("inf")),
)


def evaluate_frames(gt, pred, threshold=0.50, continuity_id_key=None):
    """Perform role-agnostic one-to-one frame matching at an IoU threshold.

    When ``continuity_id_key`` is set, valid assignments from the immediately
    previous frame are retained before Hungarian matching, as required by the
    CLEAR MOT protocol. Detection evaluation leaves this unset.
    """
    matches = []
    gt_outcomes = []
    pred_outcomes = []
    frame_metrics = []
    counts = Counter()
    previous_assignments = {}

    for frame in sorted(set(gt) | set(pred)):
        gt_rows = gt.get(frame, [])
        pred_rows = pred.get(frame, [])
        used_gt = set()
        used_pred = set()
        frame_matches = []
        current_assignments = {}

        if gt_rows and pred_rows:
            similarities = np.asarray(
                [
                    [bbox_iou(gt_row["bbox"], pred_row["bbox"]) for pred_row in pred_rows]
                    for gt_row in gt_rows
                ],
                dtype=np.float64,
            )
            if continuity_id_key:
                gt_by_identity = {
                    row["gt_track_id"]: index for index, row in enumerate(gt_rows)
                }
                pred_by_identity = {
                    row[continuity_id_key]: index for index, row in enumerate(pred_rows)
                }
                for gt_identity, pred_identity in previous_assignments.items():
                    gt_index = gt_by_identity.get(gt_identity)
                    pred_index = pred_by_identity.get(pred_identity)
                    if gt_index is None or pred_index is None:
                        continue
                    score = float(similarities[gt_index, pred_index])
                    if score < threshold:
                        continue
                    used_gt.add(gt_index)
                    used_pred.add(pred_index)
                    current_assignments[gt_identity] = pred_identity
                    match = {
                        "frame": int(frame),
                        "iou": score,
                        **gt_rows[gt_index],
                        **pred_rows[pred_index],
                    }
                    matches.append(match)
                    frame_matches.append(match)

            remaining_gt = [index for index in range(len(gt_rows)) if index not in used_gt]
            remaining_pred = [index for index in range(len(pred_rows)) if index not in used_pred]
            if remaining_gt and remaining_pred:
                remaining_similarities = similarities[np.ix_(remaining_gt, remaining_pred)]
                cost = 1.0 - remaining_similarities
                cost[remaining_similarities < threshold] = 1e6
                assigned_gt, assigned_pred = linear_sum_assignment(cost)
                gt_indices = [remaining_gt[int(index)] for index in assigned_gt]
                pred_indices = [remaining_pred[int(index)] for index in assigned_pred]
            else:
                gt_indices, pred_indices = [], []
            for gt_index, pred_index in zip(gt_indices, pred_indices):
                score = float(similarities[gt_index, pred_index])
                if score < threshold:
                    continue
                gt_index = int(gt_index)
                pred_index = int(pred_index)
                used_gt.add(gt_index)
                used_pred.add(pred_index)
                if continuity_id_key:
                    current_assignments[gt_rows[gt_index]["gt_track_id"]] = pred_rows[pred_index][continuity_id_key]
                match = {
                    "frame": int(frame),
                    "iou": score,
                    **gt_rows[gt_index],
                    **pred_rows[pred_index],
                }
                matches.append(match)
                frame_matches.append(match)

        if continuity_id_key:
            previous_assignments = current_assignments

        for index, row in enumerate(gt_rows):
            gt_outcomes.append(
                {"frame": int(frame), "matched": index in used_gt, **row}
            )
        for index, row in enumerate(pred_rows):
            pred_outcomes.append(
                {"frame": int(frame), "matched": index in used_pred, **row}
            )

        tp = len(used_gt)
        fn = len(gt_rows) - tp
        fp = len(pred_rows) - len(used_pred)
        counts.update(gt=len(gt_rows), pred=len(pred_rows), tp=tp, fn=fn, fp=fp)
        frame_metrics.append(
            {
                "frame": int(frame),
                "gt": len(gt_rows),
                "pred": len(pred_rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": ratio(tp, tp + fp),
                "recall": ratio(tp, tp + fn),
                "f1": ratio(2 * tp, 2 * tp + fp + fn),
                "mean_iou": mean(row["iou"] for row in frame_matches),
            }
        )

    return {
        "threshold": float(threshold),
        "matches": matches,
        "counts": counts,
        "gt_outcomes": gt_outcomes,
        "pred_outcomes": pred_outcomes,
        "frame_metrics": frame_metrics,
    }


def detection_summary(evaluation):
    counts = evaluation["counts"]
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "mean_iou": mean(row["iou"] for row in evaluation["matches"]),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "gt": int(counts["gt"]),
        "pred": int(counts["pred"]),
        "fp_per_frame": ratio(fp, len(evaluation["frame_metrics"])),
        "fn_per_frame": ratio(fn, len(evaluation["frame_metrics"])),
    }


def average_precision(gt, pred, threshold):
    """Compute 101-point interpolated AP from exported detector confidence."""
    predictions = [row for frame_rows in pred.values() for row in frame_rows]
    missing = sum(row.get("detection_confidence") is None for row in predictions)
    if not predictions:
        return {
            "available": True,
            "reason": None,
            "ap": 0.0 if sum(len(rows) for rows in gt.values()) else None,
            "gt": sum(len(rows) for rows in gt.values()),
            "pred": 0,
            "curve": [],
        }
    if missing:
        return {
            "available": False,
            "reason": "detection_confidence_missing",
            "missing_predictions": int(missing),
            "gt": sum(len(rows) for rows in gt.values()),
            "pred": len(predictions),
            "ap": None,
            "curve": [],
        }

    ranked = []
    for frame, rows in pred.items():
        for index, row in enumerate(rows):
            ranked.append((float(row["detection_confidence"]), int(frame), index, row))
    ranked.sort(key=lambda item: (-item[0], item[1], str(item[3].get("pred_identity_id")), item[2]))

    matched_gt = defaultdict(set)
    tp_flags = []
    fp_flags = []
    for confidence, frame, _, prediction in ranked:
        candidates = []
        for gt_index, gt_row in enumerate(gt.get(frame, [])):
            if gt_index in matched_gt[frame]:
                continue
            candidates.append((bbox_iou(gt_row["bbox"], prediction["bbox"]), gt_index))
        best_iou, best_index = max(candidates, default=(0.0, None))
        is_true = best_index is not None and best_iou >= threshold
        if is_true:
            matched_gt[frame].add(best_index)
        tp_flags.append(1 if is_true else 0)
        fp_flags.append(0 if is_true else 1)

    total_gt = sum(len(rows) for rows in gt.values())
    cumulative_tp = np.cumsum(tp_flags, dtype=np.float64)
    cumulative_fp = np.cumsum(fp_flags, dtype=np.float64)
    recalls = cumulative_tp / total_gt if total_gt else np.zeros(len(ranked))
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precisions[recalls >= recall_level]))
        if np.any(recalls >= recall_level)
        else 0.0
        for recall_level in recall_grid
    ]
    curve = [
        {
            "rank": index + 1,
            "confidence": float(ranked[index][0]),
            "precision": float(precisions[index]),
            "recall": float(recalls[index]),
            "tp": int(cumulative_tp[index]),
            "fp": int(cumulative_fp[index]),
        }
        for index in range(len(ranked))
    ]
    return {
        "available": True,
        "reason": None,
        "ap": float(np.mean(interpolated)) if total_gt else None,
        "gt": int(total_gt),
        "pred": len(ranked),
        "curve": curve,
    }


def tracking_summary(evaluation, gt, pred, id_key="pred_identity_id"):
    """Compute CLEAR MOT, identity and trajectory diagnostics at one IoU gate."""
    matches = evaluation["matches"]
    total_gt = sum(len(rows) for rows in gt.values())
    total_pred = sum(len(rows) for rows in pred.values())
    gt_counts = identity_counts(gt, "gt_track_id")
    pred_counts = identity_counts(pred, id_key)
    pair_counts = Counter((row["gt_track_id"], row[id_key]) for row in matches)
    identity_pairs = potential_identity_pair_counts(
        gt, pred, evaluation["threshold"], id_key
    )

    idtp = global_identity_true_positives(identity_pairs)
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    switches, fragments, coverage = temporal_identity_errors(gt, matches, id_key)

    pred_to_gt = defaultdict(Counter)
    for (gt_id, pred_id), count in pair_counts.items():
        pred_to_gt[pred_id][gt_id] += count
    purity_numerator = sum(
        counter.most_common(1)[0][1] for counter in pred_to_gt.values() if counter
    )

    mostly_tracked = sum(value >= 0.80 for value in coverage.values())
    mostly_lost = sum(value < 0.20 for value in coverage.values())
    partially_tracked = len(coverage) - mostly_tracked - mostly_lost
    tp = evaluation["counts"]["tp"]
    fp = evaluation["counts"]["fp"]
    fn = evaluation["counts"]["fn"]

    return {
        "identity_surface": id_key,
        "gt_tracks": len(gt_counts),
        "pred_tracks": len(pred_counts),
        "gt_detections": int(total_gt),
        "pred_detections": int(total_pred),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "gt_tracks_matched": len({row["gt_track_id"] for row in matches}),
        "pred_tracks_matched": len({row[id_key] for row in matches}),
        "mota": None if not total_gt else float(1.0 - (fn + fp + switches) / total_gt),
        "motp": mean(row["iou"] for row in matches),
        "id_precision": ratio(idtp, idtp + idfp),
        "id_recall": ratio(idtp, idtp + idfn),
        "idf1": ratio(2 * idtp, 2 * idtp + idfp + idfn),
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
        "id_switches": int(switches),
        "id_switches_per_1000_gt": ratio(1000 * switches, total_gt),
        "fragmentations": int(fragments),
        "fragmentations_per_1000_gt": ratio(1000 * fragments, total_gt),
        "mostly_tracked": int(mostly_tracked),
        "partially_tracked": int(partially_tracked),
        "mostly_lost": int(mostly_lost),
        "mostly_tracked_rate": ratio(mostly_tracked, len(coverage)),
        "mostly_lost_rate": ratio(mostly_lost, len(coverage)),
        "association_purity": ratio(purity_numerator, len(matches)),
        "track_coverage_mean": mean(coverage.values()),
        "track_coverage_median": percentile(coverage.values(), 50),
    }


def hota_summary(gt, pred, id_key="pred_identity_id", thresholds=HOTA_THRESHOLDS):
    """Compute HOTA with TrackEval's global-alignment matching procedure."""
    gt_counter = identity_counts(gt, "gt_track_id")
    pred_counter = identity_counts(pred, id_key)
    gt_ids = sorted(gt_counter, key=str)
    pred_ids = sorted(pred_counter, key=str)
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    pred_index = {value: index for index, value in enumerate(pred_ids)}
    gt_counts = np.asarray([gt_counter[value] for value in gt_ids], dtype=np.float64)
    pred_counts = np.asarray([pred_counter[value] for value in pred_ids], dtype=np.float64)

    frame_data = []
    potential_matches = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for frame in sorted(set(gt) | set(pred)):
        gt_rows = gt.get(frame, [])
        pred_rows = pred.get(frame, [])
        gt_indices = np.asarray(
            [gt_index[row["gt_track_id"]] for row in gt_rows], dtype=np.int64
        )
        pred_indices = np.asarray(
            [pred_index[row[id_key]] for row in pred_rows], dtype=np.int64
        )
        similarities = np.asarray(
            [
                [bbox_iou(gt_row["bbox"], pred_row["bbox"]) for pred_row in pred_rows]
                for gt_row in gt_rows
            ],
            dtype=np.float64,
        )
        if not gt_rows or not pred_rows:
            similarities = np.zeros((len(gt_rows), len(pred_rows)), dtype=np.float64)
        if similarities.size:
            denominator = (
                similarities.sum(axis=0)[None, :]
                + similarities.sum(axis=1)[:, None]
                - similarities
            )
            similarity_iou = np.divide(
                similarities,
                denominator,
                out=np.zeros_like(similarities),
                where=denominator > 0,
            )
            potential_matches[np.ix_(gt_indices, pred_indices)] += similarity_iou
        frame_data.append((gt_indices, pred_indices, similarities))

    alignment_denominator = (
        gt_counts[:, None] + pred_counts[None, :] - potential_matches
    )
    global_alignment = np.divide(
        potential_matches,
        alignment_denominator,
        out=np.zeros_like(potential_matches),
        where=alignment_denominator > 0,
    )

    per_alpha = []
    for threshold in thresholds:
        match_counts = np.zeros_like(potential_matches)
        tp = 0
        localization_sum = 0.0
        for gt_indices, pred_indices, similarities in frame_data:
            if not similarities.size:
                continue
            score = global_alignment[np.ix_(gt_indices, pred_indices)] * similarities
            row_indices, column_indices = linear_sum_assignment(-score)
            valid = similarities[row_indices, column_indices] >= threshold
            row_indices = row_indices[valid]
            column_indices = column_indices[valid]
            if not len(row_indices):
                continue
            matched_gt = gt_indices[row_indices]
            matched_pred = pred_indices[column_indices]
            np.add.at(match_counts, (matched_gt, matched_pred), 1)
            tp += len(row_indices)
            localization_sum += float(similarities[row_indices, column_indices].sum())

        fn = int(gt_counts.sum()) - tp
        fp = int(pred_counts.sum()) - tp
        association_denominator = (
            gt_counts[:, None] + pred_counts[None, :] - match_counts
        )
        association_accuracy = np.divide(
            match_counts,
            association_denominator,
            out=np.zeros_like(match_counts),
            where=association_denominator > 0,
        )
        association_recall = np.divide(
            match_counts,
            gt_counts[:, None],
            out=np.zeros_like(match_counts),
            where=gt_counts[:, None] > 0,
        )
        association_precision = np.divide(
            match_counts,
            pred_counts[None, :],
            out=np.zeros_like(match_counts),
            where=pred_counts[None, :] > 0,
        )
        assa = ratio(float((match_counts * association_accuracy).sum()), tp)
        ass_recall = ratio(float((match_counts * association_recall).sum()), tp)
        ass_precision = ratio(float((match_counts * association_precision).sum()), tp)
        if tp == 0 and (fn or fp):
            assa = ass_recall = ass_precision = 0.0
        deta = ratio(tp, tp + fn + fp)
        hota = (
            math.sqrt(max(0.0, deta * assa))
            if deta is not None and assa is not None
            else None
        )
        per_alpha.append(
            {
                "alpha": float(threshold),
                "hota": hota,
                "deta": deta,
                "assa": assa,
                "ass_recall": ass_recall,
                "ass_precision": ass_precision,
                "loca": ratio(localization_sum, tp) if tp else (0.0 if fn or fp else None),
                "det_recall": ratio(tp, tp + fn),
                "det_precision": ratio(tp, tp + fp),
            }
        )

    alpha_50 = min(per_alpha, key=lambda row: abs(row["alpha"] - 0.50))
    return {
        "thresholds": [row["alpha"] for row in per_alpha],
        "hota": mean(row["hota"] for row in per_alpha if row["hota"] is not None),
        "deta": mean(row["deta"] for row in per_alpha if row["deta"] is not None),
        "assa": mean(row["assa"] for row in per_alpha if row["assa"] is not None),
        "loca": mean(row["loca"] for row in per_alpha if row["loca"] is not None),
        "at_0_50": dict(alpha_50),
        "per_alpha": per_alpha,
    }


def gs_hota_similarity(gt_row, pred_row, tau=5.0):
    """Sim_GS-HOTA(P, G) = LocSim(P, G) x IdSim(P, G).

    From Somers et al., "SoccerNet Game State Reconstruction: End-to-End
    Athlete Tracking and Identification on a Minimap" (CVPR24 CVSports
    workshop, arXiv:2404.11335, eq. 3-5). LocSim is a Gaussian kernel over
    Euclidean pitch distance in metres (not bounding-box IoU); tau=5m is the
    paper's own distance-tolerance parameter. IdSim requires role, team and
    jersey number to all match, except an attribute is skipped when the
    ground truth does not provide it (e.g. a referee's jersey number).
    """
    gt_point = gt_row.get("gt_position_pitch")
    pred_point = pred_row.get("pred_position_pitch")
    if gt_point is None or pred_point is None:
        return 0.0
    dx = float(gt_point[0]) - float(pred_point[0])
    dy = float(gt_point[1]) - float(pred_point[1])
    squared_distance = dx * dx + dy * dy
    loc_sim = math.exp(math.log(0.05) * squared_distance / (float(tau) ** 2))

    for gt_key, pred_key in (
        ("gt_role", "pred_role"),
        ("gt_team", "pred_team"),
        ("gt_jersey", "pred_jersey"),
    ):
        gt_value = gt_row.get(gt_key)
        if gt_value is None:
            continue
        if gt_value != pred_row.get(pred_key):
            return 0.0
    return loc_sim


def gs_hota_summary(gt, pred, id_key="pred_identity_id", thresholds=HOTA_THRESHOLDS, tau=5.0):
    """GS-HOTA: HOTA's global-alignment procedure with Sim_GS-HOTA in place
    of bounding-box IoU. Requires gt_position_pitch/pred_position_pitch
    (metres) and the role/team/jersey attribute pairs on every row, as
    loaded by scripts/evaluate_ft_gsr.py's load_ground_truth/load_predictions.
    """
    gt_counter = identity_counts(gt, "gt_track_id")
    pred_counter = identity_counts(pred, id_key)
    gt_ids = sorted(gt_counter, key=str)
    pred_ids = sorted(pred_counter, key=str)
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    pred_index = {value: index for index, value in enumerate(pred_ids)}
    gt_counts = np.asarray([gt_counter[value] for value in gt_ids], dtype=np.float64)
    pred_counts = np.asarray([pred_counter[value] for value in pred_ids], dtype=np.float64)

    frame_data = []
    potential_matches = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for frame in sorted(set(gt) | set(pred)):
        gt_rows = gt.get(frame, [])
        pred_rows = pred.get(frame, [])
        gt_indices = np.asarray(
            [gt_index[row["gt_track_id"]] for row in gt_rows], dtype=np.int64
        )
        pred_indices = np.asarray(
            [pred_index[row[id_key]] for row in pred_rows], dtype=np.int64
        )
        similarities = np.asarray(
            [
                [gs_hota_similarity(gt_row, pred_row, tau) for pred_row in pred_rows]
                for gt_row in gt_rows
            ],
            dtype=np.float64,
        )
        if not gt_rows or not pred_rows:
            similarities = np.zeros((len(gt_rows), len(pred_rows)), dtype=np.float64)
        if similarities.size:
            denominator = (
                similarities.sum(axis=0)[None, :]
                + similarities.sum(axis=1)[:, None]
                - similarities
            )
            similarity_iou = np.divide(
                similarities,
                denominator,
                out=np.zeros_like(similarities),
                where=denominator > 0,
            )
            potential_matches[np.ix_(gt_indices, pred_indices)] += similarity_iou
        frame_data.append((gt_indices, pred_indices, similarities))

    alignment_denominator = (
        gt_counts[:, None] + pred_counts[None, :] - potential_matches
    )
    global_alignment = np.divide(
        potential_matches,
        alignment_denominator,
        out=np.zeros_like(potential_matches),
        where=alignment_denominator > 0,
    )

    per_alpha = []
    for threshold in thresholds:
        match_counts = np.zeros_like(potential_matches)
        tp = 0
        localization_sum = 0.0
        for gt_indices, pred_indices, similarities in frame_data:
            if not similarities.size:
                continue
            score = global_alignment[np.ix_(gt_indices, pred_indices)] * similarities
            row_indices, column_indices = linear_sum_assignment(-score)
            valid = similarities[row_indices, column_indices] >= threshold
            row_indices = row_indices[valid]
            column_indices = column_indices[valid]
            if not len(row_indices):
                continue
            matched_gt = gt_indices[row_indices]
            matched_pred = pred_indices[column_indices]
            np.add.at(match_counts, (matched_gt, matched_pred), 1)
            tp += len(row_indices)
            localization_sum += float(similarities[row_indices, column_indices].sum())

        fn = int(gt_counts.sum()) - tp
        fp = int(pred_counts.sum()) - tp
        association_denominator = (
            gt_counts[:, None] + pred_counts[None, :] - match_counts
        )
        association_accuracy = np.divide(
            match_counts,
            association_denominator,
            out=np.zeros_like(match_counts),
            where=association_denominator > 0,
        )
        assa = ratio(float((match_counts * association_accuracy).sum()), tp)
        if tp == 0 and (fn or fp):
            assa = 0.0
        deta = ratio(tp, tp + fn + fp)
        hota = (
            math.sqrt(max(0.0, deta * assa))
            if deta is not None and assa is not None
            else None
        )
        per_alpha.append(
            {
                "alpha": float(threshold),
                "gs_hota": hota,
                "gs_deta": deta,
                "gs_assa": assa,
                "loca": ratio(localization_sum, tp) if tp else (0.0 if fn or fp else None),
                "det_recall": ratio(tp, tp + fn),
                "det_precision": ratio(tp, tp + fp),
            }
        )

    alpha_50 = min(per_alpha, key=lambda row: abs(row["alpha"] - 0.50))
    return {
        "tau_metres": float(tau),
        "thresholds": [row["alpha"] for row in per_alpha],
        "gs_hota": mean(row["gs_hota"] for row in per_alpha if row["gs_hota"] is not None),
        "gs_deta": mean(row["gs_deta"] for row in per_alpha if row["gs_deta"] is not None),
        "gs_assa": mean(row["gs_assa"] for row in per_alpha if row["gs_assa"] is not None),
        "loca": mean(row["loca"] for row in per_alpha if row["loca"] is not None),
        "at_0_50": dict(alpha_50),
        "per_alpha": per_alpha,
    }


def size_breakdown(evaluation, image_width, image_height):
    image_area = float(image_width) * float(image_height)
    result = {}
    for name, low, high in SIZE_BINS:
        gt_rows = [
            row
            for row in evaluation["gt_outcomes"]
            if low <= bbox_area(row["bbox"]) / image_area < high
        ]
        pred_rows = [
            row
            for row in evaluation["pred_outcomes"]
            if low <= bbox_area(row["bbox"]) / image_area < high
        ]
        tp_gt = sum(row["matched"] for row in gt_rows)
        tp_pred = sum(row["matched"] for row in pred_rows)
        result[name] = {
            "area_ratio_min": low,
            "area_ratio_max": None if math.isinf(high) else high,
            "gt": len(gt_rows),
            "matched_gt": int(tp_gt),
            "recall": ratio(tp_gt, len(gt_rows)),
            "pred": len(pred_rows),
            "matched_pred": int(tp_pred),
            "precision": ratio(tp_pred, len(pred_rows)),
        }
    return result


def role_breakdown(evaluation):
    roles = sorted(
        {row.get("gt_role") for row in evaluation["gt_outcomes"] if row.get("gt_role")}
        | {row.get("pred_role") for row in evaluation["pred_outcomes"] if row.get("pred_role")}
    )
    result = {}
    for role in roles:
        gt_rows = [row for row in evaluation["gt_outcomes"] if row.get("gt_role") == role]
        pred_rows = [row for row in evaluation["pred_outcomes"] if row.get("pred_role") == role]
        result[role] = {
            "gt": len(gt_rows),
            "recall": ratio(sum(row["matched"] for row in gt_rows), len(gt_rows)),
            "pred": len(pred_rows),
            "precision": ratio(sum(row["matched"] for row in pred_rows), len(pred_rows)),
        }
    return result


def temporal_identity_errors(gt, matches, id_key):
    match_by_gt_frame = {
        (row["gt_track_id"], int(row["frame"])): row[id_key] for row in matches
    }
    frames_by_gt = defaultdict(list)
    for frame, rows in gt.items():
        for row in rows:
            frames_by_gt[row["gt_track_id"]].append(int(frame))

    switches = 0
    fragments = 0
    coverage = {}
    for gt_id, frames in frames_by_gt.items():
        frames = sorted(set(frames))
        matched_count = 0
        last_identity = None
        previous_frame = None
        previous_was_matched = False
        missed_since_match = False
        for frame in frames:
            current = match_by_gt_frame.get((gt_id, frame))
            consecutive_annotation = previous_frame is not None and frame == previous_frame + 1
            if current is not None:
                matched_count += 1
                if (
                    consecutive_annotation
                    and previous_was_matched
                    and last_identity is not None
                    and current != last_identity
                ):
                    switches += 1
                if consecutive_annotation and missed_since_match and not previous_was_matched:
                    fragments += 1
                last_identity = current
                missed_since_match = False
                previous_was_matched = True
            else:
                if consecutive_annotation and last_identity is not None:
                    missed_since_match = True
                elif not consecutive_annotation:
                    missed_since_match = False
                previous_was_matched = False
            previous_frame = frame
        coverage[gt_id] = matched_count / len(frames) if frames else 0.0
    return switches, fragments, coverage


def global_identity_true_positives(pair_counts):
    if not pair_counts:
        return 0
    gt_ids = sorted({key[0] for key in pair_counts}, key=str)
    pred_ids = sorted({key[1] for key in pair_counts}, key=str)
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    gt_index = {value: index for index, value in enumerate(gt_ids)}
    pred_index = {value: index for index, value in enumerate(pred_ids)}
    for (gt_id, pred_id), count in pair_counts.items():
        matrix[gt_index[gt_id], pred_index[pred_id]] = count
    rows, columns = linear_sum_assignment(-matrix)
    return int(matrix[rows, columns].sum())


def potential_identity_pair_counts(gt, pred, threshold, id_key):
    """Count all localized trajectory-pair coincidences for global ID metrics."""
    counts = Counter()
    for frame in sorted(set(gt) | set(pred)):
        for gt_row in gt.get(frame, []):
            for pred_row in pred.get(frame, []):
                if bbox_iou(gt_row["bbox"], pred_row["bbox"]) >= threshold:
                    counts[(gt_row["gt_track_id"], pred_row[id_key])] += 1
    return counts


def identity_counts(rows_by_frame, key):
    counts = Counter()
    for rows in rows_by_frame.values():
        for row in rows:
            counts[row[key]] += 1
    return counts


def bbox_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_area(bbox):
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(
        0.0, float(bbox[3]) - float(bbox[1])
    )


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def percentile(values, q):
    values = list(values)
    return float(np.percentile(values, q)) if values else None
