#!/usr/bin/env python3
"""Evaluate FT frame-level artifacts against SoccerNet-GSR v1.3 ground truth."""

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ft.evaluation.gsr_detection_tracking import (
    average_precision,
    detection_summary,
    evaluate_frames,
    gs_hota_summary,
    hota_summary,
    role_breakdown,
    size_breakdown,
    tracking_summary as mot_tracking_summary,
)


EMPTY = {None, "", "None", "null", "unknown", "-1", -1}
ATHLETE_ROLES = {"player", "goalkeeper", "referee", "other"}
TRACKED_ROLES = {"player", "goalkeeper", "referee"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--tracklets", required=True)
    parser.add_argument(
        "--detections",
        default=None,
        help="Optional pre-tracker YOLO detections CSV. Without it, detection metrics use legacy tracklet boxes.",
    )
    parser.add_argument(
        "--detection-confidence-threshold",
        type=float,
        default=0.05,
        help="Operating threshold for P/R/F1; AP uses all exported detections.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--gt-roles",
        nargs="+",
        default=sorted(TRACKED_ROLES),
        choices=sorted(ATHLETE_ROLES),
        help="Ground-truth roles in the detection/tracking target surface.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--gt-pitch-coordinate-system",
        choices=["soccernet_centered", "ft"],
        default="soccernet_centered",
        help="Coordinate system of bbox_pitch. FT uses [0,105] x [0,68].",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Evaluate only frames [0, max_frames); intended for smoke tests.",
    )
    parser.add_argument(
        "--detection-iou-thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.75],
        help="IoU gates reported for detection metrics and AP.",
    )
    args = parser.parse_args()

    gt, metadata = load_ground_truth(
        Path(args.labels),
        allowed_roles=args.gt_roles,
        pitch_coordinate_system=args.gt_pitch_coordinate_system,
    )
    pred = load_predictions(Path(args.tracklets))
    all_detection_pred = (
        load_detections(Path(args.detections)) if args.detections else pred
    )
    detection_pred = (
        filter_detection_confidence(
            all_detection_pred, args.detection_confidence_threshold
        )
        if args.detections
        else all_detection_pred
    )
    if args.max_frames is not None:
        gt = {frame: rows for frame, rows in gt.items() if frame < args.max_frames}
        pred = {frame: rows for frame, rows in pred.items() if frame < args.max_frames}
        all_detection_pred = {
            frame: rows
            for frame, rows in all_detection_pred.items()
            if frame < args.max_frames
        }
        detection_pred = (
            filter_detection_confidence(
                all_detection_pred, args.detection_confidence_threshold
            )
            if args.detections
            else all_detection_pred
        )
    tracking_primary = evaluate_frames(gt, pred, args.iou_threshold)
    matches, counts = tracking_primary["matches"], tracking_primary["counts"]
    summary = summarize(
        matches,
        counts,
        gt,
        pred,
        metadata,
        args.iou_threshold,
        evaluation=tracking_primary,
    )

    detection_primary = evaluate_frames(gt, detection_pred, args.iou_threshold)
    summary["matching"]["source"] = "tracker_output"
    summary["detection_matching"] = {
        "iou_threshold": args.iou_threshold,
        "matched": detection_primary["counts"]["tp"],
        "gt": detection_primary["counts"]["gt"],
        "pred": detection_primary["counts"]["pred"],
        "source": (
            "raw_yolo_pre_tracker" if args.detections else "tracker_filtered_tracklets_legacy"
        ),
    }
    summary["detection"] = detection_summary(detection_primary)
    summary["detection"]["source"] = summary["detection_matching"]["source"]
    summary["detection"]["operating_confidence_threshold"] = (
        args.detection_confidence_threshold if args.detections else None
    )
    exported_confidences = [
        row.get("detection_confidence")
        for rows in all_detection_pred.values()
        for row in rows
        if row.get("detection_confidence") is not None
    ]
    summary["detection"]["exported_confidence_floor_observed"] = (
        min(exported_confidences) if exported_confidences else None
    )

    thresholds = sorted(set(args.detection_iou_thresholds + [args.iou_threshold]))
    threshold_evaluations = {
        threshold: detection_primary
        if abs(threshold - args.iou_threshold) < 1e-9
        else evaluate_frames(gt, detection_pred, threshold)
        for threshold in thresholds
    }
    summary["detection"]["by_iou"] = {
        threshold_key(threshold): detection_summary(evaluation)
        for threshold, evaluation in threshold_evaluations.items()
    }
    precision_recall_rows = []
    ap_metrics = {}
    for threshold in thresholds:
        ap_result = average_precision(gt, all_detection_pred, threshold)
        curve = ap_result.pop("curve")
        ap_metrics[threshold_key(threshold)] = ap_result
        for row in curve:
            precision_recall_rows.append({"iou_threshold": threshold, **row})
    summary["detection"]["average_precision"] = ap_metrics
    summary["detection"]["by_bbox_size"] = size_breakdown(
        detection_primary,
        metadata["image_width"],
        metadata["image_height"],
    )
    summary["detection"]["by_role"] = role_breakdown(detection_primary)

    display_hota = hota_summary(gt, pred, id_key="pred_identity_id")
    raw_hota = hota_summary(gt, pred, id_key="raw_pred_identity_id")
    add_hota(summary["tracking"], display_hota)
    add_hota(summary["tracking_raw"], raw_hota)

    # GT team is "left"/"right"; pred_team is our own numeric team_id. They
    # are not directly comparable, so IdSim needs the same left/right mapping
    # already computed from matched pairs for the "team" accuracy summary
    # above -- applied here to every predicted row, not just matched ones,
    # since gs_hota_summary scores unmatched rows too.
    team_mapping = choose_team_mapping(matches)
    pred_for_gs_hota = {
        frame: [
            {**row, "pred_team": team_mapping.get(row["pred_team"], row["pred_team"])}
            for row in frame_rows
        ]
        for frame, frame_rows in pred.items()
    }
    display_gs_hota = gs_hota_summary(gt, pred_for_gs_hota, id_key="pred_identity_id")
    raw_gs_hota = gs_hota_summary(gt, pred_for_gs_hota, id_key="raw_pred_identity_id")
    add_gs_hota(summary["tracking"], display_gs_hota)
    add_gs_hota(summary["tracking_raw"], raw_gs_hota)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_matches(output / "frame_matches.csv", matches)
    write_matches(output / "detection_frame_matches.csv", detection_primary["matches"])
    write_rows(output / "frame_metrics.csv", detection_primary["frame_metrics"])
    write_rows(output / "tracking_frame_metrics.csv", tracking_primary["frame_metrics"])
    write_rows(output / "pitch_frame_metrics.csv", pitch_frame_metrics(matches))
    write_rows(output / "precision_recall.csv", precision_recall_rows)
    print(json.dumps(summary, indent=2))


def load_ground_truth(path, allowed_roles=None, pitch_coordinate_system="soccernet_centered"):
    data = json.loads(path.read_text())
    allowed_roles = set(allowed_roles or ATHLETE_ROLES)
    images = data.get("images") or []
    frame_by_image = {}
    for index, image in enumerate(images):
        image_id = str(image.get("image_id", image.get("id")))
        file_name = str(image.get("file_name") or "")
        try:
            frame = int(Path(file_name).stem) - 1
        except ValueError:
            frame = index
        frame_by_image[image_id] = frame

    rows = defaultdict(list)
    for ann in data.get("annotations") or []:
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes") or {}
        role = normalize_role(attrs.get("role"))
        if role not in allowed_roles:
            continue
        bbox = gsr_bbox(ann.get("bbox_image") or {})
        frame = frame_by_image.get(str(ann.get("image_id")))
        if bbox is None or frame is None:
            continue
        pitch = ann.get("bbox_pitch") or {}
        rows[frame].append({
            "gt_track_id": str(ann.get("track_id")),
            "bbox": bbox,
            "gt_role": role,
            "gt_team": normalize_team(attrs.get("team")),
            "gt_jersey": normalize_jersey(attrs.get("jersey")),
            "gt_position_pitch": normalize_gt_pitch_point(
                point(pitch.get("x_bottom_middle"), pitch.get("y_bottom_middle")),
                pitch_coordinate_system,
            ),
        })
    # Retain annotated image frames even when they contain no athlete. This is
    # required for an honest FP-per-frame denominator.
    for frame in frame_by_image.values():
        rows[int(frame)]
    metadata = dict(data.get("info") or {})
    image_width, image_height, image_size_source = infer_image_size(images, rows, metadata)
    metadata.update({
        "image_width": image_width,
        "image_height": image_height,
        "image_size_source": image_size_source,
        "evaluation_roles": sorted(allowed_roles),
        "gt_pitch_coordinate_system_input": pitch_coordinate_system,
        "pitch_coordinate_system_evaluated": "ft_top_left_metres",
    })
    return rows, metadata


def load_predictions(path):
    rows = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group = str(row.get("track_group") or "players")
            if group not in {"players", "referees"}:
                continue
            bbox = parse_list(row.get("bbox"))
            if bbox is None or len(bbox) != 4:
                continue
            frame = int(row["frame"])
            display_track_id = str(row.get("display_track_id") or row.get("raw_track_id"))
            raw_track_id = str(row.get("raw_track_id") or row.get("track_id"))
            rows[frame].append({
                "pred_track_id": display_track_id,
                "raw_pred_track_id": raw_track_id,
                "pred_identity_id": f"{group}:{display_track_id}",
                "raw_pred_identity_id": f"{group}:{raw_track_id}",
                "bbox": [float(v) for v in bbox],
                "pred_role": normalize_role(row.get("role_detection")),
                "pred_team": normalize_pred_team(row.get("team_id")),
                "pred_jersey": normalize_jersey(row.get("jersey_number")),
                "detection_confidence": number(row.get("detection_confidence")),
                "jersey_confidence": number(row.get("jersey_confidence")),
                "pred_position_pitch": parse_list(row.get("position_pitch")),
            })
    return rows


def load_detections(path):
    rows = defaultdict(list)
    with path.open(newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            bbox = parse_list(row.get("bbox"))
            if bbox is None or len(bbox) != 4:
                continue
            frame = int(row["frame"])
            detection_id = str(row.get("detection_id") or row_index)
            rows[frame].append({
                "pred_track_id": f"detection:{frame}:{detection_id}",
                "raw_pred_track_id": f"detection:{frame}:{detection_id}",
                "pred_identity_id": f"detection:{frame}:{detection_id}",
                "raw_pred_identity_id": f"detection:{frame}:{detection_id}",
                "bbox": [float(value) for value in bbox],
                "pred_role": normalize_role(row.get("role_detection")),
                "detection_confidence": number(row.get("detection_confidence")),
            })
    return rows


def filter_detection_confidence(rows, threshold):
    return {
        frame: [
            row
            for row in frame_rows
            if row.get("detection_confidence") is not None
            and float(row["detection_confidence"]) >= float(threshold)
        ]
        for frame, frame_rows in rows.items()
    }


def match_frames(gt, pred, threshold):
    evaluation = evaluate_frames(gt, pred, threshold)
    return evaluation["matches"], evaluation["counts"]


def summarize(matches, counts, gt, pred, metadata, threshold, evaluation=None):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    team_mapping = choose_team_mapping(matches)
    for row in matches:
        row["pred_team_mapped"] = team_mapping.get(row["pred_team"])

    role_known = [row for row in matches if row["gt_role"] is not None]
    team_known = [row for row in matches if row["gt_team"] is not None and row["gt_track_id"]]
    jersey_visible = [row for row in matches if row["gt_jersey"] is not None]
    jersey_emitted = [row for row in jersey_visible if row["pred_jersey"] is not None]
    gt_not_visible = [row for row in matches if row["gt_jersey"] is None]

    evaluation = evaluation or evaluate_frames(gt, pred, threshold)
    display_tracking_evaluation = evaluate_frames(
        gt, pred, threshold, continuity_id_key="pred_identity_id"
    )
    raw_tracking_evaluation = evaluate_frames(
        gt, pred, threshold, continuity_id_key="raw_pred_identity_id"
    )
    track_metrics = mot_tracking_summary(
        display_tracking_evaluation, gt, pred, id_key="pred_identity_id"
    )
    raw_track_metrics = mot_tracking_summary(
        raw_tracking_evaluation, gt, pred, id_key="raw_pred_identity_id"
    )
    pitch_errors = [
        euclidean(row["gt_position_pitch"], row["pred_position_pitch"])
        for row in matches
        if row["gt_position_pitch"] is not None and row["pred_position_pitch"] is not None
    ]

    correct_jersey = sum(row["gt_jersey"] == row["pred_jersey"] for row in jersey_emitted)
    emitted_visible = len(jersey_emitted)
    false_not_visible = sum(row["pred_jersey"] is not None for row in gt_not_visible)
    jersey_tracklets = jersey_tracklet_summary(matches)
    jersey_calibration = confidence_calibration(jersey_emitted)

    return {
        "dataset": {
            "name": metadata.get("name"),
            "version": metadata.get("version"),
            "frames": len(evaluation["frame_metrics"]),
            "frames_with_gt": sum(bool(rows) for rows in gt.values()),
            "image_width": metadata.get("image_width"),
            "image_height": metadata.get("image_height"),
            "image_size_source": metadata.get("image_size_source"),
            "evaluation_roles": metadata.get("evaluation_roles"),
        },
        "matching": {"iou_threshold": threshold, "matched": tp, "gt": counts["gt"], "pred": counts["pred"]},
        "detection": {
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "mean_iou": mean(row["iou"] for row in matches),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "gt": counts["gt"], "pred": counts["pred"],
            "fp_per_frame": ratio(fp, len(evaluation["frame_metrics"])),
            "fn_per_frame": ratio(fn, len(evaluation["frame_metrics"])),
        },
        "tracking": track_metrics,
        "tracking_raw": raw_track_metrics,
        "team": {
            "mapping": {str(k): v for k, v in team_mapping.items()},
            "coverage": ratio(sum(row["pred_team_mapped"] is not None for row in team_known), len(team_known)),
            "accuracy_on_emitted": ratio(sum(row["pred_team_mapped"] == row["gt_team"] for row in team_known if row["pred_team_mapped"] is not None), sum(row["pred_team_mapped"] is not None for row in team_known)),
        },
        "role": {"accuracy": ratio(sum(row["gt_role"] == row["pred_role"] for row in role_known), len(role_known))},
        "jersey": {
            "gt_visible_matches": len(jersey_visible),
            "emitted_on_visible": emitted_visible,
            "coverage_visible": ratio(emitted_visible, len(jersey_visible)),
            "accuracy_on_emitted_visible": ratio(correct_jersey, emitted_visible),
            "accuracy_all_visible": ratio(correct_jersey, len(jersey_visible)),
            "gt_not_visible_matches": len(gt_not_visible),
            "false_emitted_not_visible": false_not_visible,
            "false_positive_rate_not_visible": ratio(false_not_visible, len(gt_not_visible)),
            "tracklet_level": jersey_tracklets,
            "confidence_calibration_on_emitted_visible": jersey_calibration,
        },
        "pitch": {
            "matched_with_coordinates": len(pitch_errors), "coverage": ratio(len(pitch_errors), len(matches)),
            "mean_error": mean(pitch_errors), "median_error": percentile(pitch_errors, 50), "p90_error": percentile(pitch_errors, 90),
            "gt_coordinate_system_input": metadata.get("gt_pitch_coordinate_system_input"),
            "evaluated_coordinate_system": metadata.get("pitch_coordinate_system_evaluated"),
        },
    }


def normalize_gt_pitch_point(value, coordinate_system, pitch_length=105.0, pitch_width=68.0):
    if value is None:
        return None
    x, y = float(value[0]), float(value[1])
    if coordinate_system == "soccernet_centered":
        return [x + pitch_length / 2.0, y + pitch_width / 2.0]
    if coordinate_system == "ft":
        return [x, y]
    raise ValueError(f"Unsupported pitch coordinate system: {coordinate_system}")


def pitch_frame_metrics(matches):
    grouped = defaultdict(list)
    for row in matches:
        gt = row.get("gt_position_pitch")
        pred = row.get("pred_position_pitch")
        if gt is None or pred is None:
            continue
        grouped[int(row["frame"])].append(euclidean(gt, pred))
    return [
        {
            "frame": frame,
            "matched_with_coordinates": len(errors),
            "mean_error_m": mean(errors),
            "median_error_m": percentile(errors, 50),
            "p90_error_m": percentile(errors, 90),
        }
        for frame, errors in sorted(grouped.items())
    ]


def legacy_tracking_summary(matches):
    by_gt = defaultdict(list)
    pred_to_gt = defaultdict(Counter)
    for row in sorted(matches, key=lambda item: (item["gt_track_id"], item["frame"])):
        by_gt[row["gt_track_id"]].append(row)
        pred_to_gt[row["pred_track_id"]][row["gt_track_id"]] += 1
    switches = 0
    fragments = 0
    for rows in by_gt.values():
        ids = [row["pred_track_id"] for row in rows]
        switches += sum(a != b for a, b in zip(ids, ids[1:]))
        fragments += max(0, len(set(ids)) - 1)
    purity_numerator = sum(counter.most_common(1)[0][1] for counter in pred_to_gt.values() if counter)
    return {
        "gt_tracks_matched": len(by_gt), "pred_tracks_matched": len(pred_to_gt),
        "id_switches_proxy": switches, "fragments_proxy": fragments,
        "association_purity": ratio(purity_numerator, len(matches)),
    }


def add_hota(target, hota):
    target.update({
        "hota": hota["hota"],
        "deta": hota["deta"],
        "assa": hota["assa"],
        "loca": hota["loca"],
        "hota_at_0_50": hota["at_0_50"]["hota"],
        "deta_at_0_50": hota["at_0_50"]["deta"],
        "assa_at_0_50": hota["at_0_50"]["assa"],
        "hota_thresholds": hota["thresholds"],
        "hota_per_alpha": hota["per_alpha"],
    })


def add_gs_hota(target, gs_hota):
    """Somers et al. CVPR24 GS-HOTA: pitch-position + role/team/jersey
    identity similarity in place of bounding-box IoU. Requires GT/prediction
    rows to carry gt_position_pitch/pred_position_pitch (metres) and the
    role/team/jersey attribute pairs, as load_ground_truth/load_predictions
    already provide."""
    target.update({
        "gs_hota": gs_hota["gs_hota"],
        "gs_deta": gs_hota["gs_deta"],
        "gs_assa": gs_hota["gs_assa"],
        "gs_loca": gs_hota["loca"],
        "gs_hota_tau_metres": gs_hota["tau_metres"],
        "gs_hota_at_0_50": gs_hota["at_0_50"]["gs_hota"],
        "gs_deta_at_0_50": gs_hota["at_0_50"]["gs_deta"],
        "gs_assa_at_0_50": gs_hota["at_0_50"]["gs_assa"],
        "gs_hota_thresholds": gs_hota["thresholds"],
        "gs_hota_per_alpha": gs_hota["per_alpha"],
    })


def threshold_key(value):
    return f"{float(value):.2f}"


def infer_image_size(images, gt, metadata):
    for image in images:
        width = number(image.get("width") or image.get("image_width"))
        height = number(image.get("height") or image.get("image_height"))
        if width and height:
            return int(width), int(height), "labels.images"
    width = number(metadata.get("width") or metadata.get("image_width"))
    height = number(metadata.get("height") or metadata.get("image_height"))
    if width and height:
        return int(width), int(height), "labels.info"
    # SoccerNet-GSR v1.3 game-state frames are distributed at 1920x1080.
    # Report the fallback explicitly so an unexpected dataset variant is visible.
    return 1920, 1080, "soccernet_gsr_default"


def jersey_tracklet_summary(matches):
    by_gt = defaultdict(list)
    for row in matches:
        by_gt[row["gt_track_id"]].append(row)

    visible = emitted = correct = 0
    not_visible = false_emitted = 0
    rows = []
    for gt_track_id, items in sorted(by_gt.items(), key=lambda item: str(item[0])):
        gt_jersey = mode_value(row["gt_jersey"] for row in items)
        pred_values = [row["pred_jersey"] for row in items if row["pred_jersey"] is not None]
        pred_jersey = mode_value(pred_values) if pred_values else None
        if gt_jersey is None:
            not_visible += 1
            false_emitted += int(pred_jersey is not None)
        else:
            visible += 1
            emitted += int(pred_jersey is not None)
            correct += int(pred_jersey == gt_jersey)
        rows.append({"gt_track_id": gt_track_id, "gt_jersey": gt_jersey, "pred_jersey": pred_jersey})
    return {
        "gt_visible_tracklets": visible,
        "emitted_visible_tracklets": emitted,
        "coverage_visible": ratio(emitted, visible),
        "accuracy_on_emitted_visible": ratio(correct, emitted),
        "accuracy_all_visible": ratio(correct, visible),
        "gt_not_visible_tracklets": not_visible,
        "false_emitted_not_visible_tracklets": false_emitted,
        "false_positive_rate_not_visible": ratio(false_emitted, not_visible),
        "decisions": rows,
    }


def confidence_calibration(rows, bins=10):
    samples = []
    for row in rows:
        confidence = row.get("jersey_confidence")
        if confidence is None:
            continue
        confidence = max(0.0, min(1.0, float(confidence)))
        correct = float(row["gt_jersey"] == row["pred_jersey"])
        samples.append((confidence, correct))
    if not samples:
        return {"samples": 0, "brier": None, "ece": None}
    brier = mean((confidence - correct) ** 2 for confidence, correct in samples)
    ece = 0.0
    bin_rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            (confidence, correct) for confidence, correct in samples
            if low <= confidence < high or (index == bins - 1 and confidence == 1.0)
        ]
        if not selected:
            continue
        avg_confidence = mean(item[0] for item in selected)
        accuracy = mean(item[1] for item in selected)
        ece += len(selected) / len(samples) * abs(avg_confidence - accuracy)
        bin_rows.append({
            "low": low, "high": high, "samples": len(selected),
            "mean_confidence": avg_confidence, "accuracy": accuracy,
        })
    return {"samples": len(samples), "brier": brier, "ece": float(ece), "bins": bin_rows}


def choose_team_mapping(matches):
    pred_values = sorted({row["pred_team"] for row in matches if row["pred_team"] is not None}, key=str)
    if not pred_values:
        return {}
    candidates = [{pred_values[0]: "left"}, {pred_values[0]: "right"}]
    if len(pred_values) >= 2:
        candidates = [
            {pred_values[0]: "left", pred_values[1]: "right"},
            {pred_values[0]: "right", pred_values[1]: "left"},
        ]
    return max(candidates, key=lambda mapping: sum(mapping.get(row["pred_team"]) == row["gt_team"] for row in matches))


def write_matches(path, matches):
    rows = []
    for row in matches:
        out = dict(row)
        for key in ("bbox", "gt_position_pitch", "pred_position_pitch"):
            if key in out:
                out[key] = json.dumps(out[key])
        rows.append(out)
    write_rows(path, rows)


def write_rows(path, rows):
    fields = sorted({key for row in rows for key in row})
    if not fields:
        Path(path).write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def gsr_bbox(bbox):
    try:
        w, h = float(bbox["w"]), float(bbox["h"])
        x = float(bbox["x"]) if "x" in bbox else float(bbox["x_center"]) - w / 2
        y = float(bbox["y"]) if "y" in bbox else float(bbox["y_center"]) - h / 2
        return [x, y, x + w, y + h]
    except (KeyError, TypeError, ValueError):
        return None


def bbox_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_list(value):
    if value in EMPTY:
        return None
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
    except Exception:
        try: result = ast.literal_eval(value)
        except Exception: return None
    return result if isinstance(result, list) else None


def normalize_role(value):
    value = str(value or "").lower()
    return {"keeper": "goalkeeper", "gk": "goalkeeper"}.get(value, value or None)


def normalize_team(value):
    value = str(value or "").lower()
    return value if value in {"left", "right"} else None


def normalize_pred_team(value):
    if value in EMPTY: return None
    try: return int(float(value))
    except (TypeError, ValueError): return str(value)


def normalize_jersey(value):
    if value in EMPTY: return None
    try:
        number_value = int(float(value))
        return number_value if 1 <= number_value <= 99 else None
    except (TypeError, ValueError): return None


def number(value):
    if value in EMPTY: return None
    try: return float(value)
    except (TypeError, ValueError): return None


def point(x, y):
    try: return [float(x), float(y)]
    except (TypeError, ValueError): return None


def ratio(a, b): return float(a / b) if b else None
def mean(values):
    values = list(values); return float(sum(values) / len(values)) if values else None
def percentile(values, q): return float(np.percentile(values, q)) if values else None
def euclidean(a, b): return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
def mode_value(values):
    values = list(values)
    if not values: return None
    return Counter(values).most_common(1)[0][0]


if __name__ == "__main__":
    main()
