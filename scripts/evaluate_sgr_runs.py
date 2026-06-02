#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate FT runs against normalized SoccerNet-GSR detections.")
    parser.add_argument("--sequence", required=True, help="SoccerNet-GSR sequence id, e.g. SNGS-151")
    parser.add_argument("--gt-jsonl", required=True, type=Path)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/costume-video"))
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec label=run_name[:video_id]. If video_id is omitted, the only *_tracklets.csv is used.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    gt_by_frame = load_gt(args.gt_jsonl, args.sequence)
    results = []
    for spec in args.run:
        label, run_name, video_id = parse_run_spec(spec)
        result = evaluate_run(
            label=label,
            run_name=run_name,
            video_id=video_id,
            artifacts_root=args.artifacts_root,
            gt_by_frame=gt_by_frame,
            iou_threshold=args.iou_threshold,
        )
        results.append(result)
        print_result(result)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, results)


def parse_run_spec(spec):
    if "=" in spec:
        label, rest = spec.split("=", 1)
    else:
        label, rest = spec, spec
    parts = rest.split(":")
    run_name = parts[0]
    video_id = parts[1] if len(parts) > 1 else None
    return label, run_name, video_id


def load_gt(path, sequence):
    gt_by_frame = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("sequence") != sequence:
                continue
            if str(row.get("role") or "").lower() not in {"player", "goalkeeper"}:
                continue
            gt_by_frame[int(row["frame"]) - 1].append(row)
    return gt_by_frame


def evaluate_run(label, run_name, video_id, artifacts_root, gt_by_frame, iou_threshold):
    metadata = artifacts_root / run_name / "metadata"
    tracklets_path = resolve_tracklets_path(metadata, video_id)
    rows = list(csv.DictReader(tracklets_path.open()))
    preds = [row for row in rows if row.get("track_group", "players") == "players"]
    matches = match_predictions(preds, gt_by_frame, iou_threshold)

    team_name, team_mapping, team_ok = best_team_mapping(matches)
    gt_with_number = [(pred, gt) for pred, gt in matches if known(gt_jersey(gt))]
    known_pred = [(pred, gt) for pred, gt in gt_with_number if known(pred_jersey(pred))]
    correct = [(pred, gt) for pred, gt in known_pred if pred_jersey(pred) == gt_jersey(gt)]
    wrong = [(pred, gt) for pred, gt in known_pred if pred_jersey(pred) != gt_jersey(gt)]
    candidates = [(pred, gt) for pred, gt in gt_with_number if known(pred.get("candidate_player_id"))]
    candidate_correct = [
        (pred, gt)
        for pred, gt in candidates
        if pred.get("candidate_player_id") == gt_identity(gt, team_mapping)
    ]

    ocr = load_optional_json(metadata / f"{tracklets_path.stem.replace('_tracklets', '')}_jersey_ocr.json")
    gk_matches = [(pred, gt) for pred, gt in matches if str(gt.get("role") or "").lower() == "goalkeeper"]
    result = {
        "label": label,
        "run": run_name,
        "video_id": tracklets_path.stem.replace("_tracklets", ""),
        "pred_rows": len(preds),
        "matches": len(matches),
        "match_rate": ratio(len(matches), len(preds)),
        "best_team_mapping": team_name,
        "team_accuracy": ratio(team_ok, len(matches)),
        "team_correct": team_ok,
        "team_total": len(matches),
        "jersey_recall_accuracy": ratio(len(correct), len(gt_with_number)),
        "jersey_correct": len(correct),
        "jersey_total": len(gt_with_number),
        "known_pred_precision": ratio(len(correct), len(known_pred)),
        "known_pred_correct": len(correct),
        "known_pred_total": len(known_pred),
        "wrong_known": len(wrong),
        "unknown_jersey": len(gt_with_number) - len(known_pred),
        "candidate_accuracy": ratio(len(candidate_correct), len(candidates)),
        "candidate_correct": len(candidate_correct),
        "candidate_total": len(candidates),
        "pred_jerseys": Counter(pred_jersey(row) for row in preds).most_common(20),
        "top_confusions": Counter((gt_jersey(gt), pred_jersey(pred)) for pred, gt in gt_with_number).most_common(20),
        "top_wrong": Counter((gt_jersey(gt), pred_jersey(pred)) for pred, gt in wrong).most_common(20),
        "gk_confusions": Counter((gt_jersey(gt), pred_jersey(pred)) for pred, gt in gk_matches).most_common(20),
        "ocr_number_roi": ocr.get("number_roi"),
        "ocr_mmocr": ocr.get("mmocr"),
        "ocr_cache": ocr.get("cache"),
        "goalkeeper_ocr_filter": ocr.get("goalkeeper_ocr_filter"),
    }
    return result


def resolve_tracklets_path(metadata, video_id):
    if video_id:
        path = metadata / f"{video_id}_tracklets.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(metadata.glob("*_tracklets.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one *_tracklets.csv in {metadata}, found {len(candidates)}")
    return candidates[0]


def match_predictions(preds, gt_by_frame, iou_threshold):
    matches = []
    for pred in preds:
        frame = int(pred["frame"])
        best = None
        best_iou = 0.0
        for gt in gt_by_frame.get(frame, []):
            score = iou(pred_box(pred), gt_box(gt))
            if score > best_iou:
                best_iou = score
                best = gt
        if best is not None and best_iou >= iou_threshold:
            matches.append((pred, best))
    return matches


def best_team_mapping(matches):
    mappings = [
        ("normal", {"left": "1", "right": "2", "1": "1", "2": "2", "team1": "1", "team2": "2"}),
        ("swapped", {"left": "2", "right": "1", "1": "2", "2": "1", "team1": "2", "team2": "1"}),
    ]
    best = ("normal", mappings[0][1], 0)
    for name, mapping in mappings:
        correct = sum(1 for pred, gt in matches if str(pred.get("team_id")) == map_team(gt_team(gt), mapping))
        if correct > best[2]:
            best = (name, mapping, correct)
    return best


def pred_box(row):
    value = row["bbox"]
    if isinstance(value, str):
        value = json.loads(value)
    return [float(item) for item in value]


def gt_box(row):
    value = row.get("bbox_image") or row.get("bbox")
    if isinstance(value, str):
        value = json.loads(value)
    return [float(item) for item in value]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    den = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / den if den else 0.0


def known(value):
    return value not in ("", "None", "unknown", None)


def pred_jersey(row):
    return str(row.get("jersey_number") or "unknown")


def gt_jersey(row):
    return str(row.get("jersey_number") or row.get("jersey") or row.get("number") or "unknown")


def gt_team(row):
    return str(row.get("team") or row.get("team_id") or row.get("side") or "").lower()


def map_team(value, mapping):
    return mapping.get(str(value).lower(), str(value))


def gt_identity(row, team_mapping):
    team = map_team(gt_team(row), team_mapping)
    jersey = gt_jersey(row)
    if team in {"1", "2"} and known(jersey):
        return f"team{team}_{int(jersey):02d}"
    return None


def load_optional_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(num, den):
    return round(num / den, 6) if den else 0.0


def print_result(result):
    print("\n" + "=" * 100)
    print(result["label"], result["run"])
    for key in (
        "pred_rows",
        "matches",
        "match_rate",
        "best_team_mapping",
        "team_accuracy",
        "jersey_recall_accuracy",
        "known_pred_precision",
        "wrong_known",
        "unknown_jersey",
        "candidate_accuracy",
    ):
        print(key, result[key])
    print("pred_jerseys", result["pred_jerseys"])
    print("top_confusions", result["top_confusions"])
    print("top_wrong", result["top_wrong"])
    print("gk_confusions", result["gk_confusions"])


def write_csv(path, results):
    fields = [
        "label",
        "run",
        "video_id",
        "pred_rows",
        "matches",
        "match_rate",
        "best_team_mapping",
        "team_accuracy",
        "jersey_recall_accuracy",
        "known_pred_precision",
        "wrong_known",
        "unknown_jersey",
        "candidate_accuracy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})


if __name__ == "__main__":
    main()
