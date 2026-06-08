#!/usr/bin/env python3
"""Audit unknown player display IDs and their OCR/crop diagnostics."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {"", "None", "unknown", None}


def nonempty(value):
    return value not in EMPTY


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tracklets(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_player_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        display_id = str(row.get("display_track_id") or row.get("track_id"))
        grouped[display_id].append(row)
    return grouped


def unknown_display_groups(grouped):
    out = {}
    for display_id, rows in grouped.items():
        if not rows:
            continue
        all_unknown_identity = all(not nonempty(row.get("player_id")) for row in rows)
        all_unknown_jersey = all(not nonempty(row.get("jersey_number")) for row in rows)
        if all_unknown_identity and all_unknown_jersey:
            out[display_id] = rows
    return out


def ocr_entries_for_display(ocr, display_id):
    tracklets = ocr.get("tracklets") or {}
    exact = tracklets.get(str(display_id))
    if exact:
        return [exact]
    prefix = f"{display_id}:"
    return [value for key, value in tracklets.items() if str(key).startswith(prefix)]


def summarize_ocr_entries(entries):
    statuses = Counter()
    raw_reasons = Counter()
    voting_reasons = Counter()
    raw_numbers = Counter()
    voting_numbers = Counter()
    raw_detection_count = 0
    voting_detection_count = 0
    selected_crops = 0
    usable_crops = 0
    available_crops = 0

    for entry in entries:
        decision = entry.get("decision") or {}
        statuses[str(decision.get("status") or "missing")] += 1
        raw_reasons.update(as_counter(decision.get("raw_rejection_reasons")))
        voting_reasons.update(as_counter(decision.get("voting_rejection_reasons")))
        raw_numbers.update(as_counter(decision.get("raw_number_counts")))
        voting_numbers.update(as_counter(decision.get("voting_number_counts")))
        raw_detection_count += int(entry.get("raw_detection_count") or 0)
        voting_detection_count += int(entry.get("voting_detection_count") or 0)
        crop_selection = entry.get("crop_selection") or {}
        selected_crops += int(crop_selection.get("selected") or 0)
        usable_crops += int(crop_selection.get("usable") or 0)
        available_crops += int(crop_selection.get("available") or entry.get("available_crops") or 0)

    return {
        "ocr_attempted": bool(entries),
        "ocr_statuses": dict(statuses),
        "ocr_primary_status": statuses.most_common(1)[0][0] if statuses else "not_attempted",
        "ocr_raw_detection_count": raw_detection_count,
        "ocr_voting_detection_count": voting_detection_count,
        "ocr_selected_crops": selected_crops,
        "ocr_usable_crops": usable_crops,
        "ocr_available_crops": available_crops,
        "raw_rejection_reasons": dict(raw_reasons),
        "voting_rejection_reasons": dict(voting_reasons),
        "raw_number_counts": dict(raw_numbers),
        "voting_number_counts": dict(voting_numbers),
    }


def as_counter(value):
    if not value:
        return Counter()
    if isinstance(value, Counter):
        return value
    if isinstance(value, dict):
        out = Counter()
        for key, count in value.items():
            try:
                out[str(key)] += int(count)
            except Exception:
                continue
        return out
    return Counter()


def summarize_display(display_id, rows, ocr):
    frames = [int(row.get("frame", 0) or 0) for row in rows]
    qualities = [float(row.get("crop_quality", 0.0) or 0.0) for row in rows]
    sorted_by_quality = sorted(rows, key=lambda row: float(row.get("crop_quality", 0.0) or 0.0), reverse=True)
    ocr_summary = summarize_ocr_entries(ocr_entries_for_display(ocr, display_id))
    mean_quality = sum(qualities) / len(qualities) if qualities else 0.0
    best_quality = max(qualities) if qualities else 0.0
    result = {
        "display_track_id": display_id,
        "n_frames": len(rows),
        "start_frame": min(frames) if frames else None,
        "end_frame": max(frames) if frames else None,
        "mean_crop_quality": round(mean_quality, 6),
        "max_crop_quality": round(best_quality, 6),
        "quality_bucket": quality_bucket(mean_quality),
        "role_counts": dict(Counter(str(row.get("role_detection") or "") for row in rows)),
        "team_counts": dict(Counter(str(row.get("team_id") or "unknown") for row in rows)),
        "best_crop": sorted_by_quality[0].get("crop_path") if sorted_by_quality else None,
        "sample_crops": [
            row.get("crop_path")
            for row in sorted_by_quality[:5]
            if row.get("crop_path")
        ],
    }
    result.update(ocr_summary)
    return result


def quality_bucket(mean_quality):
    if mean_quality > 0.30:
        return "high"
    if mean_quality > 0.10:
        return "mid"
    return "low"


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(results, top):
    print("unknown_display_ids", len(results))
    print("quality_buckets", Counter(row["quality_bucket"] for row in results).most_common())
    print("ocr_attempted", sum(1 for row in results if row["ocr_attempted"]))
    print("ocr_not_attempted", sum(1 for row in results if not row["ocr_attempted"]))
    print("ocr_statuses", Counter(row["ocr_primary_status"] for row in results).most_common())

    interesting = [
        row
        for row in results
        if row["quality_bucket"] == "high"
        and row["ocr_attempted"]
        and row["ocr_primary_status"] != "assigned"
    ]
    print("high_quality_ocr_attempted_not_assigned", len(interesting))
    for row in sorted(interesting, key=lambda item: (-int(item["n_frames"]), -float(item["mean_crop_quality"])))[:top]:
        print(
            "display",
            row["display_track_id"],
            "frames",
            row["n_frames"],
            "quality",
            row["mean_crop_quality"],
            "raw_det",
            row["ocr_raw_detection_count"],
            "voting_det",
            row["ocr_voting_detection_count"],
            "status",
            row["ocr_primary_status"],
        )


def write_crop_sheets(results, output_dir, samples_per_bucket=24, thumb_size=(96, 160)):
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        print(f"crop_sheets skipped: {type(exc).__name__}: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for bucket in ("high", "mid", "low"):
        bucket_rows = [row for row in results if row["quality_bucket"] == bucket and row.get("best_crop")]
        bucket_rows = sorted(bucket_rows, key=lambda row: (-int(row["n_frames"]), -float(row["mean_crop_quality"])))
        thumbs = []
        for row in bucket_rows[:samples_per_bucket]:
            path = Path(row["best_crop"])
            image = cv2.imread(str(path))
            if image is None:
                continue
            image = cv2.resize(image, thumb_size)
            label = f"id {row['display_track_id']} f{row['n_frames']} q{float(row['mean_crop_quality']):.2f}"
            cv2.putText(image, label[:32], (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
            thumbs.append(image)
        if not thumbs:
            continue
        rows = []
        columns = 8
        for index in range(0, len(thumbs), columns):
            row_images = thumbs[index : index + columns]
            while len(row_images) < columns:
                row_images.append(np.zeros_like(thumbs[0]))
            rows.append(np.hstack(row_images))
        sheet = np.vstack(rows)
        out_path = output_dir / f"unknown_{bucket}_sheet.jpg"
        cv2.imwrite(str(out_path), sheet)
        print("wrote", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/costume-video"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--sheet-samples", type=int, default=32)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    metadata_dir = args.artifacts_root / args.run / "metadata"
    tracklets_path = metadata_dir / f"{args.video_id}_tracklets.csv"
    ocr_path = metadata_dir / f"{args.video_id}_jersey_ocr.json"
    if not tracklets_path.exists():
        raise FileNotFoundError(tracklets_path)

    rows = load_tracklets(tracklets_path)
    ocr = load_json(ocr_path)
    grouped = group_player_rows(rows)
    unknown_groups = unknown_display_groups(grouped)
    results = [
        summarize_display(display_id, display_rows, ocr)
        for display_id, display_rows in sorted(unknown_groups.items(), key=lambda item: int(item[0]))
    ]

    output_dir = args.output_dir or Path("evaluation/realvideo_unknown_audit") / args.run
    write_csv(results, output_dir / f"{args.video_id}_unknown_displays.csv")
    write_json(results, output_dir / f"{args.video_id}_unknown_displays.json")
    if not args.no_sheets:
        write_crop_sheets(results, output_dir / "sheets", samples_per_bucket=args.sheet_samples)
    print_summary(results, args.top)
    print("output_dir", output_dir)


if __name__ == "__main__":
    main()
