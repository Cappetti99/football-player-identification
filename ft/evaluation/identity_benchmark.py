import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {None, "", "None", "unknown"}
IDENTITY_LABEL_FIELDS = (
    "annotation_status",
    "gt_player_id",
    "gt_team_id",
    "gt_jersey_number",
    "jersey_visibility",
    "uncertainty_reason",
    "notes",
)
PAIR_LABEL_FIELDS = ("pair_label", "uncertainty_reason", "notes")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [normalize_row(row) for row in csv.DictReader(handle)]


def write_csv(rows, path, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (list(rows[0]) if rows else ["item_id"]))
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def normalize_row(row):
    output = dict(row)
    for key in (
        "bbox",
        "identity_evidence",
        "identity_sources",
        "identity_risk_flags",
        "crop_paths",
        "anchors",
    ):
        value = output.get(key)
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                output[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return output


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bbox_iou(left, right):
    left = parse_bbox(left)
    right = parse_bbox(right)
    if left is None or right is None:
        return 0.0
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def parse_bbox(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def identity_group_key(row):
    tracklet = row.get("identity_tracklet_id")
    if tracklet in EMPTY:
        tracklet = row.get("display_track_id", row.get("track_id"))
    segment = row.get("scene_segment_id")
    return str(tracklet), "" if segment in EMPTY else str(segment)


def group_identity_units(rows, run, video_id, split, max_frames=1200):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        frame = to_int(row.get("frame"))
        if frame is None or frame < 0 or frame >= int(max_frames):
            continue
        if parse_bbox(row.get("bbox")) is None:
            continue
        grouped[identity_group_key(row)].append(row)
    units = []
    for (tracklet_id, scene_id), items in sorted(grouped.items()):
        items.sort(key=lambda item: (int(item["frame"]), int(item.get("raw_track_id") or 0)))
        strata = classify_unit_strata(items)
        units.append({
            "member_id": f"{run}:{tracklet_id}:{scene_id}",
            "run": run,
            "video_id": video_id,
            "split": split,
            "identity_tracklet_id": tracklet_id,
            "scene_segment_id": scene_id,
            "display_track_ids": sorted({str(item.get("display_track_id")) for item in items}),
            "start_frame": int(items[0]["frame"]),
            "end_frame": int(items[-1]["frame"]),
            "frames": frame_map(items),
            "rows": items,
            "strata": strata,
        })
    return units


def classify_unit_strata(rows):
    statuses = {str(row.get("identity_status") or "unknown").lower() for row in rows}
    evidence_statuses = {
        str((row.get("identity_evidence") or {}).get("status") or "").lower()
        for row in rows
        if isinstance(row.get("identity_evidence"), dict)
    }
    strata = set()
    if "propagated" in statuses | evidence_statuses:
        strata.add("propagated")
    if statuses.intersection({"invalidated", "cleared"}):
        strata.add("constraint_invalidated")
    if any(row.get("player_id") not in EMPTY for row in rows):
        strata.add("direct_assigned" if "propagated" not in strata else "assigned")
    else:
        strata.add("unknown")
    confidences = [to_float(row.get("identity_confidence"), 0.0) for row in rows]
    confidence = sum(confidences) / max(1, len(confidences))
    strata.add("confidence_low" if confidence < 0.4 else "confidence_mid" if confidence < 0.75 else "confidence_high")
    return sorted(strata)


def frame_map(rows):
    output = {}
    for row in rows:
        frame = int(row["frame"])
        current = output.get(frame)
        if current is None or to_float(row.get("crop_quality"), 0.0) > to_float(current.get("crop_quality"), 0.0):
            output[frame] = row
    return output


def units_match(left, right, iou_threshold=0.5, min_shared_frames=3, min_overlap_fraction=0.2):
    shared = sorted(set(left["frames"]).intersection(right["frames"]))
    matched = sum(
        bbox_iou(left["frames"][frame].get("bbox"), right["frames"][frame].get("bbox")) >= float(iou_threshold)
        for frame in shared
    )
    shorter = min(len(left["frames"]), len(right["frames"]))
    required = min(int(min_shared_frames), max(1, shorter))
    return matched >= required and matched / max(1, shorter) >= float(min_overlap_fraction)


def merge_identity_units(units, anchors_per_unit=8, **match_args):
    units = sorted(units, key=lambda row: row["member_id"])
    parent = list(range(len(units)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(units)):
        for right in range(left + 1, len(units)):
            if units[left]["video_id"] != units[right]["video_id"]:
                continue
            if units[left]["run"] == units[right]["run"]:
                continue
            if units_match(units[left], units[right], **match_args):
                union(left, right)

    components = defaultdict(list)
    for index, unit in enumerate(units):
        components[find(index)].append(unit)
    output = []
    for members in components.values():
        members.sort(key=lambda row: row["member_id"])
        representative = max(members, key=lambda row: (len(row["frames"]), row["member_id"]))
        anchors = representative_anchors(representative["rows"], anchors_per_unit)
        stable = {
            "video_id": representative["video_id"],
            "anchors": [{"frame": row["frame"], "bbox": row["bbox"]} for row in anchors],
        }
        unit_id = f"{representative['video_id']}_unit_{canonical_hash(stable)[:12]}"
        output.append({
            "item_id": unit_id,
            "item_type": "identity",
            "video_id": representative["video_id"],
            "split": representative["split"],
            "start_frame": min(member["start_frame"] for member in members),
            "end_frame": max(member["end_frame"] for member in members),
            "strata": sorted({value for member in members for value in member["strata"]}),
            "members": [
                {
                    "run": member["run"],
                    "member_id": member["member_id"],
                    "identity_tracklet_id": member["identity_tracklet_id"],
                    "scene_segment_id": member["scene_segment_id"],
                    "display_track_ids": member["display_track_ids"],
                }
                for member in members
            ],
            "anchors": [anchor_payload(row) for row in anchors],
        })
    return sorted(output, key=lambda row: (row["video_id"], row["start_frame"], row["item_id"]))


def representative_anchors(rows, limit):
    by_frame = list(frame_map(rows).values())
    by_frame.sort(key=lambda row: int(row["frame"]))
    if len(by_frame) <= int(limit):
        return by_frame
    indices = {
        round(index * (len(by_frame) - 1) / (int(limit) - 1))
        for index in range(int(limit))
    } if int(limit) > 1 else {len(by_frame) // 2}
    return [by_frame[index] for index in sorted(indices)]


def anchor_payload(row):
    return {
        "frame": int(row["frame"]),
        "bbox": parse_bbox(row.get("bbox")),
        "crop_path": row.get("crop_path"),
        "crop_quality": to_float(row.get("crop_quality"), 0.0),
    }


def assign_second_review(items, fraction=0.2, seed=20260702):
    grouped = defaultdict(list)
    for item in items:
        strata = set(item.get("strata") or {"unknown"})
        assignment = next(
            (
                value for value in (
                    "propagated",
                    "constraint_invalidated",
                    "direct_assigned",
                    "unknown",
                    "assigned",
                )
                if value in strata
            ),
            sorted(strata)[0],
        )
        confidence = next(
            (value for value in ("confidence_low", "confidence_mid", "confidence_high") if value in strata),
            "",
        )
        grouped[(item["item_type"], item["video_id"], assignment, confidence)].append(item)
    target = min(len(items), math.ceil(len(items) * float(fraction)))
    allocation = {
        group: min(len(rows), math.floor(len(rows) * float(fraction)))
        for group, rows in grouped.items()
    }
    remaining = target - sum(allocation.values())
    order = sorted(
        grouped,
        key=lambda group: (
            -(len(grouped[group]) * float(fraction) - allocation[group]),
            group,
        ),
    )
    while remaining > 0:
        progressed = False
        for group in order:
            if allocation[group] >= len(grouped[group]):
                continue
            allocation[group] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    selected = set()
    for group, rows in grouped.items():
        ranked = sorted(
            rows,
            key=lambda row: canonical_hash({"seed": int(seed), "item_id": row["item_id"]}),
        )
        selected.update(row["item_id"] for row in ranked[:allocation[group]])
    for item in items:
        item["second_review_required"] = item["item_id"] in selected
    return items


def annotation_rows(items, reviewer, second_only=False):
    rows = []
    for item in items:
        if second_only and not item.get("second_review_required"):
            continue
        rows.append({
            "item_id": item["item_id"],
            "item_type": item["item_type"],
            "video_id": item["video_id"],
            "split": item["split"],
            "reviewer": reviewer,
            "second_review_required": bool(item.get("second_review_required")),
            "annotation_status": "",
            "gt_player_id": "",
            "gt_team_id": "",
            "gt_jersey_number": "",
            "jersey_visibility": "",
            "pair_label": "",
            "uncertainty_reason": "",
            "notes": "",
        })
    return rows


def annotation_signature(row):
    if row.get("item_type") == "pair":
        return tuple(str(row.get(key) or "").strip() for key in PAIR_LABEL_FIELDS[:-1])
    return tuple(str(row.get(key) or "").strip() for key in IDENTITY_LABEL_FIELDS[:-1])


def validate_annotation(row):
    item_type = row.get("item_type")
    if item_type == "pair":
        if row.get("pair_label") not in {"same", "different", "uncertain", "exclude"}:
            return "pair_label must be same, different, uncertain or exclude"
        return None
    status = row.get("annotation_status")
    if status not in {"determinate", "not_determinable", "exclude"}:
        return "annotation_status must be determinate, not_determinable or exclude"
    if status == "determinate" and row.get("gt_player_id") in EMPTY:
        return "determinate identity requires gt_player_id"
    if status in {"determinate", "not_determinable"} and row.get("jersey_visibility") not in {"full", "partial", "not_visible"}:
        return "jersey_visibility must be full, partial or not_visible"
    if status in {"not_determinable", "exclude"} and not str(row.get("uncertainty_reason") or "").strip():
        return f"{status} identity requires uncertainty_reason"
    return None


def cohen_kappa(left, right):
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left).union(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right))
        for label in labels
    )
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def adjudicate(reviewer_a, reviewer_b, adjudication=None):
    a_by_id = {row["item_id"]: row for row in reviewer_a}
    b_by_id = {row["item_id"]: row for row in reviewer_b}
    resolved_by_id = {row["item_id"]: row for row in (adjudication or [])}
    final = []
    disagreements = []
    compared = []
    for item_id, left in sorted(a_by_id.items()):
        error = validate_annotation(left)
        if error:
            raise ValueError(f"{item_id} reviewer A: {error}")
        if not truthy(left.get("second_review_required")):
            final.append(left)
            continue
        right = b_by_id.get(item_id)
        if right is None:
            raise ValueError(f"missing reviewer B label for {item_id}")
        error = validate_annotation(right)
        if error:
            raise ValueError(f"{item_id} reviewer B: {error}")
        compared.append((left, right))
        if annotation_signature(left) == annotation_signature(right):
            final.append(left)
            continue
        disagreements.append({"item_id": item_id, "reviewer_a": left, "reviewer_b": right})
        resolved = resolved_by_id.get(item_id)
        if resolved is not None:
            error = validate_annotation(resolved)
            if error:
                raise ValueError(f"{item_id} adjudication: {error}")
            final.append(resolved)
    report = agreement_report(compared, disagreements)
    return final, disagreements, report


def agreement_report(compared, disagreements):
    identity = [(a, b) for a, b in compared if a.get("item_type") == "identity"]
    pairs = [(a, b) for a, b in compared if a.get("item_type") == "pair"]
    determinate_a = [a.get("annotation_status") for a, _ in identity]
    determinate_b = [b.get("annotation_status") for _, b in identity]
    pair_a = [a.get("pair_label") for a, _ in pairs]
    pair_b = [b.get("pair_label") for _, b in pairs]
    both_determinate = [
        (a, b) for a, b in identity
        if a.get("annotation_status") == b.get("annotation_status") == "determinate"
    ]
    return {
        "double_reviewed": len(compared),
        "disagreements": len(disagreements),
        "determination_agreement": ratio(sum(a == b for a, b in zip(determinate_a, determinate_b)), len(identity)),
        "determination_kappa": cohen_kappa(determinate_a, determinate_b),
        "exact_identity_agreement": ratio(
            sum(a.get("gt_player_id") == b.get("gt_player_id") for a, b in both_determinate),
            len(both_determinate),
        ),
        "pair_agreement": ratio(sum(a == b for a, b in zip(pair_a, pair_b)), len(pairs)),
        "pair_kappa": cohen_kappa(pair_a, pair_b),
    }


def match_anchors(anchors, candidate_rows, iou_threshold=0.5, ambiguity_margin=0.05):
    by_frame = defaultdict(list)
    for row in candidate_rows:
        if row.get("track_group", "players") == "players" or "track_group" not in row:
            frame = to_int(row.get("frame"))
            if frame is not None:
                by_frame[frame].append(row)
    matches = []
    for anchor in anchors:
        ranked = sorted(
            (
                (bbox_iou(anchor.get("bbox"), row.get("bbox")), row)
                for row in by_frame.get(int(anchor["frame"]), [])
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_iou = ranked[0][0] if ranked else 0.0
        second_iou = ranked[1][0] if len(ranked) > 1 else 0.0
        ambiguous = best_iou >= iou_threshold and best_iou - second_iou < float(ambiguity_margin)
        matches.append({
            "frame": int(anchor["frame"]),
            "matched": bool(ranked and best_iou >= float(iou_threshold) and not ambiguous),
            "ambiguous": bool(ambiguous),
            "iou": float(best_iou),
            "row": ranked[0][1] if ranked and best_iou >= float(iou_threshold) and not ambiguous else None,
        })
    return matches


def evaluate_identity_units(manifest, ground_truth, rows_by_video, iou_threshold=0.5, ambiguity_margin=0.05):
    gt_by_id = {row["item_id"]: row for row in ground_truth if row.get("item_type") == "identity"}
    results = []
    for unit in manifest.get("identity_units", []):
        gt = gt_by_id.get(unit["item_id"])
        if not gt:
            continue
        matches = match_anchors(
            unit["anchors"],
            rows_by_video.get(unit["video_id"], []),
            iou_threshold=iou_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        matched_rows = [item["row"] for item in matches if item["matched"]]
        assigned = [row for row in matched_rows if row.get("player_id") not in EMPTY]
        gt_player = str(gt.get("gt_player_id") or "")
        determinate = gt.get("annotation_status") == "determinate"
        correct_frames = sum(str(row.get("player_id")) == gt_player for row in assigned) if determinate else 0
        wrong_frames = len(assigned) - correct_frames if determinate else 0
        results.append({
            "item_id": unit["item_id"],
            "video_id": unit["video_id"],
            "split": unit["split"],
            "determinate": determinate,
            "excluded": gt.get("annotation_status") == "exclude",
            "gt_player_id": gt_player,
            "anchors": len(matches),
            "matched_anchors": len(matched_rows),
            "ambiguous_anchors": sum(item["ambiguous"] for item in matches),
            "assigned_anchors": len(assigned),
            "correct_anchors": correct_frames,
            "wrong_anchors": wrong_frames,
            "unit_assigned": bool(assigned),
            "unit_correct": bool(determinate and assigned and wrong_frames == 0),
            "unit_wrong": bool(determinate and wrong_frames > 0),
            "unit_confidence": mean(to_float(row.get("identity_confidence"), 0.0) for row in assigned),
            "source": identity_source(assigned),
            "predicted_player_ids": sorted({str(row.get("player_id")) for row in assigned}),
        })
    return results


def identity_source(rows):
    if any(row.get("jersey_link_previous_display_track_id") not in EMPTY for row in rows):
        return "jersey_identity_linker"
    statuses = {str(row.get("identity_status") or "").lower() for row in rows}
    evidence = [row.get("identity_evidence") for row in rows if isinstance(row.get("identity_evidence"), dict)]
    sources = statuses | {str(item.get("source") or item.get("status") or "").lower() for item in evidence}
    for name in ("prtreid_bridge", "propagated", "segment_candidate", "jersey_identity_linker"):
        if name in sources:
            return name
    return "direct" if rows else "unknown"


def identity_metrics(results):
    usable = [row for row in results if row["determinate"] and not row["excluded"]]
    assigned_units = [row for row in usable if row["unit_assigned"]]
    assigned_frames = sum(row["assigned_anchors"] for row in usable)
    correct_frames = sum(row["correct_anchors"] for row in usable)
    total_anchors = sum(row["anchors"] for row in usable)
    correct_units = sum(row["unit_correct"] for row in usable)
    metrics = {
        "total_labeled_units": len(results),
        "determinate_units": len(usable),
        "not_determinable_units": sum(not row["determinate"] and not row["excluded"] for row in results),
        "excluded_units": sum(row["excluded"] for row in results),
        "assigned_units": len(assigned_units),
        "correct_units": int(correct_units),
        "wrong_units": sum(row["unit_wrong"] for row in usable),
        "identity_precision_unit": ratio(correct_units, len(assigned_units)),
        "identity_precision_unit_ci95": wilson_interval(correct_units, len(assigned_units)),
        "identity_precision_frame": ratio(correct_frames, assigned_frames),
        "identity_precision_frame_ci95": wilson_interval(correct_frames, assigned_frames),
        "correct_coverage": ratio(correct_units, len(usable)),
        "assignment_coverage_frame": ratio(assigned_frames, total_anchors),
        "correct_coverage_frame": ratio(correct_frames, total_anchors),
        "abstention_rate": ratio(sum(not row["unit_assigned"] for row in usable), len(usable)),
        "anchor_match_rate": ratio(sum(row["matched_anchors"] for row in usable), sum(row["anchors"] for row in usable)),
        "ambiguous_anchor_rate": ratio(sum(row["ambiguous_anchors"] for row in usable), sum(row["anchors"] for row in usable)),
        "by_source": {},
        "risk_coverage_curve": [],
    }
    for source in sorted({row["source"] for row in assigned_units}):
        selected = [row for row in assigned_units if row["source"] == source]
        correct = sum(row["unit_correct"] for row in selected)
        metrics["by_source"][source] = {
            "assigned_units": len(selected),
            "correct_units": int(correct),
            "precision": ratio(correct, len(selected)),
            "ci95": wilson_interval(correct, len(selected)),
        }
    for threshold in (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        selected = [row for row in usable if row["unit_assigned"] and row["unit_confidence"] >= threshold]
        correct = sum(row["unit_correct"] for row in selected)
        metrics["risk_coverage_curve"].append({
            "threshold": threshold,
            "coverage": ratio(len(selected), len(usable)),
            "risk": None if not selected else 1.0 - ratio(correct, len(selected)),
            "assigned_units": len(selected),
        })
    return metrics


def compare_runs(baseline, candidate, all_ground_truth):
    baseline_by_id = {row["item_id"]: row for row in baseline}
    candidate_by_id = {row["item_id"]: row for row in candidate}
    gt_by_id = {row["item_id"]: row for row in all_ground_truth}
    new = []
    removed = []
    for item_id in sorted(set(baseline_by_id).intersection(candidate_by_id)):
        left, right = baseline_by_id[item_id], candidate_by_id[item_id]
        gt = gt_by_id.get(item_id, {})
        changed_prediction = left["predicted_player_ids"] != right["predicted_player_ids"]
        if right["unit_assigned"] and (not left["unit_assigned"] or changed_prediction):
            new.append({**right, "annotation_status": gt.get("annotation_status")})
        if left["unit_assigned"] and (not right["unit_assigned"] or changed_prediction):
            removed.append({**left, "annotation_status": gt.get("annotation_status")})
    return {
        "new_decisions": len(new),
        "new_determinate": sum(row.get("annotation_status") == "determinate" for row in new),
        "new_indeterminate": sum(row.get("annotation_status") != "determinate" for row in new),
        "new_false_positives": sum(row.get("annotation_status") == "determinate" and not row["unit_correct"] for row in new),
        "removed_decisions": len(removed),
        "removed_correct": sum(row.get("annotation_status") == "determinate" and row["unit_correct"] for row in removed),
        "new_items": [row["item_id"] for row in new],
        "removed_items": [row["item_id"] for row in removed],
    }


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_clustered(results, iterations=1000, seed=20260702):
    usable = [row for row in results if row["determinate"] and not row["excluded"]]
    by_video = defaultdict(list)
    for row in usable:
        by_video[row["video_id"]].append(row)
    randomizer = random.Random(int(seed))
    values = []
    for _ in range(int(iterations)):
        sampled = []
        for rows in by_video.values():
            sampled.extend(randomizer.choice(rows) for _ in rows)
        assigned = [row for row in sampled if row["unit_assigned"]]
        values.append({
            "precision": ratio(sum(row["unit_correct"] for row in assigned), len(assigned)),
            "correct_coverage": ratio(sum(row["unit_correct"] for row in sampled), len(sampled)),
        })
    return {
        key: percentile_interval([row[key] for row in values if row[key] is not None])
        for key in ("precision", "correct_coverage")
    }


def paired_precision_delta_interval(baseline, candidate, iterations=1000, seed=20260702):
    baseline_by_id = {row["item_id"]: row for row in baseline if row["determinate"] and not row["excluded"]}
    candidate_by_id = {row["item_id"]: row for row in candidate if row["determinate"] and not row["excluded"]}
    common = sorted(set(baseline_by_id).intersection(candidate_by_id))
    by_video = defaultdict(list)
    for item_id in common:
        by_video[baseline_by_id[item_id]["video_id"]].append(item_id)
    randomizer = random.Random(int(seed))
    deltas = []
    for _ in range(int(iterations)):
        sampled = []
        for ids in by_video.values():
            sampled.extend(randomizer.choice(ids) for _ in ids)
        base_assigned = [baseline_by_id[item] for item in sampled if baseline_by_id[item]["unit_assigned"]]
        cand_assigned = [candidate_by_id[item] for item in sampled if candidate_by_id[item]["unit_assigned"]]
        base_precision = ratio(sum(row["unit_correct"] for row in base_assigned), len(base_assigned))
        cand_precision = ratio(sum(row["unit_correct"] for row in cand_assigned), len(cand_assigned))
        if base_precision is not None and cand_precision is not None:
            deltas.append(cand_precision - base_precision)
    return percentile_interval(deltas)


def percentile_interval(values):
    if not values:
        return [None, None]
    ordered = sorted(values)
    return [
        ordered[max(0, round(0.025 * (len(ordered) - 1)))],
        ordered[min(len(ordered) - 1, round(0.975 * (len(ordered) - 1)))],
    ]


def promotion_gate(
    baseline_metrics,
    candidate_metrics,
    delta,
    duplicate_delta=0,
    violation_delta=0,
    pair_false_positives=0,
    pair_indeterminate=0,
    hashes_match=True,
    precision_delta_ci95=None,
):
    checks = {
        "zero_false_positive_delta": delta["new_false_positives"] == 0,
        "delta_fully_determinate": delta["new_indeterminate"] == 0,
        "zero_pair_false_positives": int(pair_false_positives) == 0,
        "accepted_pairs_fully_determinate": int(pair_indeterminate) == 0,
        "duplicates_not_increased": int(duplicate_delta) <= 0,
        "violations_not_increased": int(violation_delta) <= 0,
        "correct_coverage_not_lower": (
            candidate_metrics.get("correct_coverage") is not None
            and baseline_metrics.get("correct_coverage") is not None
            and candidate_metrics["correct_coverage"] >= baseline_metrics["correct_coverage"]
        ),
        "no_significant_precision_regression": (
            (
                baseline_metrics.get("identity_precision_unit") is None
                and candidate_metrics.get("identity_precision_unit") is not None
            )
            or (
                precision_delta_ci95 is not None
                and precision_delta_ci95[1] is not None
                and precision_delta_ci95[1] >= 0.0
            )
        ),
        "hashes_match": bool(hashes_match),
    }
    indeterminate = not checks["delta_fully_determinate"] or not checks["accepted_pairs_fully_determinate"]
    return {"status": "pass" if all(checks.values()) else "inconclusive" if indeterminate else "fail", "checks": checks}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes"}
