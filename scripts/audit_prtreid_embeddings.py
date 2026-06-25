#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


EMPTY = {"", "None", "none", "null", "unknown", None}
SIMILARITY_THRESHOLDS = (0.90, 0.95, 0.97, 0.98, 0.99)

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(
        description="Measure whether exported PRTReID embeddings separate player identity, team and role labels."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/costume-video")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--min-identity-confidence", type=float, default=0.75)
    parser.add_argument("--include-propagated", action="store_true")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    metadata = Path(args.artifacts_root) / args.run / "metadata"
    tracklets_path = metadata / f"{args.video_id}_tracklets.csv"
    if not tracklets_path.exists():
        raise SystemExit(f"missing tracklets CSV: {tracklets_path}")

    rows = read_csv(tracklets_path)
    tracklets = aggregate_tracklets(
        rows,
        min_frames=args.min_frames,
        min_identity_confidence=args.min_identity_confidence,
        include_propagated=args.include_propagated,
    )
    propagation_path = metadata / f"{args.video_id}_identity_propagation.json"
    propagation = read_json(propagation_path)
    report = build_report(rows, tracklets, propagation=propagation, max_examples=args.max_examples)
    report.update(
        {
            "video_id": args.video_id,
            "run": args.run,
            "tracklets_path": str(tracklets_path),
            "propagation_path": str(propagation_path) if propagation_path.exists() else None,
            "filters": {
                "min_frames": args.min_frames,
                "min_identity_confidence": args.min_identity_confidence,
                "include_propagated": args.include_propagated,
            },
        }
    )
    print_report(report)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def aggregate_tracklets(rows, min_frames=8, min_identity_confidence=0.75, include_propagated=False):
    groups = defaultdict(list)
    for row in rows:
        if row.get("track_group", "players") != "players":
            continue
        embedding = parse_embedding(row.get("visual_embedding"))
        if not embedding:
            continue
        display_id = row.get("display_track_id")
        if display_id in EMPTY:
            continue
        groups[str(display_id)].append((row, normalize(embedding)))

    output = []
    for display_id, items in groups.items():
        if len(items) < min_frames:
            continue
        rows_for_tracklet = [item[0] for item in items]
        embeddings = [item[1] for item in items]
        reliable_rows = [
            row
            for row in rows_for_tracklet
            if row.get("player_id") not in EMPTY
            and float_or_zero(row.get("identity_confidence")) >= min_identity_confidence
            and (include_propagated or row.get("identity_status") != "propagated")
        ]
        player_id = mode(row.get("player_id") for row in reliable_rows)
        team_id = mode(normalized_int(row.get("team_id")) for row in rows_for_tracklet)
        role_detection = mode(row.get("role_detection") for row in rows_for_tracklet)
        reid_role_detection = mode(row.get("reid_role_detection") for row in rows_for_tracklet)
        output.append(
            {
                "display_track_id": display_id,
                "frames": len(items),
                "embedding": normalize(mean_vector(embeddings)),
                "player_id": player_id,
                "team_id": team_id,
                "role_detection": role_detection,
                "reid_role_detection": reid_role_detection,
                "mean_identity_confidence": mean(float_or_zero(row.get("identity_confidence")) for row in reliable_rows),
                "mean_crop_quality": mean(float_or_zero(row.get("crop_quality")) for row in rows_for_tracklet),
                "reid_model": mode(row.get("reid_model") for row in rows_for_tracklet),
            }
        )
    return output


def build_report(rows, tracklets, propagation=None, max_examples=10):
    pairwise = pairwise_metrics(tracklets, max_examples=max_examples)
    nearest = nearest_neighbor_metrics(tracklets, max_examples=max_examples)
    return {
        "row_summary": row_summary(rows),
        "tracklet_summary": tracklet_summary(tracklets),
        "pairwise": pairwise,
        "threshold_risk": threshold_risk(pairwise),
        "nearest_neighbor": nearest,
        "propagation_pairs": propagation_pair_metrics(tracklets, propagation or {}, max_examples=max_examples),
        "role": role_metrics(tracklets),
        "interpretation": interpret(pairwise, nearest),
    }


def row_summary(rows):
    embedded = [row for row in rows if parse_embedding(row.get("visual_embedding"))]
    return {
        "rows": len(rows),
        "embedded_rows": len(embedded),
        "prtreid_rows": sum(1 for row in embedded if row.get("reid_model") == "prtreid"),
        "reid_models": dict(Counter(row.get("reid_model") or "missing" for row in embedded).most_common()),
        "reid_roles": dict(Counter(row.get("reid_role_detection") or "missing" for row in embedded).most_common()),
    }


def tracklet_summary(tracklets):
    labeled = [item for item in tracklets if item.get("player_id") not in EMPTY]
    teams = [item for item in tracklets if item.get("team_id") is not None]
    dims = sorted({len(item["embedding"]) for item in tracklets if item.get("embedding")})
    return {
        "embedded_tracklets": len(tracklets),
        "identity_labeled_tracklets": len(labeled),
        "team_labeled_tracklets": len(teams),
        "embedding_dims": dims,
        "players_with_multiple_tracklets": sum(
            1 for _player_id, count in Counter(item.get("player_id") for item in labeled).items() if count >= 2
        ),
    }


def pairwise_metrics(tracklets, max_examples=10):
    buckets = {
        "same_player": [],
        "different_player_same_team": [],
        "different_team": [],
    }
    examples = {
        "lowest_same_player": [],
        "highest_different_player_same_team": [],
        "highest_different_team": [],
    }
    for left_index, left in enumerate(tracklets):
        for right in tracklets[left_index + 1 :]:
            similarity = cosine(left["embedding"], right["embedding"])
            left_player = left.get("player_id")
            right_player = right.get("player_id")
            left_team = left.get("team_id")
            right_team = right.get("team_id")
            labeled_players = left_player not in EMPTY and right_player not in EMPTY
            labeled_teams = left_team is not None and right_team is not None
            pair = pair_example(left, right, similarity)
            if labeled_players and left_player == right_player:
                buckets["same_player"].append(similarity)
                examples["lowest_same_player"].append(pair)
            elif labeled_players and labeled_teams and left_team == right_team:
                buckets["different_player_same_team"].append(similarity)
                examples["highest_different_player_same_team"].append(pair)
            elif labeled_teams and left_team != right_team:
                buckets["different_team"].append(similarity)
                examples["highest_different_team"].append(pair)

    summary = {name: distribution(values) for name, values in buckets.items()}
    summary["values"] = buckets
    summary["examples"] = {
        "lowest_same_player": sorted(examples["lowest_same_player"], key=lambda row: row["similarity"])[:max_examples],
        "highest_different_player_same_team": sorted(
            examples["highest_different_player_same_team"], key=lambda row: -row["similarity"]
        )[:max_examples],
        "highest_different_team": sorted(examples["highest_different_team"], key=lambda row: -row["similarity"])[
            :max_examples
        ],
    }
    return summary


def threshold_risk(pairwise):
    values = pairwise.get("values", {})
    same_team_impostors = values.get("different_player_same_team", [])
    different_team = values.get("different_team", [])
    return [
        {
            "threshold": threshold,
            "same_team_impostor_pairs_at_or_above": count_at_or_above(same_team_impostors, threshold),
            "same_team_impostor_fraction_at_or_above": rounded_ratio_at_or_above(same_team_impostors, threshold),
            "different_team_pairs_at_or_above": count_at_or_above(different_team, threshold),
            "different_team_fraction_at_or_above": rounded_ratio_at_or_above(different_team, threshold),
        }
        for threshold in SIMILARITY_THRESHOLDS
    ]


def nearest_neighbor_metrics(tracklets, max_examples=10):
    player_queries = []
    team_queries = []
    failures = []
    player_counts = Counter(item.get("player_id") for item in tracklets if item.get("player_id") not in EMPTY)
    for query in tracklets:
        neighbors = [
            (cosine(query["embedding"], candidate["embedding"]), candidate)
            for candidate in tracklets
            if candidate["display_track_id"] != query["display_track_id"]
        ]
        if not neighbors:
            continue
        neighbors.sort(key=lambda item: -item[0])
        nearest_similarity, nearest = neighbors[0]
        if query.get("team_id") is not None and nearest.get("team_id") is not None:
            team_queries.append(query.get("team_id") == nearest.get("team_id"))
        if query.get("player_id") not in EMPTY and player_counts[query.get("player_id")] >= 2:
            ok = query.get("player_id") == nearest.get("player_id")
            player_queries.append(ok)
            if not ok:
                failures.append(
                    {
                        "query_display_track_id": query["display_track_id"],
                        "query_player_id": query.get("player_id"),
                        "nearest_display_track_id": nearest["display_track_id"],
                        "nearest_player_id": nearest.get("player_id"),
                        "nearest_team_id": nearest.get("team_id"),
                        "similarity": round(nearest_similarity, 4),
                    }
                )
    return {
        "player_top1_accuracy": ratio(player_queries),
        "player_queries": len(player_queries),
        "team_top1_accuracy": ratio(team_queries),
        "team_queries": len(team_queries),
        "player_failures": failures[:max_examples],
    }


def role_metrics(tracklets):
    comparable = [
        item
        for item in tracklets
        if item.get("role_detection") not in EMPTY and item.get("reid_role_detection") not in EMPTY
    ]
    matches = [item.get("role_detection") == item.get("reid_role_detection") for item in comparable]
    confusion = Counter((item.get("role_detection"), item.get("reid_role_detection")) for item in comparable)
    return {
        "comparable_tracklets": len(comparable),
        "accuracy": ratio(matches),
        "confusion": [
            {"role_detection": key[0], "reid_role_detection": key[1], "count": count}
            for key, count in confusion.most_common()
        ],
    }


def propagation_pair_metrics(tracklets, propagation, max_examples=10):
    by_display_id = {str(item["display_track_id"]): item for item in tracklets}
    rows = []
    for item in propagation.get("propagations", []) if isinstance(propagation, dict) else []:
        source_id = str(item.get("source_display_id"))
        target_id = str(item.get("target_display_id"))
        source = by_display_id.get(source_id)
        target = by_display_id.get(target_id)
        if not source or not target:
            continue
        similarity = cosine(source["embedding"], target["embedding"])
        rows.append(
            {
                "source_display_id": source_id,
                "target_display_id": target_id,
                "player_id": item.get("player_id"),
                "team_id": target.get("team_id"),
                "computed_similarity": round(similarity, 4),
                "reported_visual_similarity": float_or_none(item.get("visual_similarity")),
                "assignment_confidence": float_or_none(item.get("assignment_confidence")),
                "risk_note": propagation_risk_note(similarity),
                "target_frames": item.get("target_frames"),
                "temporal_gap": item.get("temporal_gap"),
            }
        )
    return {
        "count": len(rows),
        "similarity": distribution([row["computed_similarity"] for row in rows]),
        "pairs": sorted(rows, key=lambda row: row["computed_similarity"])[:max_examples],
    }


def propagation_risk_note(similarity):
    if similarity < 0.98:
        return "below 0.98; audit shows same-team impostors still common in this range"
    if similarity < 0.99:
        return "0.98-0.99; still requires jersey/temporal validation"
    return ">=0.99; no same-team impostors observed at this threshold in current audit"


def interpret(pairwise, nearest):
    notes = []
    same = pairwise.get("same_player", {})
    same_team_diff = pairwise.get("different_player_same_team", {})
    diff_team = pairwise.get("different_team", {})
    if same.get("count", 0) == 0:
        notes.append("No repeated reliable player identities; same-player separation cannot be measured.")
    elif same.get("p50") is not None and same_team_diff.get("p90") is not None:
        gap = same["p50"] - same_team_diff["p90"]
        notes.append(f"same-player p50 minus same-team impostor p90 = {gap:.4f}")
    if diff_team.get("p90") is not None and same_team_diff.get("p90") is not None:
        notes.append(
            "different-team p90 is "
            f"{diff_team['p90']:.4f}; same-team impostor p90 is {same_team_diff['p90']:.4f}"
        )
    if nearest.get("player_queries", 0) > 0:
        notes.append(f"player nearest-neighbor top1 = {nearest['player_top1_accuracy']:.3f}")
    if nearest.get("team_queries", 0) > 0:
        notes.append(f"team nearest-neighbor top1 = {nearest['team_top1_accuracy']:.3f}")
    return notes


def print_report(report):
    print(f"=== PRTReID embedding audit: {report['video_id']} / {report['run']} ===")
    print("filters", report["filters"])
    print("\n-- rows --")
    print(report["row_summary"])
    print("\n-- tracklets --")
    print(report["tracklet_summary"])
    print("\n-- pairwise cosine similarity --")
    for name in ("same_player", "different_player_same_team", "different_team"):
        print(name, {key: value for key, value in report["pairwise"][name].items() if key != "values"})
    print("\n-- threshold risk --")
    for row in report["threshold_risk"]:
        print(row)
    print("\n-- nearest neighbor --")
    print(
        {
            key: report["nearest_neighbor"][key]
            for key in ("player_top1_accuracy", "player_queries", "team_top1_accuracy", "team_queries")
        }
    )
    if report["nearest_neighbor"]["player_failures"]:
        print("player_failures")
        for row in report["nearest_neighbor"]["player_failures"]:
            print(row)
    print("\n-- propagation pairs --")
    print({key: value for key, value in report["propagation_pairs"].items() if key != "pairs"})
    for row in report["propagation_pairs"]["pairs"]:
        print(row)
    print("\n-- role --")
    print(report["role"])
    print("\n-- examples --")
    for name, rows in report["pairwise"]["examples"].items():
        print(name)
        for row in rows:
            print(row)
    print("\n-- interpretation --")
    for note in report["interpretation"]:
        print("-", note)


def pair_example(left, right, similarity):
    return {
        "left_display_track_id": left["display_track_id"],
        "left_player_id": left.get("player_id"),
        "left_team_id": left.get("team_id"),
        "right_display_track_id": right["display_track_id"],
        "right_player_id": right.get("player_id"),
        "right_team_id": right.get("team_id"),
        "similarity": round(similarity, 4),
    }


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "p10": round(percentile(ordered, 0.10), 4),
        "p50": round(percentile(ordered, 0.50), 4),
        "p90": round(percentile(ordered, 0.90), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_embedding(value):
    if not value or value in EMPTY:
        return None
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = json.loads(value)
        except Exception:
            return None
    if not isinstance(parsed, list):
        return None
    try:
        vector = [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None
    return vector if vector else None


def mean_vector(vectors):
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    count = 0
    for vector in vectors:
        if len(vector) != dim:
            continue
        count += 1
        for index, value in enumerate(vector):
            sums[index] += value
    if count == 0:
        return []
    return [value / count for value in sums]


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return list(vector)
    return [value / norm for value in vector]


def cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def percentile(ordered, q):
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def ratio(values):
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def count_at_or_above(values, threshold):
    return sum(1 for value in values if value >= threshold)


def rounded_ratio_at_or_above(values, threshold):
    if not values:
        return None
    return round(count_at_or_above(values, threshold) / len(values), 4)


def mode(values):
    counter = Counter(value for value in values if value not in EMPTY)
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def normalized_int(value):
    if value in EMPTY:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_zero(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
