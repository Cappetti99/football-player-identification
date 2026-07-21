"""Conservative one-digit/two-digit jersey vote consolidation."""

from collections import defaultdict


class JerseyPrefixConsolidator:
    MODES = {"audit", "propose"}

    def __init__(
        self,
        mode="audit",
        min_long_votes=2,
        short_vote_weights=None,
        selected_short_vote_weight=None,
    ):
        self.mode = str(mode).lower()
        self.min_long_votes = int(min_long_votes)
        self.short_vote_weights = [float(value) for value in (short_vote_weights or [1.0, .75, .5, .25])]
        self.selected_short_vote_weight = (
            None if selected_short_vote_weight is None else float(selected_short_vote_weight)
        )
        self.counterfactual_rows = []
        if self.mode not in self.MODES:
            raise ValueError(f"unsupported jersey prefix consolidation mode: {self.mode!r}")
        if self.min_long_votes <= 0:
            raise ValueError("jersey_prefix_consolidation.min_long_votes must be positive")
        if not self.short_vote_weights or any(value < 0 or value > 1 for value in self.short_vote_weights):
            raise ValueError("short_vote_weights must contain values between 0 and 1")
        if self.mode == "propose":
            if self.selected_short_vote_weight is None:
                raise ValueError("selected_short_vote_weight is required in propose mode")
            if self.selected_short_vote_weight not in self.short_vote_weights:
                raise ValueError("selected_short_vote_weight must be present in short_vote_weights")

    def consolidate(
        self,
        display_track_id,
        detections,
        vote_fn,
        min_raw_confidence=0.0,
        digit_confusion_overrides=None,
        min_margin=0.0,
        min_votes=1,
        segment_index=None,
    ):
        baseline = vote_fn(
            detections,
            min_raw_confidence=min_raw_confidence,
            digit_confusion_overrides=digit_confusion_overrides,
        )
        supported_longs = supported_long_numbers(detections, self.min_long_votes, min_raw_confidence)
        results = {}
        for weight in self.short_vote_weights:
            weighted, affected = weight_short_detections(detections, supported_longs, weight)
            voted = vote_fn(
                weighted,
                min_raw_confidence=min_raw_confidence,
                digit_confusion_overrides=digit_confusion_overrides,
            )
            long_scores = own_long_scores(weighted, supported_longs, min_raw_confidence)
            rejection = prefix_rejection_reason(
                voted, long_scores, affected, min_margin, min_votes
            )
            decision = None if rejection else (voted.get("jersey_number") if voted else None)
            row = {
                "display_track_id": int(display_track_id),
                "segment_index": segment_index,
                "mode": self.mode,
                "short_vote_weight": float(weight),
                "supported_long_numbers": sorted(supported_longs),
                "supported_long_scores": long_scores,
                "affected_short_numbers": sorted(affected),
                "baseline_winner": baseline.get("jersey_number") if baseline else None,
                "baseline_votes": baseline.get("votes", 0) if baseline else 0,
                "baseline_margin": baseline.get("winner_margin") if baseline else None,
                "baseline_candidates": baseline.get("candidates", []) if baseline else [],
                "counterfactual_winner": voted.get("jersey_number") if voted else None,
                "counterfactual_votes": voted.get("votes", 0) if voted else 0,
                "counterfactual_margin": voted.get("winner_margin") if voted else None,
                "accepted": decision is not None,
                "decision": decision,
                "rejection_reason": rejection,
                "candidates": voted.get("candidates", []) if voted else [],
            }
            self.counterfactual_rows.append(row)
            results[float(weight)] = (voted, rejection)
        if self.mode == "audit":
            return baseline
        proposed, rejection = results[self.selected_short_vote_weight]
        return None if rejection else proposed

    def diagnostics(self):
        return {
            "enabled": True,
            "mode": self.mode,
            "min_long_votes": self.min_long_votes,
            "short_vote_weights": self.short_vote_weights,
            "selected_short_vote_weight": self.selected_short_vote_weight,
            "counterfactuals": len(self.counterfactual_rows),
            "changed_winners": sum(
                row["counterfactual_winner"] != row["baseline_winner"]
                for row in self.counterfactual_rows
            ),
            "accepted_counterfactuals": sum(bool(row["accepted"]) for row in self.counterfactual_rows),
        }


def supported_long_numbers(detections, min_votes, min_raw_confidence=0.0):
    crop_keys = defaultdict(set)
    for item in detections:
        number = item.get("number")
        if number is None or not (10 <= int(number) <= 99):
            continue
        if float(item.get("confidence", 0.0) or 0.0) < float(min_raw_confidence):
            continue
        crop_keys[int(number)].add(independent_crop_key(item))
    return {
        number for number, keys in crop_keys.items() if len(keys) >= int(min_votes)
    }


def weight_short_detections(detections, supported_longs, weight):
    supported_digits = {
        int(digit) for number in supported_longs for digit in str(int(number))
    }
    affected = set()
    output = []
    for item in detections:
        row = dict(item)
        number = row.get("number")
        if number is not None and 0 <= int(number) <= 9 and int(number) in supported_digits:
            row["vote_weight"] = float(row.get("vote_weight", 1.0) or 0.0) * float(weight)
            row["prefix_short_vote_weight"] = float(weight)
            affected.add(int(number))
        output.append(row)
    return output, affected


def own_long_scores(detections, supported_longs, min_raw_confidence=0.0):
    scores = defaultdict(float)
    for item in detections:
        number = item.get("number")
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if number is None or int(number) not in supported_longs or confidence < float(min_raw_confidence):
            continue
        scores[int(number)] += max(0.01, confidence) * max(
            0.0, float(item.get("vote_weight", 1.0) or 0.0)
        )
    return dict(scores)


def prefix_rejection_reason(
    voted, supported_long_scores, affected_shorts, min_margin, min_votes=1,
):
    if not voted:
        return "no_evidence"
    if int(voted.get("votes", 0) or 0) < int(min_votes):
        return "insufficient_votes"
    winner = int(voted["jersey_number"])
    compatible_long_numbers = {
        int(number) for number in supported_long_scores
        if any(str(short) in str(int(number)) for short in affected_shorts)
    }
    if affected_shorts and winner not in set(affected_shorts) | compatible_long_numbers:
        return "winner_not_prefix_compatible"
    compatible_scores = [
        float(supported_long_scores[number]) for number in compatible_long_numbers
    ]
    if len(compatible_scores) >= 2:
        best_long_score = max(compatible_scores)
        if sum(abs(score - best_long_score) <= 1e-9 for score in compatible_scores) > 1:
            return "supported_long_tie"
    if float(voted.get("winner_margin", 0.0) or 0.0) < float(min_margin):
        return "insufficient_margin"
    return None


def independent_crop_key(item):
    return (
        str(item.get("crop_path") or ""),
        int(item.get("frame", 0) or 0),
    )
