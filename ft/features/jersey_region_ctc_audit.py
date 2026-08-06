"""Audit-only YOLO number-region detector plus numeric CTC recognizer."""

import gc
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
)
from ft.identity.roster import roster_numbers_by_team


class JerseyRegionCTCAuditor:
    def __init__(
        self,
        ctc_checkpoint,
        ctc_checkpoint_sha256,
        detector_checkpoint,
        detector_checkpoint_sha256,
        detector_confidence=0.25,
        box_padding=0.10,
        batch_size=64,
        detector_batch_size=8,
        device="cuda",
        detector_device="0",
        min_override_confidence=0.90,
        fusion_preview_enabled=True,
        max_crops_per_tracklet=5,
        min_frame_gap=5,
        roster=None,
        roster_reranking_enabled=False,
        roster_reranking_top_k=5,
        super_resolution_enabled=False,
        super_resolution_scale=4,
        super_resolution_max_side=100,
    ):
        self.ctc_checkpoint = verified_file(ctc_checkpoint, ctc_checkpoint_sha256)
        self.detector_checkpoint = verified_file(
            detector_checkpoint, detector_checkpoint_sha256
        )
        self.ctc_checkpoint_sha256 = str(ctc_checkpoint_sha256).lower()
        self.detector_checkpoint_sha256 = str(detector_checkpoint_sha256).lower()
        self.detector_confidence = float(detector_confidence)
        self.box_padding = float(box_padding)
        self.batch_size = int(batch_size)
        self.detector_batch_size = int(detector_batch_size)
        self.device = str(device)
        self.detector_device = str(detector_device)
        self.min_override_confidence = float(min_override_confidence)
        self.fusion_preview_enabled = bool(fusion_preview_enabled)
        self.max_crops_per_tracklet = int(max_crops_per_tracklet)
        self.min_frame_gap = int(min_frame_gap)
        self.roster_reranking_enabled = bool(roster_reranking_enabled)
        self.roster_reranking_top_k = int(roster_reranking_top_k)
        self.roster_by_team = roster_numbers_by_team(roster or []) if roster_reranking_enabled else {}
        self.super_resolution_enabled = bool(super_resolution_enabled)
        self.super_resolution_scale = max(1, int(super_resolution_scale))
        self.super_resolution_max_side = max(1, int(super_resolution_max_side))

    def run(self, jersey_diagnostics, primary_assignments, frame_selection_rows=None, player_rows=None):
        import torch
        from PIL import Image
        from torchvision import transforms
        from ultralytics import YOLO

        scene_segment_by_frame = frame_scene_segment_lookup(player_rows)
        selected = collect_selected_crops(
            jersey_diagnostics,
            frame_selection_rows=frame_selection_rows,
            max_crops_per_tracklet=self.max_crops_per_tracklet,
            min_frame_gap=self.min_frame_gap,
            scene_segment_by_frame=scene_segment_by_frame,
        )
        if not selected:
            return self._empty("no_selected_crops")

        detector = YOLO(str(self.detector_checkpoint))
        regions = []
        for start in range(0, len(selected), self.detector_batch_size):
            batch = selected[start:start + self.detector_batch_size]
            results = detector.predict(
                [row["crop_path"] for row in batch],
                conf=self.detector_confidence,
                device=self.detector_device,
                batch=self.detector_batch_size,
                verbose=False,
                stream=False,
            )
            for row, result in zip(batch, results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                index = int(result.boxes.conf.argmax().item())
                image = crop_region(
                    row["crop_path"],
                    tuple(float(value) for value in result.boxes.xyxyn[index].tolist()),
                    self.box_padding,
                )
                if image is None:
                    continue
                image = upscale_small_region(
                    image,
                    enabled=self.super_resolution_enabled,
                    scale=self.super_resolution_scale,
                    max_side=self.super_resolution_max_side,
                )
                regions.append({
                    **row,
                    "image": image,
                    "region_width": int(image.size[0]),
                    "region_height": int(image.size[1]),
                    "detector_confidence": float(result.boxes.conf[index]),
                    "region_xyxyn": [
                        float(value) for value in result.boxes.xyxyn[index].tolist()
                    ],
                })
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        checkpoint = torch.load(self.ctc_checkpoint, map_location="cpu")
        metadata = checkpoint["metadata"]
        recognizer = build_numeric_crnn(pretrained=False).to(self.device)
        recognizer.load_state_dict(checkpoint["state_dict"])
        recognizer.eval()
        transform = transforms.Compose([
            transforms.Resize(tuple(metadata["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                metadata["normalization"]["mean"],
                metadata["normalization"]["std"],
            ),
        ])
        tracks = defaultdict(lambda: {"scores": [], "weights": [], "frames": []})
        crop_rows = []
        for start in range(0, len(regions), self.batch_size):
            batch = regions[start:start + self.batch_size]
            images = torch.stack([transform(row["image"]) for row in batch]).to(self.device)
            with torch.no_grad():
                logits = recognizer(images).cpu()
            for index, row in enumerate(batch):
                scores = candidate_log_probabilities(logits[:, index, :])
                key = tracklet_key(row["display_track_id"], row.get("scene_segment_id"))
                tracks[key]["scores"].append(scores)
                tracks[key]["weights"].append(row["detector_confidence"])
                tracks[key]["frames"].append(row["frame"])
                ranking = sorted(scores.items(), key=lambda item: -item[1])
                crop_rows.append({
                    "display_track_id": row["display_track_id"],
                    "frame": row["frame"],
                    "crop_path": row["crop_path"],
                    "crop_sha256": sha256_file(row["crop_path"]),
                    "crop_bytes": Path(row["crop_path"]).stat().st_size,
                    "crop_quality": row["crop_quality"],
                    "selection_score": row["selection_score"],
                    "selection_reason": row.get("selection_reason"),
                    "selection_rank": row.get("selection_rank"),
                    "detector_confidence": row["detector_confidence"],
                    "detector_checkpoint_sha256": self.detector_checkpoint_sha256,
                    "region_xyxyn": row["region_xyxyn"],
                    "region_width": row["region_width"],
                    "region_height": row["region_height"],
                    "box_padding": self.box_padding,
                    "ctc_top1": int(ranking[0][0]),
                    "ctc_top1_log_probability": ranking[0][1],
                    "ctc_top5": [int(value) for value, _ in ranking[:5]],
                    "ctc_checkpoint_sha256": self.ctc_checkpoint_sha256,
                })
        del recognizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        team_by_display = (
            team_by_display_track_id(player_rows) if self.roster_reranking_enabled else {}
        )

        primary = primary_by_display(primary_assignments)
        standalone = {}
        fusion_preview = {}
        roster_preview = {}
        tracklets = {}
        for key in sorted(tracks, key=numeric_sort_key):
            evidence = tracks[key]
            result = aggregate_frames(evidence["scores"], evidence["weights"])
            display_track_id = display_track_id_from_key(key)
            proposal = {
                "display_track_id": display_track_id,
                "jersey_number": result["prediction"],
                "confidence": result["confidence"],
                "winner_margin": result["margin"],
                "recognized_frames": len(evidence["scores"]),
                "frames": evidence["frames"],
                "top5": [int(value) for value in list(result["scores"])[:5]],
                "applied": False,
            }
            standalone[key] = proposal
            # primary_assignments is not scene-segment-aware upstream, so this
            # comparison intentionally looks up by the plain display_track_id
            # regardless of which scene segment produced this CTC evidence --
            # unrelated to the roster-reranking scene-disambiguation below.
            before = (primary.get(str(display_track_id)) or {}).get("jersey_number")
            if self.fusion_preview_enabled:
                chosen, reason = conservative_preview(
                    before,
                    proposal["jersey_number"],
                    proposal["confidence"],
                    self.min_override_confidence,
                )
            else:
                chosen, reason = before, "audit_no_fusion_preview"
            fusion_preview[key] = {
                "display_track_id": display_track_id,
                "primary_number": before,
                "candidate_number": proposal["jersey_number"],
                "preview_number": chosen,
                "reason": reason,
                "applied": False,
            }
            roster_preview[key] = roster_rerank_preview(
                key,
                proposal["jersey_number"],
                result["scores"],
                team_by_display,
                self.roster_by_team,
                self.roster_reranking_top_k,
                enabled=self.roster_reranking_enabled,
            )
            tracklets[key] = {
                **proposal,
                "fusion_preview": fusion_preview[key],
                "roster_rerank_preview": roster_preview[key],
            }

        return {
            "enabled": True,
            "status": "ok",
            "mode": "audit",
            "audit_only": True,
            "identity_mutations": 0,
            "selected_crops": len(selected),
            "detected_regions": len(regions),
            "region_detection_coverage": ratio(len(regions), len(selected)),
            "tracklets_with_prediction": len(standalone),
            "standalone_assignments": standalone,
            "fusion_preview": fusion_preview,
            "roster_rerank_preview": roster_preview,
            "roster_reranking_summary": roster_rerank_summary(roster_preview),
            "comparison_to_primary": compare_numbers(primary, standalone_by_plain_display_id(standalone)),
            "tracklets": tracklets,
            "crops": crop_rows,
            "configuration": self.configuration(),
        }

    def configuration(self):
        return {
            "ctc_checkpoint": str(self.ctc_checkpoint),
            "ctc_checkpoint_sha256": self.ctc_checkpoint_sha256,
            "detector_checkpoint": str(self.detector_checkpoint),
            "detector_checkpoint_sha256": self.detector_checkpoint_sha256,
            "detector_confidence": self.detector_confidence,
            "box_padding": self.box_padding,
            "batch_size": self.batch_size,
            "detector_batch_size": self.detector_batch_size,
            "device": self.device,
            "detector_device": self.detector_device,
            "min_override_confidence": self.min_override_confidence,
            "fusion_preview_enabled": self.fusion_preview_enabled,
            "max_crops_per_tracklet": self.max_crops_per_tracklet,
            "min_frame_gap": self.min_frame_gap,
            "roster_reranking_enabled": self.roster_reranking_enabled,
            "roster_reranking_top_k": self.roster_reranking_top_k,
            "roster_teams_loaded": sorted(self.roster_by_team),
            "super_resolution_enabled": self.super_resolution_enabled,
            "super_resolution_scale": self.super_resolution_scale,
            "super_resolution_max_side": self.super_resolution_max_side,
        }

    def _empty(self, status):
        return {
            "enabled": True,
            "status": status,
            "mode": "audit",
            "audit_only": True,
            "identity_mutations": 0,
            "selected_crops": 0,
            "detected_regions": 0,
            "tracklets_with_prediction": 0,
            "standalone_assignments": {},
            "fusion_preview": {},
            "roster_rerank_preview": {},
            "roster_reranking_summary": roster_rerank_summary({}),
            "comparison_to_primary": compare_numbers({}, {}),
            "tracklets": {},
            "crops": [],
            "configuration": self.configuration(),
        }


def collect_selected_crops(
    diagnostics,
    frame_selection_rows=None,
    max_crops_per_tracklet=5,
    min_frame_gap=5,
    scene_segment_by_frame=None,
):
    """Groups selected crops by tracklet for top-k/temporal-gap filtering.

    A raw display_track_id can be reused for different physical players
    across scene resets (e.g. resetbytetrack). Grouping by display_track_id
    alone would merge unrelated players' crops into one CTC-recognized
    "tracklet". scene_segment_by_frame (built from live player_rows, see
    frame_scene_segment_lookup) lets us group by the finer
    (display_track_id, scene_segment_id) key instead whenever that
    information is available; falls back to the plain display_track_id key
    when it is not (e.g. in tests that call this without player_rows).
    """
    scene_segment_by_frame = scene_segment_by_frame or {}
    selection_by_path = {
        str(row.get("crop_path")): {
            "score": float(
                row.get("selection_score", row.get("legibility_score", 0.0)) or 0.0
            ),
            "reason": row.get("selection_reason"),
            "rank": row.get("selection_rank"),
        }
        for row in (frame_selection_rows or [])
        if row.get("crop_path")
    }
    grouped, seen = defaultdict(list), set()
    for diagnostic in (diagnostics.get("tracklets") or {}).values():
        display = diagnostic.get("display_track_id")
        if display is None:
            continue
        for crop in diagnostic.get("selected_crops", []):
            path = str(crop.get("crop_path") or "")
            identity = str(display), path
            if not path or identity in seen or not Path(path).is_file():
                continue
            seen.add(identity)
            frame = int(crop.get("frame") or 0)
            scene_segment_id = scene_segment_by_frame.get((str(display), frame))
            group_key = tracklet_key(display, scene_segment_id)
            grouped[group_key].append({
                "display_track_id": int(display),
                "scene_segment_id": scene_segment_id,
                "frame": frame,
                "crop_path": path,
                "crop_quality": float(crop.get("crop_quality") or 0.0),
                "selection_score": (
                    selection_by_path.get(path) or {}
                ).get("score"),
                "selection_reason": (
                    selection_by_path.get(path) or {}
                ).get("reason"),
                "selection_rank": (
                    selection_by_path.get(path) or {}
                ).get("rank"),
            })
    output = []
    for rows in grouped.values():
        ranked = sorted(
            rows,
            key=lambda row: (
                -(row["selection_score"] if row["selection_score"] is not None else row["crop_quality"]),
                row["frame"],
                row["crop_path"],
            ),
        )
        selected = []
        for row in ranked:
            if any(
                abs(row["frame"] - other["frame"]) < int(min_frame_gap)
                for other in selected
            ):
                continue
            selected.append(row)
            if len(selected) >= int(max_crops_per_tracklet):
                break
        output.extend(selected)
    return output


def team_by_display_track_id(player_rows):
    """Majority-vote team_id per tracklet from already-team-tagged rows.

    Keyed by tracklet_key(display_track_id, scene_segment_id) when rows carry
    scene_segment_id (a raw display_track_id can be reused for a different
    physical player after a scene reset), falling back to the plain
    display_track_id when they don't.
    """
    votes = defaultdict(Counter)
    for row in player_rows or []:
        display_id = row.get("display_track_id")
        team_id = row.get("team_id")
        if display_id is None or team_id is None:
            continue
        key = tracklet_key(display_id, row.get("scene_segment_id"))
        votes[key][int(team_id)] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items() if counter}


def tracklet_key(display_track_id, scene_segment_id):
    """Groups CTC/roster evidence by the finer (display_track_id,
    scene_segment_id) pair when scene_segment_id is known, since a raw
    display_track_id can be reused for a different physical player after a
    scene reset. Falls back to the plain display_track_id when
    scene_segment_id is unavailable (e.g. in unit tests without player_rows),
    preserving prior behavior."""
    if scene_segment_id in (None, ""):
        return str(display_track_id)
    return f"{display_track_id}#{scene_segment_id}"


def display_track_id_from_key(key):
    return int(str(key).split("#", 1)[0])


def frame_scene_segment_lookup(player_rows):
    lookup = {}
    for row in player_rows or []:
        display_id = row.get("display_track_id")
        frame = row.get("frame")
        if display_id is None or frame is None:
            continue
        lookup[(str(display_id), int(frame))] = row.get("scene_segment_id")
    return lookup


def standalone_by_plain_display_id(standalone):
    """Collapses scene-segment-keyed standalone proposals back to one entry
    per plain display_track_id for the comparison_to_primary diagnostic,
    which pre-dates scene-segment disambiguation and is not scene-aware.
    Picks the proposal with the most recognized frames as the representative
    one when a raw ID was reused across scene segments."""
    best = {}
    for proposal in standalone.values():
        display_id = str(proposal["display_track_id"])
        current = best.get(display_id)
        if current is None or proposal["recognized_frames"] > current["recognized_frames"]:
            best[display_id] = proposal
    return best


def roster_rerank_preview(key, original_number, ranked_scores, team_by_display, roster_by_team, top_k, enabled):
    """Audit-only: what the roster-constrained top-1 WOULD be, restricted to
    the top_k CTC candidates. Never written back into jersey_number/tracklets
    used downstream -- identity_mutations stays 0 regardless of this preview.
    """
    display_track_id = display_track_id_from_key(key)
    if not enabled or original_number is None:
        return {
            "display_track_id": display_track_id,
            "team_id": None,
            "roster_size": 0,
            "original_number": original_number,
            "preview_number": original_number,
            "changed": False,
            "reason": "roster_reranking_disabled" if not enabled else "candidate_abstained",
        }
    team_id = team_by_display.get(str(key))
    roster = roster_by_team.get(team_id) if team_id is not None else None
    if not roster:
        return {
            "display_track_id": display_track_id,
            "team_id": team_id,
            "roster_size": 0,
            "original_number": original_number,
            "preview_number": original_number,
            "changed": False,
            "reason": "no_roster_for_team",
        }
    ranked = list(ranked_scores.items())[:top_k]
    in_roster = [int(candidate) for candidate, _score in ranked if int(candidate) in roster]
    preview_number = in_roster[0] if in_roster else original_number
    return {
        "display_track_id": display_track_id,
        "team_id": team_id,
        "roster_size": len(roster),
        "original_number": original_number,
        "preview_number": preview_number,
        "changed": preview_number != original_number,
        "reason": "roster_constrained_topk" if in_roster else "no_topk_candidate_in_roster",
    }


def roster_rerank_summary(roster_preview):
    changed = [row for row in roster_preview.values() if row.get("changed")]
    return {
        "tracklets": len(roster_preview),
        "changed": len(changed),
        "changed_display_ids": sorted(row["display_track_id"] for row in changed),
    }


def primary_by_display(assignments):
    output = {}
    for key, value in (assignments or {}).items():
        display = value.get("display_track_id", key)
        output[str(display)] = value
    return output


def conservative_preview(primary, candidate, confidence, threshold):
    if primary is None:
        return None, "primary_abstention_preserved"
    if candidate is None:
        return primary, "candidate_abstained"
    if int(primary) == int(candidate):
        return int(primary), "recognizers_agree"
    if float(confidence) >= float(threshold):
        return int(candidate), "high_confidence_override"
    return int(primary), "candidate_below_threshold"


def compare_numbers(primary, candidate):
    counts = {
        "tracklets": 0,
        "agreement": 0,
        "disagreement": 0,
        "primary_only": 0,
        "candidate_only": 0,
        "both_abstain": 0,
    }
    for key in sorted(set(primary) | set(candidate), key=numeric_sort_key):
        before = (primary.get(key) or {}).get("jersey_number")
        after = (candidate.get(key) or {}).get("jersey_number")
        counts["tracklets"] += 1
        if before is None and after is None:
            counts["both_abstain"] += 1
        elif before is None:
            counts["candidate_only"] += 1
        elif after is None:
            counts["primary_only"] += 1
        elif int(before) == int(after):
            counts["agreement"] += 1
        else:
            counts["disagreement"] += 1
    return counts


def crop_region(path, xyxyn, padding):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = xyxyn
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    box = (
        max(0, int((x1 - pad_x) * width)),
        max(0, int((y1 - pad_y) * height)),
        min(width, int((x2 + pad_x) * width)),
        min(height, int((y2 + pad_y) * height)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image.crop(box)


def upscale_small_region(image, enabled=False, scale=4, max_side=100):
    """Bicubic-upscale a region crop before CTC recognition, mirroring
    jersey_ocr.super_resolve_small_crop for the primary OCR's own crops."""
    if not enabled or image is None:
        return image
    from PIL import Image

    width, height = image.size
    if max(width, height) >= int(max_side):
        return image
    scale = max(1, int(scale))
    if scale <= 1:
        return image
    return image.resize((width * scale, height * scale), Image.BICUBIC)


def verified_file(path, expected_sha256):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest.lower() != str(expected_sha256 or "").lower():
        raise ValueError(f"checkpoint SHA-256 mismatch for {path}: {digest}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_sort_key(value):
    try:
        return 0, int(value), ""
    except (TypeError, ValueError):
        pass
    try:
        display_part, _, rest = str(value).partition("#")
        return 0, int(display_part), rest
    except (TypeError, ValueError):
        return 1, 0, str(value)


def ratio(a, b):
    return a / b if b else 0.0
