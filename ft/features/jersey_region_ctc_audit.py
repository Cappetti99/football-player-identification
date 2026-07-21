"""Audit-only YOLO number-region detector plus numeric CTC recognizer."""

import gc
import hashlib
from collections import defaultdict
from pathlib import Path

from ft.features.jersey_number_ctc import (
    aggregate_frames,
    build_numeric_crnn,
    candidate_log_probabilities,
)


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

    def run(self, jersey_diagnostics, primary_assignments, frame_selection_rows=None):
        import torch
        from PIL import Image
        from torchvision import transforms
        from ultralytics import YOLO

        selected = collect_selected_crops(
            jersey_diagnostics,
            frame_selection_rows=frame_selection_rows,
            max_crops_per_tracklet=self.max_crops_per_tracklet,
            min_frame_gap=self.min_frame_gap,
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
                key = str(row["display_track_id"])
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

        primary = primary_by_display(primary_assignments)
        standalone = {}
        fusion_preview = {}
        tracklets = {}
        for key in sorted(tracks, key=numeric_sort_key):
            evidence = tracks[key]
            result = aggregate_frames(evidence["scores"], evidence["weights"])
            proposal = {
                "display_track_id": int(key),
                "jersey_number": result["prediction"],
                "confidence": result["confidence"],
                "winner_margin": result["margin"],
                "recognized_frames": len(evidence["scores"]),
                "frames": evidence["frames"],
                "top5": [int(value) for value in list(result["scores"])[:5]],
                "applied": False,
            }
            standalone[key] = proposal
            before = (primary.get(key) or {}).get("jersey_number")
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
                "display_track_id": int(key),
                "primary_number": before,
                "candidate_number": proposal["jersey_number"],
                "preview_number": chosen,
                "reason": reason,
                "applied": False,
            }
            tracklets[key] = {**proposal, "fusion_preview": fusion_preview[key]}

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
            "comparison_to_primary": compare_numbers(primary, standalone),
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
):
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
            grouped[str(display)].append({
                "display_track_id": int(display),
                "frame": int(crop.get("frame") or 0),
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
        return 0, int(value)
    except (TypeError, ValueError):
        return 1, str(value)


def ratio(a, b):
    return a / b if b else 0.0
