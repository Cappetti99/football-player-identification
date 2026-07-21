#!/usr/bin/env python3
"""Run TVCalib on sampled SoccerNet-GSR frames.

This script intentionally depends on the separate ``tvcalib-ft`` environment.
It writes the JSON-lines format consumed by FT's TVCalib adapter and never
modifies tracking output.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample GSR frames and estimate one TVCalib homography per frame."
    )
    parser.add_argument("--gsr-dir", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--optim-steps", type=int, default=1000)
    parser.add_argument("--sigma-scale", type=float, default=1.96)
    parser.add_argument("--segmentation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--pp-radius", type=int, default=4)
    parser.add_argument("--pp-maxdist", type=int, default=30)
    parser.add_argument("--num-points-lines", type=int, default=4)
    parser.add_argument("--num-points-circles", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def frame_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def sample_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        raise ValueError("--samples must be positive")
    if len(paths) <= count:
        return paths
    # For 750 frames and 30 samples this produces indices 0, 25, ..., 725.
    return [paths[(index * len(paths)) // count] for index in range(count)]


def to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def squeeze_temporal(value: Any, index: int) -> Any:
    value = to_python(value)
    item = value[index]
    if isinstance(item, list) and len(item) == 1:
        return item[0]
    return item


def prepare_frames(image_dir: Path, frames_dir: Path, samples: int) -> list[Path]:
    paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=frame_key,
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    selected = sample_evenly(paths, samples)
    frames_dir.mkdir(parents=True, exist_ok=True)
    expected = {path.name for path in selected}
    for stale in frames_dir.iterdir():
        if stale.is_file() and stale.name not in expected:
            stale.unlink()
    for source in selected:
        target = frames_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
    return selected


def run_sequence(args: argparse.Namespace, sequence: str) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from pytorch_lightning import seed_everything
    from torch.utils.data import DataLoader

    from sn_segmentation.src.custom_extremities import (
        generate_class_synthesis,
        get_line_extremities,
    )
    from tvcalib.cam_distr.tv_main_center import get_cam_distr
    from tvcalib.inference import (
        InferenceDatasetCalibration,
        InferenceDatasetSegmentation,
        InferenceSegmentationModel,
    )
    from tvcalib.module import TVCalibModule
    from tvcalib.utils.objects_3d import (
        SoccerPitchLineCircleSegments,
        SoccerPitchSNCircleCentralSplit,
    )

    seed_everything(seed=args.seed, workers=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_root / sequence
    output_path = output_dir / "per_sample_output.json"
    summary_path = output_dir / "summary.json"
    if args.resume and output_path.is_file() and summary_path.is_file():
        print(f"FT TVCalib: resume skip {sequence}: {output_path}", flush=True)
        return

    image_dir = args.gsr_dir / args.split / sequence / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    frames_dir = output_dir / "frames"
    selected = prepare_frames(image_dir, frames_dir, args.samples)
    with Image.open(selected[0]) as first:
        image_width, image_height = first.size
    print(
        f"FT TVCalib: sampled {len(selected)}/{len(list(image_dir.iterdir()))} images "
        f"from {image_dir} to {frames_dir}",
        flush=True,
    )

    segmentation_dataset = InferenceDatasetSegmentation(
        frames_dir, image_width, image_height
    )
    segmentation_loader = DataLoader(
        segmentation_dataset,
        batch_size=args.segmentation_batch_size,
        num_workers=0,
        shuffle=False,
    )
    segmentation_model = InferenceSegmentationModel(args.checkpoint, args.device)
    print(
        f"FT TVCalib: segmentation images={frames_dir} "
        f"size={image_width}x{image_height} device={args.device}",
        flush=True,
    )

    keypoints_by_image: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(segmentation_loader, start=1):
            masks = segmentation_model.inference(batch["image"].to(args.device)).cpu().numpy()
            for image_id, mask in zip(batch["image_id"], masks):
                buckets = generate_class_synthesis(mask.astype(np.uint8), args.pp_radius)
                mask_height, mask_width = mask.shape
                keypoints_by_image[str(image_id)] = get_line_extremities(
                    buckets,
                    args.pp_maxdist,
                    mask_width,
                    mask_height,
                    args.num_points_lines,
                    args.num_points_circles,
                )
            print(
                f"FT TVCalib: segmented batch {batch_index}/{len(segmentation_loader)}",
                flush=True,
            )

    image_ids = [path.name for path in selected]
    keypoints = [keypoints_by_image[image_id] for image_id in image_ids]
    object3d = SoccerPitchLineCircleSegments(
        device=args.device, base_field=SoccerPitchSNCircleCentralSplit()
    )
    calibration_dataset = InferenceDatasetCalibration(
        keypoints, image_width, image_height, object3d
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=len(calibration_dataset),
        num_workers=0,
        shuffle=False,
    )
    camera_distribution = get_cam_distr(
        args.sigma_scale, len(calibration_dataset), 1
    )
    model = TVCalibModule(
        object3d,
        camera_distribution,
        None,
        (image_height, image_width),
        args.optim_steps,
        args.device,
        tqdm_kwqargs={"ncols": 100},
    )

    print(f"FT TVCalib: optimizing camera for {len(calibration_dataset)} frames", flush=True)
    batch = next(iter(calibration_loader))
    losses, camera, _ = model.self_optim_batch(batch)
    parameters = camera.get_parameters(len(calibration_dataset))

    records = []
    for index, image_id in enumerate(image_ids):
        record = {"image_id": image_id}
        for key, value in parameters.items():
            record[key] = squeeze_temporal(value, index)
        for key, value in losses.items():
            record[key] = squeeze_temporal(value, index)
        record["frame"] = frame_key(Path(image_id))[0]
        record["image_width"] = image_width
        record["image_height"] = image_height
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = {
        "images_path": str(frames_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "image_width": image_width,
        "image_height": image_height,
        "records": len(records),
        "samples_requested": args.samples,
        "optim_steps": args.optim_steps,
        "sigma_scale": args.sigma_scale,
        "seed": args.seed,
        "device": args.device,
        "output": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"FT TVCalib: wrote {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    for index, sequence in enumerate(args.sequences, start=1):
        print(f"==== {sequence} ({index}/{len(args.sequences)})", flush=True)
        run_sequence(args, sequence)
    print("FT TVCalib: complete", flush=True)


if __name__ == "__main__":
    main()
