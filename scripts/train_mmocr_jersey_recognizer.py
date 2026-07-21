#!/usr/bin/env python3
"""Fine-tune MMOCR SAR on manually verified GSR jersey-number crops."""

import argparse
import hashlib
import json
import os
from pathlib import Path


DEFAULT_CHECKPOINT = (
    "https://download.openmmlab.com/mmocr/textrecog/sar/"
    "sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real/"
    "sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real_20220915_171910-04eb4e75.pth"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--load-from", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    validate_args(args)

    # PyTorch deterministic algorithms require a deterministic CuBLAS
    # workspace on CUDA >= 10.2. This must be set before importing torch via
    # MMEngine/MMOCR.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmocr.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    data = Path(args.data).resolve()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("frozen_sequences_observed"):
        raise ValueError("dataset contains frozen sequences")
    base_config = Path(args.base_config).resolve() if args.base_config else discover_sar_config()
    cfg = Config.fromfile(str(base_config))
    train_pipeline = cfg.get("train_pipeline")
    test_pipeline = cfg.get("test_pipeline")
    if not train_pipeline or not test_pipeline:
        raise ValueError(f"SAR base config lacks train/test pipelines: {base_config}")

    cfg.train_dataloader = dataloader(
        data, "train.json", train_pipeline, args.batch_size, args.num_workers, shuffle=True
    )
    cfg.val_dataloader = dataloader(
        data, "validation.json", test_pipeline, args.batch_size, args.num_workers, shuffle=False
    )
    cfg.test_dataloader = cfg.val_dataloader
    cfg.val_evaluator = dict(type="WordMetric", mode=["exact", "ignore_case"])
    cfg.test_evaluator = cfg.val_evaluator
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=args.epochs, val_interval=1)
    cfg.val_cfg = dict(type="ValLoop")
    cfg.test_cfg = dict(type="TestLoop")
    cfg.optim_wrapper = dict(
        type="OptimWrapper",
        optimizer=dict(
            type="AdamW", lr=args.learning_rate, weight_decay=args.weight_decay
        ),
        clip_grad=dict(max_norm=5.0, norm_type=2),
    )
    cfg.param_scheduler = [
        dict(
            type="CosineAnnealingLR", T_max=args.epochs,
            eta_min=args.learning_rate * 0.05, by_epoch=True,
        )
    ]
    cfg.default_hooks = dict(
        timer=dict(type="IterTimerHook"),
        logger=dict(type="LoggerHook", interval=10),
        param_scheduler=dict(type="ParamSchedulerHook"),
        checkpoint=dict(
            type="CheckpointHook", interval=1, max_keep_ckpts=3,
            save_best="auto", rule="greater",
        ),
        sampler_seed=dict(type="DistSamplerSeedHook"),
        visualization=dict(type="VisualizationHook", enable=False),
    )
    cfg.custom_hooks = []
    if args.early_stopping > 0:
        cfg.custom_hooks.append(dict(
            type="EarlyStoppingHook",
            monitor="recog/word_acc",
            rule="greater",
            min_delta=0.001,
            patience=args.early_stopping,
            strict=True,
        ))
    # Torch 2.0 has no deterministic CUDA implementation for the 2D NLL loss
    # used by SAR. Keep every controllable seed and deterministic CuBLAS, but
    # do not make that unsupported kernel a fatal error.
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.load_from = args.load_from
    cfg.resume = False
    cfg.launcher = "none"
    cfg.env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
        dist_cfg=dict(backend="nccl"),
    )
    cfg.log_level = "INFO"
    cfg.visualizer = dict(
        type="TextRecogLocalVisualizer",
        name="visualizer",
        vis_backends=[dict(type="LocalVisBackend")],
    )
    cfg.device = args.device

    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(work / "resolved_config.py"))
    provenance = {
        "task": "jersey_number_text_recognition",
        "architecture": "SAR ResNet31 parallel decoder",
        "base_config": str(base_config),
        "load_from": args.load_from,
        "dataset": str(data),
        "dataset_manifest_sha256": sha256(data / "manifest.json"),
        "training_parameters": vars(args),
        "train_sequences": manifest["train_sequences"],
        "validation_sequences": manifest["validation_sequences"],
        "frozen_sequences": manifest["frozen_sequences"],
        "seed": args.seed,
        "determinism": {
            "seeded": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_deterministic_algorithms": False,
            "reason": "Torch 2.0 SAR nll_loss2d CUDA kernel has no deterministic implementation",
        },
    }
    write_json(work / "ft_training_metadata.json", provenance)
    runner = Runner.from_cfg(cfg)
    runner.train()
    checkpoints = sorted(work.glob("*.pth"))
    provenance["checkpoints"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in checkpoints
    }
    last_checkpoint = work / "last_checkpoint"
    provenance["last_checkpoint"] = (
        last_checkpoint.read_text().strip() if last_checkpoint.is_file() else None
    )
    write_json(work / "ft_training_metadata.json", provenance)
    print(json.dumps(provenance, indent=2))


def dataloader(root, annotation, pipeline, batch_size, workers, shuffle):
    return dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        sampler=dict(type="DefaultSampler", shuffle=shuffle),
        dataset=dict(
            # MMOCR 1.0.x registers the generic JSON-backed dataset as
            # OCRDataset. Text recognition is selected by the pipeline and
            # the annotation metainfo, not by a TextRecogDataset class.
            type="OCRDataset",
            data_root=str(root),
            ann_file=annotation,
            data_prefix=dict(img_path=""),
            pipeline=pipeline,
        ),
    )


def discover_sar_config():
    import mmocr

    root = Path(mmocr.__file__).resolve().parent
    path = root / ".mim/configs/textrecog/sar/sar_resnet31_parallel-decoder_5e_st-sub_mj-sub_sa_real.py"
    if not path.is_file():
        raise FileNotFoundError(f"installed MMOCR SAR config not found: {path}")
    return path


def validate_args(args):
    if (
        args.epochs <= 0 or args.batch_size <= 0
        or args.num_workers < 0 or args.early_stopping < 0
    ):
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not str(args.device).startswith("cuda"):
        raise ValueError("this reproducible training profile requires an explicit CUDA device")
    data = Path(args.data)
    for name in ("manifest.json", "train.json", "validation.json"):
        if not (data / name).is_file():
            raise FileNotFoundError(f"dataset file missing: {data / name}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
