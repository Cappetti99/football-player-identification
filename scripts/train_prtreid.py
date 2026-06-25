#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="FT-native identity-only PRTReID fine-tuning.")
    parser.add_argument("--dataset-dir", default="datasets/prtreid_ft_v1")
    parser.add_argument("--output-dir", default="models/reid/prtreid_ft_linking_v1")
    parser.add_argument("--initial-weights", default="models/reid/prtreid-soccernet-baseline.pth.tar")
    parser.add_argument("--hrnet-pretrained-path", default="models/reid")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing ground-truth manifest: {manifest}. Run scripts/export_prtreid_dataset.py first.")
    if not Path(args.initial_weights).is_file():
        raise SystemExit(f"Missing initial PRTReID checkpoint: {args.initial_weights}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_spec = {
        "dataset_manifest": str(manifest),
        "output_dir": str(output_dir),
        "initial_weights": args.initial_weights,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_instances": args.num_instances,
        "seed": args.seed,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "loss": "identity cross-entropy + triplet only",
        "role_team_losses": False,
        "pseudo_labels": False,
    }
    (output_dir / "training_spec.json").write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(run_spec, indent=2))
        return

    from omegaconf import OmegaConf
    from yacs.config import CfgNode as CN
    from prtreid.scripts.main import build_config, build_torchreid_model_engine

    from ft.features.prtreid import PRTReIDFeatureExtractor
    from ft.features.prtreid_dataset import install_same_team_sampler, register_ft_manifest_dataset

    disable_broken_prtreid_cuda_timers()
    dataset_name = register_ft_manifest_dataset(manifest)
    install_same_team_sampler()
    base = PRTReIDFeatureExtractor(
        enabled=True,
        weights_path=args.initial_weights,
        hrnet_pretrained_path=args.hrnet_pretrained_path,
        role_enabled=False,
    )._config_dict()
    base["project"]["job_id"] = int(os.getpid())
    base["data"]["root"] = str(dataset_dir.resolve())
    base["data"]["sources"] = [dataset_name]
    base["data"]["targets"] = [dataset_name]
    base["data"]["save_dir"] = str(output_dir.resolve())
    base["sampler"]["train_sampler"] = "PrtreidSampler"
    base["sampler"]["train_sampler_t"] = "PrtreidSampler"
    base["sampler"]["num_instances"] = int(args.num_instances)
    base["model"]["save_model_flag"] = True
    base["model"]["load_config"] = True
    base["model"]["load_weights"] = str(Path(args.initial_weights).resolve())
    base["loss"]["part_based"]["weights"]["globl"] = {"id": 1.0, "tr": 1.0}
    for key in ("foreg", "conct", "parts"):
        base["loss"]["part_based"]["weights"][key] = {"id": 0.0, "tr": 0.0}
    base["loss"]["part_based"]["weights"]["pixls"] = {"ce": 0.0}
    base["train"].update(
        {
            "batch_size": int(args.batch_size),
            "max_epoch": int(args.epochs),
            "seed": int(args.seed),
            "staged_lr": True,
            "base_lr_mult": 0.1,
            "eval_freq": 1,
        }
    )
    base["test"]["evaluate"] = False
    base["test"]["start_eval"] = 0
    base["test"]["visrank"] = False

    cfg = build_config(config=CN(OmegaConf.to_container(OmegaConf.create(base), resolve=True)))
    engine, _model = build_torchreid_model_engine(cfg)
    engine.combine_losses = types.MethodType(identity_only_loss, engine)
    checkpoint_metrics, best = train_with_early_stopping(
        engine,
        cfg,
        manifest,
        args.hrnet_pretrained_path,
        args.batch_size,
        patience=args.patience,
        min_epochs=args.min_epochs,
    )
    final_checkpoint = output_dir / "prtreid-ft-linking-v1.pth.tar"
    shutil.copy2(best["checkpoint"], final_checkpoint)
    result = {
        "status": "ok",
        "run_dir": str(cfg.data.save_dir),
        "checkpoints": checkpoint_metrics,
        "selection_metric": ["zero_false_positive_recall", "rank1", "map"],
        "selected_checkpoint": best["checkpoint"],
        "final_checkpoint": str(final_checkpoint),
    }
    (output_dir / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def disable_broken_prtreid_cuda_timers():
    """Use wall-clock meters when PRTReID reads unsynchronized CUDA events.

    The packaged TorchTimeMeter records an end event and immediately calls
    elapsed_time without synchronizing it. Recent PyTorch versions raise
    ``CUDA error: device not ready``. Timing is diagnostic only, so CPU wall
    time avoids both the error and several device synchronizations per batch.
    """
    from prtreid.utils.avgmeter import TorchTimeMeter

    if getattr(TorchTimeMeter, "_ft_cpu_timer_patch", False):
        return
    original_init = TorchTimeMeter.__init__

    def cpu_timer_init(self, name, plot=True):
        original_init(self, name, plot=plot)
        self.cuda = False

    TorchTimeMeter.__init__ = cpu_timer_init
    TorchTimeMeter._ft_cpu_timer_patch = True


def identity_only_loss(
    self,
    visibility_scores_dict,
    embeddings_dict,
    id_cls_scores_dict,
    _team_cls_scores_dict,
    _role_cls_scores_dict,
    pids,
    _teams,
    _roles,
    _pixels_cls_scores=None,
    _target_masks=None,
    bpa_weight=0,
):
    """Bypass PRTReID's hard-coded team/role objectives for FT linking fine-tuning."""
    del bpa_weight
    return self.GiLt(embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids)


def train_with_early_stopping(
    engine,
    cfg,
    manifest,
    hrnet_pretrained_path,
    batch_size,
    patience=4,
    min_epochs=10,
):
    """Train one epoch at a time and stop on the FT zero-FP validation metric."""
    state = engine.engine_state
    engine.writer.total_run_timer.start()
    state.estimated_num_batches = len(engine.train_loader)
    state.update_lr(engine.get_current_lr())
    state.training_started()
    best = None
    stale_epochs = 0
    history = []
    for epoch in range(state.start_epoch, state.max_epoch):
        engine.writer.epoch_timer.start()
        state.epoch_started()
        engine.train(fixbase_epoch=cfg.train.fixbase_epoch, open_layers=cfg.train.open_layers)
        engine.writer.epoch_timer.stop()
        state.epoch_completed()
        engine.save_model(epoch, 0.0, 0.0, 0.0, cfg.data.save_dir)
        checkpoints = sorted(Path(cfg.data.save_dir).rglob("*.pth.tar*"), key=lambda path: path.stat().st_mtime)
        if not checkpoints:
            raise RuntimeError(f"Epoch {epoch + 1} completed without a checkpoint")
        metrics = evaluate_checkpoint(checkpoints[-1], manifest, hrnet_pretrained_path, batch_size)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        score = (metrics["zero_false_positive_recall"], metrics["rank1"], metrics["map"])
        if best is None or score > best[0]:
            best = (score, metrics)
            stale_epochs = 0
        elif epoch + 1 >= int(min_epochs):
            stale_epochs += 1
        print(json.dumps({"epoch": epoch + 1, "validation": metrics, "stale_epochs": stale_epochs}, indent=2))
        if epoch + 1 >= int(min_epochs) and stale_epochs >= int(patience):
            print(f"Early stopping after epoch {epoch + 1}: no validation improvement for {patience} epochs")
            break
    state.training_completed()
    engine.writer.total_run_timer.stop()
    state.run_completed()
    engine.logger.close()
    if best is None:
        raise RuntimeError("Training completed without validation metrics")
    return history, best[1]


def evaluate_checkpoint(checkpoint, manifest, hrnet_pretrained_path, batch_size):
    from evaluate_prtreid_checkpoint import evaluate, extract_validation_rows, read_manifest
    from ft.features.prtreid import PRTReIDFeatureExtractor

    rows = [row for row in read_manifest(manifest) if row["split"] in {"query", "gallery"}]
    extractor = PRTReIDFeatureExtractor(
        enabled=True,
        weights_path=str(checkpoint),
        hrnet_pretrained_path=hrnet_pretrained_path,
        batch_size=batch_size,
        role_enabled=False,
    )
    extract_validation_rows(extractor, rows)
    metrics = evaluate(rows)
    metrics["checkpoint"] = str(checkpoint)
    return metrics


if __name__ == "__main__":
    main()
