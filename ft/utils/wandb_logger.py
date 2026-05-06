from pathlib import Path


class WandbLogger:
    def __init__(self, run=None, enabled=False, config=None, video_id=None):
        self.run = run
        self.enabled = bool(enabled and run is not None)
        self.config = config or {}
        self.video_id = video_id
        self.failed = False

    @classmethod
    def from_config(cls, config, video_id):
        wandb_cfg = config.get("wandb", {})
        if not wandb_cfg.get("enabled", False):
            return cls(enabled=False, config=wandb_cfg, video_id=video_id)
        try:
            import wandb
        except Exception as exc:
            print(f"FT wandb: disabled, import failed: {type(exc).__name__}: {exc}")
            return cls(enabled=False, config=wandb_cfg, video_id=video_id)

        name = wandb_cfg.get("name") or video_id
        run = wandb.init(
            project=wandb_cfg.get("project") or "football-tracking",
            entity=wandb_cfg.get("entity") or None,
            name=name,
            tags=wandb_cfg.get("tags") or [],
            notes=wandb_cfg.get("notes") or None,
            config=safe_config(config),
        )
        print(f"FT wandb: run={run.name} url={getattr(run, 'url', None)}")
        return cls(run=run, enabled=True, config=wandb_cfg, video_id=video_id)

    def log(self, payload):
        if not self.enabled:
            return
        try:
            self.run.log(payload)
        except Exception as exc:
            print(f"FT wandb: log failed: {type(exc).__name__}: {exc}")

    def log_success(self, result):
        if not self.enabled:
            return
        rows = result.get("rows") or []
        assigned = [row for row in rows if row.get("player_id") not in (None, "unknown")]
        payload = {
            "status_code": 1,
            "tracklet_rows": len(rows),
            "assigned_row_count": len(assigned),
            "num_tracklet_summaries": len(result.get("summaries") or []),
            "num_candidate_scores": len(result.get("candidate_scores") or []),
        }
        self.log(payload)
        self._log_artifacts(result)
        if self.config.get("alert_on_finish", True):
            self.alert("FT run finished", f"{self.video_id} finished. Assigned rows: {len(assigned)}")

    def log_failure(self, exc):
        self.failed = True
        if not self.enabled:
            return
        message = f"{type(exc).__name__}: {exc}"
        self.log({"status_code": -1, "error": message})
        if self.config.get("alert_on_failure", True):
            self.alert("FT run failed", f"{self.video_id} failed: {message}")

    def alert(self, title, text):
        if not self.enabled:
            return
        try:
            import wandb

            wandb.alert(title=title, text=text)
        except Exception as exc:
            print(f"FT wandb: alert failed: {type(exc).__name__}: {exc}")

    def finish(self):
        if not self.enabled:
            return
        try:
            self.run.finish(exit_code=1 if self.failed else 0)
        except Exception as exc:
            print(f"FT wandb: finish failed: {type(exc).__name__}: {exc}")

    def _log_artifacts(self, result):
        if not self.config.get("log_artifacts", True):
            return
        try:
            import wandb

            artifacts_dir = Path(result["artifacts_dir"])
            artifact = wandb.Artifact(f"{self.video_id}-ft-artifacts", type="ft-artifacts")
            metadata_dir = artifacts_dir / "metadata"
            if metadata_dir.exists():
                artifact.add_dir(str(metadata_dir), name="metadata")
            if self.config.get("log_video", False):
                output_path = Path(result["output_path"])
                if output_path.exists():
                    artifact.add_file(str(output_path), name=output_path.name)
            self.run.log_artifact(artifact)
        except Exception as exc:
            print(f"FT wandb: artifact logging failed: {type(exc).__name__}: {exc}")


def safe_config(config):
    clean = {}
    for key, value in config.items():
        if key.lower() in {"api_key", "token", "password"}:
            clean[key] = "***"
        elif isinstance(value, dict):
            clean[key] = safe_config(value)
        else:
            clean[key] = value
    return clean
