import os
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ft.export.artifacts import write_json


class RunDiagnostics:
    def __init__(self, artifacts_dir, video_id):
        self.artifacts_dir = Path(artifacts_dir)
        self.video_id = video_id
        self.metadata_dir = self.artifacts_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.stages = []

    @contextmanager
    def stage(self, name):
        start_time = time.perf_counter()
        start_disk = directory_size(self.artifacts_dir)
        status = "ok"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            end_disk = directory_size(self.artifacts_dir)
            record = {
                "stage": name,
                "status": status,
                "seconds": round(elapsed, 3),
                "artifacts_bytes_before": start_disk,
                "artifacts_bytes_after": end_disk,
                "artifacts_bytes_delta": end_disk - start_disk,
            }
            self.stages.append(record)
            print(
                f"FT stage {name}: status={status} seconds={record['seconds']}"
                f" artifacts_delta_mb={record['artifacts_bytes_delta'] / 1048576.0:.2f}",
                flush=True,
            )

    def write_manifest(self, config):
        payload = {
            "video_id": self.video_id,
            "started_at": self.started_at,
            "system": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "git": git_snapshot(),
            "config": config,
        }
        write_json(payload, self.metadata_dir / f"{self.video_id}_run_manifest.json")

    def write_summary(self, extra=None):
        finished_at = datetime.now(timezone.utc).isoformat()
        total_seconds = sum(stage["seconds"] for stage in self.stages)
        payload = {
            "video_id": self.video_id,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "total_stage_seconds": round(total_seconds, 3),
            "artifacts_bytes": directory_size(self.artifacts_dir),
            "stages": self.stages,
        }
        if extra:
            payload.update(extra)
        write_json(payload, self.metadata_dir / f"{self.video_id}_run_diagnostics.json")
        return payload


def directory_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def git_snapshot():
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(_git(["status", "--short"])),
    }


def _git(args):
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
