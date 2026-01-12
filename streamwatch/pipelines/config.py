# streamwatch/pipelines/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StreamWatchConfig:
    bucket: str
    prefix: str
    project: str | None
    work_dir: Path

    @staticmethod
    def from_env() -> "StreamWatchConfig":
        bucket = os.getenv("STREAMWATCH_GCS_BUCKET")
        if not bucket:
            raise ValueError("STREAMWATCH_GCS_BUCKET is required for GCS mode.")
        prefix = os.getenv("STREAMWATCH_GCS_PREFIX", "streamwatch")
        project = os.getenv("STREAMWATCH_GCP_PROJECT")
        work_dir = Path(os.getenv("STREAMWATCH_WORK_DIR", "/tmp/streamwatch")).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        return StreamWatchConfig(bucket=bucket, prefix=prefix, project=project, work_dir=work_dir)


def run_models_root(run_id: str, week_start: str) -> str:
    return f"models/runs/run_id={run_id}/week_start={week_start}"
