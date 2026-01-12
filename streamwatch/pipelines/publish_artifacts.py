# streamwatch/pipelines/publish_artifacts.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from streamwatch.io.gcs_utils import upload_file, atomic_update_json_pointer
from streamwatch.pipelines.config import StreamWatchConfig


def run(
    run_date: str,
    run_id: str,
    week_start: str,
    *,
    panel_clean_local_path: str,
    feature_columns_local_path: str,
    hgbr_remote_dir: str,
    lgbm_remote_dir: str,
) -> Dict[str, Any]:
    """
    Publishes SERVING artifacts (overwrite) to:

      data/panel_clean.parquet
      data/feature_columns.json
      data/serving_manifest.json  (written last / atomic)

    Models are referenced by dir paths (typically models/hgbr and models/lgbm).
    Metrics history is handled by train_models (data/metrics_history.parquet).
    """
    cfg = StreamWatchConfig.from_env()

    # 1) Upload current data artifacts
    upload_file(
        local_path=Path(panel_clean_local_path),
        remote_path="data/panel_clean.parquet",
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        content_type="application/octet-stream",
    )

    upload_file(
        local_path=Path(feature_columns_local_path),
        remote_path="data/feature_columns.json",
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        content_type="application/json",
    )

    # 2) Write pointer last
    serving_manifest = {
        "run_id": run_id,
        "run_date": run_date,
        "week_start": week_start,
        "panel_clean_remote_path": "data/panel_clean.parquet",
        "feature_columns_remote_path": "data/feature_columns.json",
        "metrics_history_remote_path": "data/metrics_history.parquet",
        "models": {
            "hgbr_remote_dir": hgbr_remote_dir,  # e.g., "models/hgbr"
            "lgbm_remote_dir": lgbm_remote_dir,  # e.g., "models/lgbm"
        },
    }

    atomic_update_json_pointer(
        serving_manifest,
        pointer_path="data/serving_manifest.json",
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
    )

    return {
        "serving_manifest_remote_path": "data/serving_manifest.json",
        "panel_clean_remote_path": "data/panel_clean.parquet",
        "feature_columns_remote_path": "data/feature_columns.json",
        "metrics_history_remote_path": "data/metrics_history.parquet",
    }
