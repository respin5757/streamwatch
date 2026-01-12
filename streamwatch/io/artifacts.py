# streamwatch/io/artifacts.py
from __future__ import annotations

import json
import os
from pathlib import Path
import streamlit as st

from streamwatch.io.gcs_utils import download_file, download_dir

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default

@st.cache_resource(show_spinner=True)
def ensure_serving_artifacts_local() -> dict:
    bucket = _env("STREAMWATCH_GCS_BUCKET")
    prefix = _env("STREAMWATCH_GCS_PREFIX", "streamwatch")
    project = _env("STREAMWATCH_GCP_PROJECT")

    if not bucket:
        return {"mode": "local"}

    cache_root = Path(".streamlit_cache") / "artifacts"
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1) Manifest
    manifest_local = cache_root / "serving_manifest.json"
    download_file(
        remote_path="data/serving_manifest.json",
        local_path=manifest_local,
        bucket=bucket,
        prefix=prefix,
        project=project,
    )
    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))

    # 2) Data artifacts
    data_dir = cache_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    panel_local = data_dir / "panel_clean.parquet"
    feats_local = data_dir / "feature_columns.json"
    metrics_local = data_dir / "metrics_history.parquet"

    download_file(manifest["panel_clean_remote_path"], panel_local, bucket=bucket, prefix=prefix, project=project)
    download_file(manifest["feature_columns_remote_path"], feats_local, bucket=bucket, prefix=prefix, project=project)

    # metrics is optional for Streamlit, but nice for a “Model Health” tab
    try:
        download_file(manifest["metrics_history_remote_path"], metrics_local, bucket=bucket, prefix=prefix, project=project)
        metrics_path = str(metrics_local)
    except Exception:
        metrics_path = None

    # 3) Models (normalize manifest paths)
    def _normalize_model_dir(p: str) -> str:
        # allow both legacy "models/..." and current "data/models/..."
        if p.startswith("data/"):
            return p
        return f"data/{p}"

    hgbr_remote = _normalize_model_dir(manifest["models"]["hgbr_remote_dir"])
    lgbm_remote = _normalize_model_dir(manifest["models"]["lgbm_remote_dir"])

    hgbr_dir = cache_root / "models_hgbr"
    lgbm_dir = cache_root / "models_lgbm"
    hgbr_dir.mkdir(parents=True, exist_ok=True)
    lgbm_dir.mkdir(parents=True, exist_ok=True)

    download_dir(hgbr_remote, hgbr_dir, bucket=bucket, prefix=prefix, project=project)
    download_dir(lgbm_remote, lgbm_dir, bucket=bucket, prefix=prefix, project=project)
