# streamwatch/pipelines/metrics_store.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from streamwatch.io.gcs_utils import blob_exists, download_file, upload_file
from streamwatch.pipelines.config import StreamWatchConfig


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_metrics_history(
    records: List[Dict[str, Any]],
    *,
    remote_path: str = "data/metrics_history.parquet",
) -> Dict[str, Any]:
    """
    Append-only metrics history (the only historical artifact).
    """
    if not records:
        return {"ok": True, "n_appended": 0, "remote_path": remote_path}

    cfg = StreamWatchConfig.from_env()

    tmp = cfg.work_dir / "_metrics_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    existing_local = tmp / "metrics_history_existing.parquet"
    out_local = tmp / "metrics_history_out.parquet"

    if blob_exists(remote_path, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project):
        download_file(remote_path, existing_local, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project)
        hist = pd.read_parquet(existing_local)
    else:
        hist = pd.DataFrame()

    new = pd.DataFrame(records)
    if "ts" not in new.columns:
        new["ts"] = utc_now_iso()

    out = pd.concat([hist, new], ignore_index=True) if not hist.empty else new
    out.to_parquet(out_local, index=False)

    upload_file(
        out_local,
        remote_path=remote_path,
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        content_type="application/octet-stream",
    )

    return {"ok": True, "remote_path": remote_path, "n_total": int(out.shape[0]), "n_appended": int(new.shape[0])}
