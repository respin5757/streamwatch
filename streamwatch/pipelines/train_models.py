# streamwatch/pipelines/train_models.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from streamwatch.io.gcs_utils import blob_exists, download_file, upload_dir, upload_file
from streamwatch.pipelines.config import StreamWatchConfig
from streamwatch.pipelines.manifest import utc_now_iso


RANDOM_STATE = 42
HORIZONS = [1, 2, 3, 4]

# ---- Remote layout (your chosen structure) ----
REMOTE_MODELS_HGBR_DIR = "models/hgbr"
REMOTE_MODELS_LGBM_DIR = "models/lgbm"

REMOTE_FEATURE_COLS = "data/feature_columns.json"
REMOTE_METRICS_HISTORY = "data/metrics_history.parquet"
REMOTE_TFIDF_VOCAB = "data/tfidf_vocab.json"  # optional


# -------------------------
# Dataset snapshot (metrics only)
# -------------------------
@dataclass(frozen=True)
class DatasetSnapshot:
    n_rows: int
    n_cols: int
    n_features: int
    n_unique_weeks: int
    n_unique_shows: int


def _build_snapshot(df: pd.DataFrame, n_features: int) -> DatasetSnapshot:
    ws = pd.to_datetime(df["week_start"], errors="coerce")
    return DatasetSnapshot(
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        n_features=int(n_features),
        n_unique_weeks=int(ws.dropna().nunique()),
        n_unique_shows=int(df["id"].nunique()),
    )


# -------------------------
# Feature + target prep
# -------------------------
def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["id", "week_start"]).reset_index(drop=True)
    for h in HORIZONS:
        df[f"interest_t+{h}"] = df.groupby("id", observed=True)["interest"].shift(-h)
    return df


def _build_feature_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # Encode categoricals deterministically -> numeric codes for sklearn/HGBR safety
    categorical_cols = ["status", "type", "language", "certification"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("category")
                .cat.add_categories(["__missing__"])
                .fillna("__missing__")
                .cat.codes
            )

    target_cols = [f"interest_t+{h}" for h in HORIZONS]
    exclude = {
        "id", "name", "original_name", "week_start", "label",
        "genres", "networks", "keywords", "imdb_id",
        "first_air_date", "last_air_date",
        *target_cols,
    }

    # Keep numeric-only after preprocessing
    feature_cols = [c for c in df.columns if c not in exclude and is_numeric_dtype(df[c])]
    return df, feature_cols


def _time_split(df: pd.DataFrame):
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    weeks = np.sort(df["week_start"].dropna().unique())

    if len(weeks) < 5:
        # fallback: deterministic row-order split
        cut = weeks[-1] if len(weeks) else pd.Timestamp("1970-01-01")
        mask = np.arange(len(df)) < int(len(df) * 0.8)
        tr = pd.Series(mask, index=df.index)
        va = ~tr
        return tr, va, cut

    cut = weeks[int(len(weeks) * 0.8)]
    return df["week_start"] < cut, df["week_start"] >= cut, cut


# -------------------------
# Training
# -------------------------
def _train_family(
    family: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    tr_mask: pd.Series,
    va_mask: pd.Series,
    cutoff_week,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if family == "hgbr":
        def make_model():
            return HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    elif family == "lgbm":
        def make_model():
            return lgb.LGBMRegressor(
                objective="regression",
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=31,
                min_data_in_leaf=40,
                feature_fraction=0.7,
                bagging_fraction=0.85,
                bagging_freq=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    else:
        raise ValueError(f"Unknown family={family}")

    metrics: List[Dict[str, Any]] = []

    for h in HORIZONS:
        target = f"interest_t+{h}"
        mask = df[target].notna()
        tr, va = tr_mask & mask, va_mask & mask

        Xtr, ytr = df.loc[tr, feature_cols], df.loc[tr, target]
        Xva, yva = df.loc[va, feature_cols], df.loc[va, target]

        if Xtr.empty or Xva.empty:
            raise ValueError(
                f"{family} horizon t+{h}: empty train/val split. "
                f"train_rows={Xtr.shape[0]}, val_rows={Xva.shape[0]}"
            )

        model = make_model()
        model.fit(Xtr, ytr)

        pred_tr = model.predict(Xtr)
        pred_va = model.predict(Xva)

        model_path = out_dir / f"{family}_interest_t+{h}.pkl"
        joblib.dump(model, model_path)

        metrics.append({
            "family": family,
            "horizon": int(h),
            "rmse_train": float(np.sqrt(mean_squared_error(ytr, pred_tr))),
            "rmse_val": float(np.sqrt(mean_squared_error(yva, pred_va))),
            "r2_train": float(r2_score(ytr, pred_tr)),
            "r2_val": float(r2_score(yva, pred_va)),
            "n_train": int(Xtr.shape[0]),
            "n_val": int(Xva.shape[0]),
            "cutoff_week": str(cutoff_week),
        })

    return metrics


def _append_metrics_history_to_gcs(
    cfg: StreamWatchConfig,
    records: List[Dict[str, Any]],
    work_dir: Path,
) -> Dict[str, Any]:
    """
    Append-only history stored in GCS at data/metrics_history.parquet.

    Strategy:
      - download existing parquet if present
      - concat + de-dupe on (run_id, family, horizon, created_utc)
      - upload back (overwrite)
    """
    tmp_dir = work_dir / "metrics"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_existing = tmp_dir / "metrics_history_existing.parquet"
    local_out = tmp_dir / "metrics_history.parquet"

    new_df = pd.DataFrame(records)

    if blob_exists(REMOTE_METRICS_HISTORY, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project):
        download_file(
            remote_path=REMOTE_METRICS_HISTORY,
            local_path=local_existing,
            bucket=cfg.bucket,
            prefix=cfg.prefix,
            project=cfg.project,
        )
        old_df = pd.read_parquet(local_existing)
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df

    dedupe_keys = [k for k in ["run_id", "family", "horizon", "created_utc"] if k in merged.columns]
    if dedupe_keys:
        merged = merged.drop_duplicates(subset=dedupe_keys, keep="last")

    # keep things friendly for display
    for col in ["created_utc", "run_date", "week_start"]:
        if col in merged.columns:
            merged[col] = merged[col].astype(str)

    merged.to_parquet(local_out, index=False)

    upload_file(
        local_out,
        remote_path=REMOTE_METRICS_HISTORY,
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        content_type="application/octet-stream",
    )

    return {
        "metrics_history_remote_path": REMOTE_METRICS_HISTORY,
        "metrics_rows_total": int(merged.shape[0]),
        "metrics_rows_appended": int(new_df.shape[0]),
    }


# -------------------------
# Public entrypoint
# -------------------------
def run(
    run_date: str,
    run_id: str,
    week_start: str,
    *,
    panel_clean_local_path: str,
    tfidf_vocab_local_path: str | None,
) -> Dict[str, Any]:
    cfg = StreamWatchConfig.from_env()

    df = pd.read_parquet(panel_clean_local_path)
    df = _create_targets(df)
    df, feature_cols = _build_feature_cols(df)

    tr_mask, va_mask, cutoff_week = _time_split(df)
    snapshot = _build_snapshot(df, n_features=len(feature_cols))
    ts = utc_now_iso()

    # Local work area (avoid collisions across runs)
    work = cfg.work_dir / f"run_id={run_id}" / f"week_start={week_start}" / "models"
    hgbr_dir = work / "hgbr"
    lgbm_dir = work / "lgbm"
    work.mkdir(parents=True, exist_ok=True)

    hgbr_metrics = _train_family("hgbr", df, feature_cols, tr_mask, va_mask, cutoff_week, hgbr_dir)
    lgbm_metrics = _train_family("lgbm", df, feature_cols, tr_mask, va_mask, cutoff_week, lgbm_dir)

    # Feature columns for serving
    feature_cols_path = work / "feature_columns.json"
    feature_cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    # Upload models (overwrite remote folder contents)
    upload_dir(
        local_dir=hgbr_dir,
        remote_dir=REMOTE_MODELS_HGBR_DIR,
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        include_suffixes=[".pkl"],
    )
    upload_dir(
        local_dir=lgbm_dir,
        remote_dir=REMOTE_MODELS_LGBM_DIR,
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        include_suffixes=[".pkl"],
    )

    # Upload feature columns (overwrite)
    upload_file(
        feature_cols_path,
        remote_path=REMOTE_FEATURE_COLS,
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
        content_type="application/json",
    )

    # Optional: upload tfidf vocab (overwrite)
    if tfidf_vocab_local_path:
        vocab_path = Path(tfidf_vocab_local_path)
        if vocab_path.exists():
            upload_file(
                vocab_path,
                remote_path=REMOTE_TFIDF_VOCAB,
                bucket=cfg.bucket,
                prefix=cfg.prefix,
                project=cfg.project,
                content_type="application/json",
            )

    # Build metrics records
    records: List[Dict[str, Any]] = []
    for r in (hgbr_metrics + lgbm_metrics):
        rec = dict(r)
        rec.update({
            "run_id": run_id,
            "run_date": run_date,
            "week_start": week_start,
            "created_utc": ts,
            **asdict(snapshot),
        })
        records.append(rec)

    metrics_out = _append_metrics_history_to_gcs(cfg, records, work_dir=work)

    return {
        # local outputs (for validate + publish steps)
        "hgbr_local_dir": str(hgbr_dir),
        "lgbm_local_dir": str(lgbm_dir),
        "feature_columns_local_path": str(feature_cols_path),

        # remote dirs (for publish_artifacts manifest)
        "hgbr_remote_dir": REMOTE_MODELS_HGBR_DIR,
        "lgbm_remote_dir": REMOTE_MODELS_LGBM_DIR,

        # metrics
        "metrics_records": records,
        **metrics_out,
    }
