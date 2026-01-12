# streamwatch/pipelines/build_panel_features.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering per show_id.
    Assumes df contains columns: week_start, interest
    """
    df = df.sort_values("week_start").copy()

    # Ensure numeric interest
    df["interest"] = pd.to_numeric(df["interest"], errors="coerce")

    for k in [1, 2, 4, 8]:
        df[f"lag_interest_{k}w"] = df["interest"].shift(k)

    df["delta_interest_1w"] = df["interest"].diff(1)
    df["roll_mean_4w"] = df["interest"].rolling(4, min_periods=1).mean()
    df["roll_std_4w"] = df["interest"].rolling(4, min_periods=2).std()

    # label: whether next week's interest beats 80th percentile of rolling 8-week window
    df["label"] = (
        df["interest"].shift(-1) > df["interest"].rolling(8, min_periods=3).quantile(0.8)
    ).astype("Int64")

    return df


def run(
    run_date: str,
    run_id: str,
    week_start: str,
    *,
    tmdb_catalog_local_path: str,
    trends_local_path: str,
    out_panel_local_path: str,
    # optional guards
    min_week_start: Optional[str] = None,  # e.g. "2020-01-01"
    max_week_start: Optional[str] = None,  # e.g. same as week_start
) -> Dict[str, Any]:
    """
    Build the base panel by joining TMDB catalog metadata onto trends time-series.

    Inputs:
      - tmdb_catalog_local_path: parquet with metadata keyed by id
      - trends_local_path: parquet with ['id','week_start','interest']
    Output:
      - out_panel_local_path: parquet panel (one file)
    """
    cat = pd.read_parquet(tmdb_catalog_local_path).copy()
    if "id" not in cat.columns:
        raise ValueError("tmdb_catalog missing required column: 'id'")
    cat["id"] = pd.to_numeric(cat["id"], errors="coerce").astype("Int64")

    # fill text cols for downstream feature creation
    for c in ["genres", "networks", "keywords"]:
        if c in cat.columns:
            cat[c] = cat[c].fillna("")

    trends = pd.read_parquet(trends_local_path).copy()
    required = {"id", "week_start", "interest"}
    missing = required - set(trends.columns)
    if missing:
        raise ValueError(f"trends parquet missing required cols: {sorted(missing)}")

    trends["id"] = pd.to_numeric(trends["id"], errors="coerce").astype("Int64")
    trends["week_start"] = pd.to_datetime(trends["week_start"], errors="coerce")
    trends["interest"] = pd.to_numeric(trends["interest"], errors="coerce")

    trends = trends.dropna(subset=["id", "week_start"])

    # Optional date filters
    if min_week_start:
        trends = trends[trends["week_start"] >= pd.to_datetime(min_week_start)]
    if max_week_start:
        trends = trends[trends["week_start"] <= pd.to_datetime(max_week_start)]

    rows = []
    # groupby keeps each show independent for time features
    for show_id, tdf in trends.groupby("id", sort=False):
        if tdf.empty:
            continue

        meta = cat.loc[cat["id"] == int(show_id)]
        if meta.empty:
            continue

        tdf = _add_time_features(tdf)

        meta_dict = meta.iloc[0].to_dict()
        for k, v in meta_dict.items():
            tdf[k] = v

        rows.append(tdf)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Final cleanup: remove infinities
    if not panel.empty:
        panel = panel.replace([np.inf, -np.inf], np.nan)

    out_path = Path(out_panel_local_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    return {
        "run_date": run_date,
        "run_id": run_id,
        "week_start": week_start,
        "panel_local_path": str(out_path),
        "n_rows": int(panel.shape[0]) if not panel.empty else 0,
        "n_cols": int(panel.shape[1]) if not panel.empty else 0,
    }
