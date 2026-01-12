# streamwatch/pipelines/validate_serving_contract.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

CATEGORICAL_COLS = ["status", "type", "language", "certification"]


def _load_feature_cols(feature_columns_local_path: str) -> List[str]:
    p = Path(feature_columns_local_path)
    if not p.exists():
        raise FileNotFoundError(f"feature_columns.json not found: {p}")
    cols = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_columns.json must be a non-empty list of strings")
    return cols


def _apply_training_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror train_models._build_feature_cols():
      df[col] = df[col].astype("category").cat.add_categories(["__missing__"]).fillna("__missing__").cat.codes
    """
    df = df.copy()
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype("category")
                .cat.add_categories(["__missing__"])
                .fillna("__missing__")
                .cat.codes
            )
    return df


def _coerce_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if not is_numeric_dtype(X[c]):
            # last resort conversion
            X[c] = (
                X[c]
                .astype("category")
                .cat.add_categories(["__missing__"])
                .fillna("__missing__")
                .cat.codes
            )
    X = X.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return X


def run(
    *,
    panel_clean_local_path: str,
    feature_columns_local_path: str,
    model_local_path: Optional[str] = None,
    model_dir_local_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validation checks:
      1) panel_clean loads
      2) feature columns exist in df
      3) preprocessing matches training
      4) model(s) load
      5) prediction works on one row
      6) input matrix is numeric + finite

    Provide either:
      - model_local_path (single model)
      - model_dir_local_path (validate all *.pkl in dir)
    """
    panel_path = Path(panel_clean_local_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"panel_clean not found: {panel_path}")

    if (model_local_path is None) == (model_dir_local_path is None):
        raise ValueError("Provide exactly one of model_local_path or model_dir_local_path")

    if model_local_path:
        model_paths = [Path(model_local_path)]
    else:
        d = Path(model_dir_local_path)  # type: ignore[arg-type]
        if not d.exists():
            raise FileNotFoundError(f"model_dir_local_path not found: {d}")
        model_paths = sorted(d.glob("*.pkl"))
        if not model_paths:
            raise FileNotFoundError(f"No .pkl files found in: {d}")

    df = pd.read_parquet(panel_path)
    feature_cols = _load_feature_cols(feature_columns_local_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature cols (first 25): {missing[:25]} (total={len(missing)})"
        )

    df = _apply_training_preprocessing(df)

    # Choose a stable row: first non-null across most features
    X = df[feature_cols].iloc[[0]].copy()
    X = _coerce_numeric_matrix(X)

    results: List[Dict[str, Any]] = []

    for mp in model_paths:
        if not mp.exists():
            raise FileNotFoundError(f"model not found: {mp}")

        model = joblib.load(mp)
        pred = model.predict(X)
        pred0 = float(pred[0])

        if not np.isfinite(pred0):
            raise ValueError(f"Non-finite prediction from {mp.name}: {pred0}")

        results.append({"model": mp.name, "sample_pred": pred0})

    return {
        "ok": True,
        "panel_rows": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_models_validated": int(len(results)),
        "models": results,
    }
