# app.py
import json
import os
from pathlib import Path
from typing import Optional

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from streamwatch.pipelines.config import StreamWatchConfig
from streamwatch.io.gcs_utils import download_file, download_dir

# ================== CONSTANTS ==================
HORIZONS = [1, 2, 3, 4]
ENCODED_CATEGORICAL_COLS = ["status", "type", "language", "certification"]

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def resolve_artifacts_from_gcs() -> dict:
    """
    GCS-only artifact resolver for Streamlit.

    Requires env vars (set in Streamlit Cloud / wherever you deploy):
      STREAMWATCH_GCS_BUCKET
      STREAMWATCH_GCS_PREFIX
      STREAMWATCH_GCP_PROJECT
      and auth via either:
        - STREAMWATCH_GCP_SA_JSON (recommended)  OR
        - GOOGLE_APPLICATION_CREDENTIALS / ADC
    """
    cfg = StreamWatchConfig.from_env()

    # Local cache path (Streamlit needs files on disk)
    cache_root = Path(os.getenv("STREAMWATCH_ARTIFACT_CACHE_DIR", "/tmp/streamwatch_artifacts"))
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1) Serving manifest is the single source of truth
    manifest_local = cache_root / "serving_manifest.json"
    download_file(
        remote_path="data/serving_manifest.json",
        local_path=manifest_local,
        bucket=cfg.bucket,
        prefix=cfg.prefix,
        project=cfg.project,
    )
    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))

    # 2) Download required serving artifacts
    panel_remote = manifest.get("panel_clean_remote_path", "data/panel_clean.parquet")
    feat_remote = manifest.get("feature_columns_remote_path", "data/feature_columns.json")
    metrics_remote = manifest.get("metrics_history_remote_path", "data/metrics_history.parquet")

    panel_local = cache_root / "panel_clean.parquet"
    feature_local = cache_root / "feature_columns.json"
    metrics_local = cache_root / "metrics_history.parquet"

    download_file(panel_remote, panel_local, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project)
    download_file(feat_remote, feature_local, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project)

    # metrics history might not exist on first-ever run; don't crash the app
    metrics_exists = True
    try:
        download_file(metrics_remote, metrics_local, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project)
    except Exception:
        metrics_exists = False

    # 3) Download models dirs
    hgbr_remote_dir = manifest.get("models", {}).get("hgbr_remote_dir", "data/models/hgbr")
    lgbm_remote_dir = manifest.get("models", {}).get("lgbm_remote_dir", "data/models/lgbm")

    hgbr_local_dir = cache_root / "models" / "hgbr"
    lgbm_local_dir = cache_root / "models" / "lgbm"
    hgbr_local_dir.mkdir(parents=True, exist_ok=True)
    lgbm_local_dir.mkdir(parents=True, exist_ok=True)

    download_dir(hgbr_remote_dir, hgbr_local_dir, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project)
    download_dir(lgbm_remote_dir, lgbm_local_dir, bucket=cfg.bucket, prefix=cfg.prefix, project=cfg.project)

    return {
        "mode": "gcs",
        "manifest": manifest,
        "panel_clean": str(panel_local),
        "feature_columns": str(feature_local),
        "metrics_history": str(metrics_local) if metrics_exists else "",
        "hgbr_dir": str(hgbr_local_dir),
        "lgbm_dir": str(lgbm_local_dir),
        "cache_root": str(cache_root),
    }


# ================== ARTIFACT PATHS (GCS ONLY) ==================
artifact_info = resolve_artifacts_from_gcs()

DATA_PATH = Path(artifact_info["panel_clean"])
FEATURE_COLS_PATH = Path(artifact_info["feature_columns"])
METRICS_HISTORY_PATH = (
    Path(artifact_info.get("metrics_history", "")) if artifact_info.get("mode") == "gcs" else None
)

HGBR_DIR = Path(artifact_info["hgbr_dir"])
LGBM_DIR = Path(artifact_info["lgbm_dir"])


# ================== STREAMLIT PAGE CONFIG ==================
st.set_page_config(page_title="StreamWatch — TV Popularity Forecasts", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.25rem; padding-bottom: 2.25rem; max-width: 1400px; }
.stCaption { color: rgba(255,255,255,0.62); }
.small-muted { color: rgba(255,255,255,0.62); font-size: 0.92rem; }
.kicker { font-size: 0.95rem; color: rgba(255,255,255,0.62); margin-bottom: -0.25rem; }
.hr { margin: 0.85rem 0 1.1rem 0; border-bottom: 1px solid rgba(255,255,255,0.10); }
[data-testid="stMetricValue"] { font-size: 1.7rem; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ================== UI HELPERS ==================
def section(title: str, subtitle: Optional[str] = None):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def info_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (k, v) in zip(cols, items):
        col.markdown(f"<div class='kicker'>{k}</div><div><b>{v}</b></div>", unsafe_allow_html=True)


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


# ---- UI-only display label maps (do NOT rename underlying model columns) ----
MOVER_LABELS = {
    "name": "Show",
    "last_interest": "Latest interest",
    "predicted_t+1": "Forecast (t+1)",
    "delta_t+1": "Net change (t+1)",
}

OVERVIEW_LABELS = {
    "rank_popularity": "Popularity rank",
    "name": "Show",
    "last_interest": "Latest interest",
    "predicted_t+1": "Forecast (t+1)",
    "delta_t+1": "Net change (t+1)",
}

KW_LIFT_TABLE_LABELS = {
    "keyword_readable": "Keyword",
    "coverage_%": "Coverage (%)",
    "mean_interest_present": "Mean interest (keyword present)",
    "mean_interest_absent": "Mean interest (keyword absent)",
    "lift_present_minus_absent": "Lift (present − absent)",
}

KW_COV_TABLE_LABELS = {
    "keyword_readable": "Keyword",
    "coverage_%": "Coverage (%)",
    "mean_interest_present": "Mean interest (present)",
}

MOVER_FORMAT_MAP = {
    MOVER_LABELS["last_interest"]: "{:.1f}",
    MOVER_LABELS["predicted_t+1"]: "{:.1f}",
    MOVER_LABELS["delta_t+1"]: "{:+.1f}",
}

OVERVIEW_FORMAT_MAP = {
    OVERVIEW_LABELS["rank_popularity"]: "{:d}",
    OVERVIEW_LABELS["last_interest"]: "{:.1f}",
    OVERVIEW_LABELS["predicted_t+1"]: "{:.1f}",
    OVERVIEW_LABELS["delta_t+1"]: "{:+.1f}",
}

def show_label_by_id(show_id: int) -> str:
    try:
        m = show_catalog.loc[show_catalog["id"] == show_id, "name"]
        if m is None or len(m) == 0:
            return f"(unknown) (id={show_id})"
        name = str(m.iloc[0])
        return f"{name} (id={show_id})"
    except Exception:
        return f"(unknown) (id={show_id})"


def _display_cols(df: pd.DataFrame, cols: list[str], labels: dict[str, str]) -> pd.DataFrame:
    """Return a UI-only display copy with renamed headers."""
    out = df[cols].copy()
    return out.rename(columns={c: labels.get(c, c) for c in cols})


def style_delta_table(
    df: pd.DataFrame,
    delta_col: str,
    format_map: Optional[dict[str, str]] = None,
) -> "pd.io.formats.style.Styler":
    """
    Deterministic table styling:
      - colors a specified delta column
      - uses an explicit format_map (recommended) so UI renames never break formatting
    """
    if df is None:
        return df.style
    if df.empty:
        return df.style
    if delta_col not in df.columns:
        return df.style.format(format_map or {}, na_rep="—")

    def _color_delta(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v > 0:
            return "color: #16a34a; font-weight: 700;"
        if v < 0:
            return "color: #dc2626; font-weight: 700;"
        return "color: rgba(255,255,255,0.72);"

    fmt = format_map or {}

    return (
        df.style
        .format(fmt, na_rep="—")
        .applymap(_color_delta, subset=[delta_col])
    )


# ================== SIDEBAR (OPS) ==================
with st.sidebar:
    st.header("Ops")
    st.markdown(
        f"""
<div class="small-muted">
<b>Artifact mode:</b> {artifact_info.get("mode", "unknown")}<br/>
<b>Panel:</b> <code>{str(DATA_PATH)}</code>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    if st.button("Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared.")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.caption("Shareable URLs supported: show_id + horizon stored in query params.")


# ================== LOAD DATA ==================
@st.cache_data
def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    if "week_start" in df.columns:
        df["week_start"] = _ensure_dt(df["week_start"])
    return df


panel = load_panel()

# ================== PANEL SUMMARY FOR OPS ==================
@st.cache_data
def panel_summary(df: pd.DataFrame) -> dict:
    out = {
        "n_rows": int(len(df)),
        "n_shows": int(df["id"].nunique()) if "id" in df.columns else 0,
    }
    if "week_start" in df.columns:
        out["min_week"] = _ensure_dt(df["week_start"]).min()
        out["max_week"] = _ensure_dt(df["week_start"]).max()
    if "interest" in df.columns:
        out["pct_interest_missing"] = float(df["interest"].isna().mean() * 100.0)
    return out


ps = panel_summary(panel)
with st.sidebar:
    st.markdown(
        f"""
<div class="small-muted">
<b>Coverage:</b> {ps.get("n_shows", 0):,} shows • {ps.get("n_rows", 0):,} rows<br/>
<b>Weeks:</b> {str(ps.get("min_week", ""))[:10]} → {str(ps.get("max_week", ""))[:10]}<br/>
<b>Missing interest:</b> {ps.get("pct_interest_missing", float("nan")):.1f}%
</div>
""",
        unsafe_allow_html=True,
    )

# ================== LOAD FEATURES + MODELS ==================
def _load_model_set(folder: Path, file_prefix: str) -> dict[int, object]:
    models: dict[int, object] = {}
    for h in HORIZONS:
        path = folder / f"{file_prefix}_interest_t+{h}.pkl"
        if path.exists():
            models[h] = joblib.load(path)
    return models


@st.cache_resource
def load_features_and_models():
    feature_path = FEATURE_COLS_PATH
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature_columns.json at: {feature_path}")

    with open(feature_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    models_lgbm = _load_model_set(LGBM_DIR, "lgbm")
    if not models_lgbm:
        raise FileNotFoundError(
            f"No LightGBM model files found in: {LGBM_DIR}. Expected lgbm_interest_t+{{1..4}}.pkl"
        )

    return {"lightgbm": feature_cols}, {"lightgbm": models_lgbm}


FEATURE_COLS, MODELS = load_features_and_models()

# ================== SHOW CATALOG ==================
@st.cache_data
def build_show_catalog(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    cols = [
        "id",
        "name",
        "original_name",
        "status",
        "type",
        "number_of_seasons",
        "number_of_episodes",
        "popularity",
        "vote_average",
        "vote_count",
        "language",
        "certification",
        "genres",
        "networks",
        "first_air_date",
        "last_air_date",
        "poster_url",
        "poster_path_full",
    ]
    available_cols = [c for c in cols if c in df.columns]
    if not {"id", "week_start"}.issubset(df.columns):
        cat = df.drop_duplicates(subset=["id"])[available_cols].copy()
        if "popularity" in cat.columns:
            cat = cat.sort_values("popularity", ascending=False).head(top_n)
        return cat.reset_index(drop=True)

    cat = (
        df.sort_values(["id", "week_start"])
        .groupby("id", as_index=False)
        .tail(1)[available_cols]
    )
    if "popularity" in cat.columns:
        cat = cat.sort_values("popularity", ascending=False).head(top_n)
    return cat.reset_index(drop=True)


show_catalog = build_show_catalog(panel, top_n=200)

# ================== FEATURE BUILDING (STABLE ENCODING) ==================
@st.cache_data
def build_category_encoders(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[object, int]]:
    """
    Build stable category-to-int mappings from the panel data.
    Unknowns -> -1.
    """
    encoders: dict[str, dict[object, int]] = {}
    for c in cols:
        if c not in df.columns:
            continue
        vals = df[c].dropna().astype(str).unique().tolist()
        vals = sorted(vals)
        encoders[c] = {v: i for i, v in enumerate(vals)}
    return encoders


CATEGORY_ENCODERS = build_category_encoders(panel, ENCODED_CATEGORICAL_COLS)


def build_model_input_lgbm(row_or_df: pd.Series | pd.DataFrame) -> pd.DataFrame:
    feature_cols = FEATURE_COLS["lightgbm"]

    if isinstance(row_or_df, pd.Series):
        X = row_or_df.to_frame().T.copy()
    else:
        X = row_or_df.copy()

    # Stable categorical encoding (matches training style better than per-row cat.codes)
    for col in ENCODED_CATEGORICAL_COLS:
        if col in X.columns:
            mapping = CATEGORY_ENCODERS.get(col, {})
            raw = X[col]
            # preserve NaNs as unknowns
            X[col] = (
                raw.astype(str)
                .where(~raw.isna(), None)
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

    # Ensure required features exist
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0

    # Keep only model columns, in correct order
    X = X[feature_cols].copy()

    # Coerce numeric
    for c in X.columns:
        if not is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X.fillna(0.0)


def predict_show_interest(show_id: int, max_horizon: int = 4) -> dict:
    """
    Hardened prediction wrapper:
      - handles missing week_start/interest columns gracefully
      - avoids crashing the whole app on edge-case shows
    """
    model_set = MODELS["lightgbm"]

    if "id" not in panel.columns:
        raise ValueError("Panel is missing required column: id")

    df_show = panel[panel["id"] == show_id].copy()
    if df_show.empty:
        raise ValueError(f"Show id {show_id} not found in panel.")

    # sort by week_start if available
    if "week_start" in df_show.columns:
        df_show = df_show.sort_values("week_start")
    latest = df_show.iloc[-1].copy()

    X = build_model_input_lgbm(latest)

    preds: dict[str, float] = {}
    for h in range(1, int(max_horizon) + 1):
        if h in model_set:
            raw = float(model_set[h].predict(X)[0])
            preds[f"t+{h}"] = float(np.clip(raw, 0, 100))

    # History DF for plotting
    if {"week_start", "interest"}.issubset(df_show.columns):
        history_df = df_show[["week_start", "interest"]].copy()
        history_df["week_start"] = _ensure_dt(history_df["week_start"])
        history_df["interest"] = pd.to_numeric(history_df["interest"], errors="coerce")
    else:
        # fallback: create a minimal frame to avoid downstream crashes
        history_df = pd.DataFrame({"week_start": pd.to_datetime([]), "interest": []})

    latest_week = latest.get("week_start", pd.NaT)
    if "week_start" in df_show.columns:
        latest_week = _ensure_dt(pd.Series([latest_week])).iloc[0]

    return {
        "show_id": int(show_id),
        "latest_week": latest_week,
        "model_type": "lightgbm",
        "predictions": preds,
        "history_df": history_df,
        "latest_row": latest,
        "X_latest": X,
    }


# ================== SCANS (FAST BATCH) ==================
@st.cache_data
def compute_trending_all() -> pd.DataFrame:
    model_set = MODELS["lightgbm"]
    if 1 not in model_set:
        return pd.DataFrame()

    if not {"id", "week_start"}.issubset(panel.columns):
        return pd.DataFrame()

    latest_rows = (
        panel.sort_values(["id", "week_start"])
        .groupby("id", as_index=False)
        .tail(1)
        .copy()
    )
    if latest_rows.empty or "interest" not in latest_rows.columns:
        return pd.DataFrame()

    latest_rows["last_interest"] = pd.to_numeric(latest_rows["interest"], errors="coerce")
    latest_rows = latest_rows[np.isfinite(latest_rows["last_interest"]) & (latest_rows["last_interest"] > 0)].copy()
    if latest_rows.empty:
        return pd.DataFrame()

    X_all = build_model_input_lgbm(latest_rows)
    raw_pred = model_set[1].predict(X_all)
    pred = np.clip(np.asarray(raw_pred, dtype=float), 0, 100)

    out = pd.DataFrame(
        {
            "id": latest_rows["id"].astype(int).values,
            "name": latest_rows.get("name", pd.Series([""] * len(latest_rows))).astype(str).values,
            "last_interest": latest_rows["last_interest"].astype(float).values,
            "predicted_t+1": pred,
        }
    )
    out["delta_t+1"] = out["predicted_t+1"] - out["last_interest"]

    out["popularity"] = pd.to_numeric(latest_rows.get("popularity", np.nan), errors="coerce")
    out["vote_average"] = pd.to_numeric(latest_rows.get("vote_average", np.nan), errors="coerce")

    return out.sort_values("delta_t+1", ascending=False).reset_index(drop=True)


@st.cache_data
def compute_predictions_all() -> pd.DataFrame:
    df = compute_trending_all()
    if df.empty:
        return df

    # show_catalog already top_n; merge if available
    if {"id", "popularity"}.issubset(show_catalog.columns):
        df = df.merge(show_catalog[["id", "popularity"]], on="id", how="left", suffixes=("", "_catalog"))

    df = df.sort_values("popularity", ascending=False).reset_index(drop=True)
    df["rank_popularity"] = np.arange(1, len(df) + 1)
    return df


# ================== METRICS ==================
@st.cache_data
def load_metrics_history() -> pd.DataFrame:
    if METRICS_HISTORY_PATH is None or not str(METRICS_HISTORY_PATH):
        return pd.DataFrame()
    p = Path(METRICS_HISTORY_PATH)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_parquet(p)

    if "created_utc" in df.columns:
        df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)
    if "run_date" in df.columns:
        df["run_date"] = df["run_date"].astype(str)
    if "week_start" in df.columns:
        df["week_start"] = df["week_start"].astype(str)
    if "horizon" in df.columns:
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")

    return df


metrics_df = load_metrics_history()

# ================== EXPLAINABILITY ==================
def _try_get_lgbm_feature_importance(model, feature_names: list[str], top_k: int = 20) -> pd.DataFrame:
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            df = pd.DataFrame({"feature": feature_names, "importance": imp})
            return df.sort_values("importance", ascending=False).head(top_k)
        if hasattr(model, "booster_") and hasattr(model.booster_, "feature_importance"):
            imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
            df = pd.DataFrame({"feature": feature_names, "importance": imp})
            return df.sort_values("importance", ascending=False).head(top_k)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _try_get_local_contributions(model, X: pd.DataFrame, feature_names: list[str], top_k: int = 12) -> pd.DataFrame:
    try:
        contrib = model.predict(X, pred_contrib=True)
        contrib = np.asarray(contrib).reshape(1, -1)
        if contrib.shape[1] == len(feature_names) + 1:
            vals = contrib[0, :-1]
            bias = float(contrib[0, -1])
        else:
            return pd.DataFrame()

        df = pd.DataFrame({"feature": feature_names, "contribution": vals})
        df["abs"] = df["contribution"].abs()
        df = df.sort_values("abs", ascending=False).drop(columns=["abs"]).head(top_k)
        df["direction"] = np.where(df["contribution"] >= 0, "push up", "push down")
        df.attrs["bias"] = bias
        return df
    except Exception:
        return pd.DataFrame()


# ================== KEYWORD INSIGHTS ==================
@st.cache_data
def keyword_insights(df: pd.DataFrame, prefix: str = "kw_") -> dict[str, pd.DataFrame]:
    if "interest" not in df.columns:
        return {"lift": pd.DataFrame(), "coverage": pd.DataFrame()}

    kw_cols = [c for c in df.columns if c.lower().startswith(prefix.lower())]
    if not kw_cols:
        return {"lift": pd.DataFrame(), "coverage": pd.DataFrame()}

    base = pd.to_numeric(df["interest"], errors="coerce")
    base_mean = float(np.nanmean(base.values))

    rows = []
    cov_rows = []

    for c in kw_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0)
        present = s > 0

        if float(present.mean()) < 0.01:
            continue

        mean_present = float(np.nanmean(base[present].values)) if present.any() else np.nan
        mean_absent = float(np.nanmean(base[~present].values)) if (~present).any() else np.nan
        lift_vs_absent = mean_present - mean_absent if np.isfinite(mean_present) and np.isfinite(mean_absent) else np.nan

        rows.append(
            {
                "keyword": c,
                "coverage_%": float(present.mean() * 100.0),
                "mean_interest_present": mean_present,
                "mean_interest_absent": mean_absent,
                "lift_present_minus_absent": lift_vs_absent,
                "lift_vs_global_mean": (mean_present - base_mean) if np.isfinite(mean_present) else np.nan,
            }
        )

        cov_rows.append(
            {
                "keyword": c,
                "coverage_%": float(present.mean() * 100.0),
                "mean_interest_present": mean_present,
            }
        )

    lift_df = pd.DataFrame(rows).sort_values("lift_present_minus_absent", ascending=False)
    cov_df = pd.DataFrame(cov_rows).sort_values(["coverage_%", "mean_interest_present"], ascending=False)

    return {"lift": lift_df.reset_index(drop=True), "coverage": cov_df.reset_index(drop=True)}


# ================== QUERY PARAMS (SHAREABLE URL) ==================
def _get_int_qp(key: str, default: Optional[int] = None) -> Optional[int]:
    try:
        qp = st.query_params
        if key not in qp:
            return default
        v = qp.get(key)
        if isinstance(v, list):
            v = v[0] if v else None
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _set_qp(show_id: int, horizon: int):
    try:
        st.query_params["show_id"] = str(show_id)
        st.query_params["h"] = str(horizon)
    except Exception:
        pass


# ================== APP HEADER ==================
st.title("StreamWatch")
st.caption(
    "Forecast weekly Google Trends interest for TV shows 1–4 weeks ahead with a tuned LightGBM model. "
    "Explore a single show, compare shows, or scan the catalog for movers."
)

# ================== TABS ==================
tab_dash, tab_models, tab_about = st.tabs(["Dashboard", "Modeling & data", "About"])

# ======================================================================================
# TAB 1 — DASHBOARD
# ======================================================================================
with tab_dash:
    # ---------- CONTROL STRIP ----------
    controls = st.container(border=True)
    with controls:
        c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="center")

        default_show_id = _get_int_qp("show_id", None)
        default_h = int(np.clip(_get_int_qp("h", 4) or 4, 1, 4))

        show_names = (
            show_catalog[["id", "name", "popularity"]]
            .copy()
            .sort_values("popularity", ascending=False)
        )
        show_names["label"] = show_names["name"].astype(str) + " (id=" + show_names["id"].astype(str) + ")"
        labels = show_names["label"].tolist()

        default_index = 0
        if default_show_id is not None and "id" in show_names.columns:
            match = show_names.index[show_names["id"] == default_show_id].tolist()
            if match:
                default_index = int(match[0])

        with c1:
            selected_label = st.selectbox("Show", options=labels, index=default_index)
            # robust parse: extract inside "(id=...)"
            try:
                selected_id = int(selected_label.split("id=")[-1].strip(")"))
            except Exception:
                selected_id = int(show_names.iloc[0]["id"])

        with c2:
            max_horizon = st.slider("Horizon (weeks)", 1, 4, default_h)

        with c3:
            explorer_mode = st.radio(
                "View",
                options=["Forecast", "Feature snapshot"],
                horizontal=True,
            )

        _set_qp(selected_id, max_horizon)

    # meta row (safe)
    meta_df = show_catalog[show_catalog["id"] == selected_id]
    if meta_df.empty:
        st.warning("Selected show not found in catalog. Falling back to the top show.")
        selected_id = int(show_catalog.iloc[0]["id"])
        meta_df = show_catalog[show_catalog["id"] == selected_id]
    meta = meta_df.iloc[0]

    # ---------- HERO AREA ----------
    left, right = st.columns([1.2, 2.2], gap="large")

    with left:
        card = st.container(border=True)
        with card:
            st.markdown(f"## {meta.get('name', '')}")
            st.caption(meta.get("original_name", ""))

            info_row(
                [
                    ("Status", str(meta.get("status", "N/A"))),
                    ("Type", str(meta.get("type", "N/A"))),
                ]
            )
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

            info_row(
                [
                    ("Language", str(meta.get("language", "N/A"))),
                    ("Cert", str(meta.get("certification", "N/A"))),
                ]
            )

            seasons = meta.get("number_of_seasons", np.nan)
            episodes = meta.get("number_of_episodes", np.nan)
            pop_val = meta.get("popularity", np.nan)
            rating = meta.get("vote_average", np.nan)
            votes = meta.get("vote_count", 0)

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            info_row(
                [
                    ("Seasons", str(seasons)),
                    ("Episodes", str(episodes)),
                ]
            )

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.write(f"**Genres:** {meta.get('genres', 'N/A')}")
            st.write(f"**Networks:** {meta.get('networks', 'N/A')}")

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            cA, cB = st.columns(2)
            cA.metric("TMDB popularity", f"{safe_float(pop_val):.1f}" if pd.notna(pop_val) else "N/A")
            cB.metric("Rating", f"{safe_float(rating):.1f}" if pd.notna(rating) else "N/A")
            st.caption(f"Votes: {int(votes) if pd.notna(votes) else 0:,}")

            poster_url = None
            if "poster_url" in meta and pd.notna(meta.get("poster_url", np.nan)):
                poster_url = meta.get("poster_url")
            elif "poster_path_full" in meta and pd.notna(meta.get("poster_path_full", np.nan)):
                poster_url = meta.get("poster_path_full")

            if poster_url:
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.image(poster_url, use_container_width=True)

        with st.expander("What do these numbers mean?", expanded=False):
            st.markdown(
                """
**Interest (0–100)**  
Weekly Google Trends interest score for the show’s search term (normalized by Google).  
- **100** = peak popularity in the selected timeframe  
- **0** = very low / no measurable interest

**TMDB rating**  
Average user rating on TMDB (**0–10**). This measures *quality sentiment*, not attention.

**TMDB popularity**  
A TMDB internal popularity score (higher = more “in-demand” / viewed / engaged recently). 

**Forecast (t+1 … t+4)**  
Predicted **future Google Trends Interest** 1–4 weeks ahead based on latest signals + metadata.

**KW_<word>**  
Indicates whether a specific keyword appears in a show’s description or metadata.
"""
            )

    with right:
        hero = st.container(border=True)
        with hero:
            if explorer_mode == "Forecast":
                try:
                    with st.spinner("Generating forecast..."):
                        result = predict_show_interest(show_id=selected_id, max_horizon=max_horizon)
                except Exception as e:
                    st.error("Forecast failed for this selection.")
                    st.exception(e)
                    result = None

                if result is None:
                    st.stop()

                history_df = result["history_df"].sort_values("week_start")
                if history_df.empty or "interest" not in history_df.columns or history_df["interest"].dropna().empty:
                    st.warning("No historical interest data available for this show.")
                else:
                    last_week = history_df["week_start"].max()
                    last_interest = float(history_df.loc[history_df["week_start"] == last_week, "interest"].iloc[0])

                    forecast_rows = []
                    for key, value in result["predictions"].items():
                        h = int(key.split("+")[-1])
                        future_week = last_week + pd.Timedelta(weeks=h)
                        forecast_rows.append({"week_start": future_week, "forecast": float(value)})
                    forecast_df = pd.DataFrame(forecast_rows)

                    export_df = pd.merge(
                        history_df.rename(columns={"interest": "historical_interest"}),
                        forecast_df,
                        on="week_start",
                        how="outer",
                    ).sort_values("week_start")

                    top_strip = st.container()
                    with top_strip:
                        k1, k2, k3, k4 = st.columns([1, 1, 1, 1.3])
                        k1.metric("Latest observed", f"{last_interest:.1f}")
                        if "t+1" in result["predictions"]:
                            next_pred = float(result["predictions"]["t+1"])
                            delta = next_pred - last_interest
                            k2.metric("Forecast (t+1)", f"{next_pred:.1f}", f"{delta:+.1f}")
                        else:
                            k2.metric("Forecast (t+1)", "N/A")

                        tail = history_df.sort_values("week_start").tail(4)
                        if len(tail) >= 2:
                            change_4w = float(tail["interest"].iloc[-1] - tail["interest"].iloc[0])
                            k3.metric("Last 4w change", f"{change_4w:+.1f}")
                        else:
                            k3.metric("Last 4w change", "N/A")

                        k4.download_button(
                            "Download CSV",
                            data=export_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"streamwatch_show_{selected_id}_forecast.csv",
                            mime="text/csv",
                        )

                    st.markdown("#### Interest history & forecast")

                    hist_plot = history_df.copy()
                    hist_plot["series"] = "Historical"
                    hist_plot = hist_plot.rename(columns={"interest": "value"})

                    fc_plot = (
                        forecast_df.copy()
                        if not forecast_df.empty
                        else pd.DataFrame(columns=["week_start", "forecast"])
                    )
                    fc_plot["series"] = "Forecast"
                    fc_plot = fc_plot.rename(columns={"forecast": "value"})

                    plot_df = pd.concat(
                        [hist_plot[["week_start", "value", "series"]], fc_plot[["week_start", "value", "series"]]],
                        ignore_index=True,
                    )

                    show_full = st.toggle("Show full history", value=False)
                    if not show_full:
                        cutoff = last_week - pd.Timedelta(weeks=26)
                        plot_df = plot_df[plot_df["week_start"] >= cutoff]

                    base = alt.Chart(plot_df).encode(
                        x=alt.X("week_start:T", title="Week"),
                        y=alt.Y("value:Q", title="Interest (0–100)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[
                            alt.Tooltip("week_start:T", title="Week"),
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("value:Q", title="Interest", format=".1f"),
                        ],
                    )

                    line = base.mark_line().encode(
                        color=alt.Color(
                            "series:N",
                            scale=alt.Scale(domain=["Historical", "Forecast"], range=["#60a5fa", "#fbbf24"]),
                            legend=alt.Legend(title=None),
                        )
                    )
                    pts = base.mark_circle(size=50, opacity=0.9).encode(
                        color=alt.Color(
                            "series:N",
                            scale=alt.Scale(domain=["Historical", "Forecast"], range=["#60a5fa", "#fbbf24"]),
                            legend=None,
                        )
                    )
                    st.altair_chart((line + pts).properties(height=340), use_container_width=True)

                    # Explainability
                    with st.expander("Explainability: why the model thinks this will move", expanded=False):
                        model_set = MODELS["lightgbm"]
                        h_for_explain = 1 if 1 in model_set else max_horizon
                        model = model_set.get(h_for_explain)

                        feature_names = FEATURE_COLS["lightgbm"]
                        global_imp = _try_get_lgbm_feature_importance(model, feature_names, top_k=20)
                        local_contrib = _try_get_local_contributions(model, result["X_latest"], feature_names, top_k=12)

                        e1, e2 = st.columns(2, gap="large")
                        with e1:
                            st.caption(f"Global top features (t+{h_for_explain})")
                            if global_imp.empty:
                                st.info("Global feature importance not available for this model object.")
                            else:
                                chart = (
                                    alt.Chart(global_imp)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("importance:Q", title="Importance"),
                                        y=alt.Y("feature:N", sort="-x", title=None),
                                        tooltip=["feature:N", alt.Tooltip("importance:Q", format=".2f")],
                                    )
                                    .properties(height=360)
                                )
                                st.altair_chart(chart, use_container_width=True)

                        with e2:
                            st.caption(f"Local drivers (latest week; t+{h_for_explain})")
                            if local_contrib.empty:
                                st.info("Local contributions not available (requires pred_contrib support).")
                            else:
                                lc = local_contrib.copy()
                                lc["flag"] = np.where(lc["contribution"] >= 0, "up", "down")
                                chart = (
                                    alt.Chart(lc)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("contribution:Q", title="Contribution"),
                                        y=alt.Y("feature:N", sort="-x", title=None),
                                        color=alt.Color(
                                            "flag:N",
                                            scale=alt.Scale(domain=["up", "down"], range=["#16a34a", "#dc2626"]),
                                            legend=None,
                                        ),
                                        tooltip=[
                                            "feature:N",
                                            alt.Tooltip("contribution:Q", format=".3f"),
                                            alt.Tooltip("direction:N"),
                                        ],
                                    )
                                    .properties(height=360)
                                )
                                st.altair_chart(chart, use_container_width=True)

                        st.caption("Local contributions are additive model terms (when supported): useful for intuition, not causal.")

                    # Compare Mode
                    with st.expander("Compare shows (overlay history + t+1 forecast)", expanded=False):
                        compare_ids = st.multiselect(
                            "Pick 2–5 shows",
                            options=show_catalog["id"].tolist(),
                            default=[selected_id],
                            format_func=show_label_by_id,
                        )

                        compare_ids = compare_ids[:5]

                        if len(compare_ids) >= 2:
                            rows = []
                            for sid in compare_ids:
                                try:
                                    res = predict_show_interest(sid, max_horizon=1)
                                    hdf = res["history_df"].sort_values("week_start")
                                    if "interest" not in hdf.columns or hdf.empty or hdf["interest"].dropna().empty:
                                        continue
                                    lw = hdf["week_start"].max()
                                    show_name = str(show_catalog.loc[show_catalog["id"] == sid, "name"].iloc[0])

                                    rows.append(
                                        pd.DataFrame(
                                            {
                                                "week_start": hdf["week_start"],
                                                "value": hdf["interest"].astype(float),
                                                "series": "Historical",
                                                "show": show_name,
                                            }
                                        )
                                    )
                                    if "t+1" in res["predictions"]:
                                        rows.append(
                                            pd.DataFrame(
                                                {
                                                    "week_start": [lw + pd.Timedelta(weeks=1)],
                                                    "value": [float(res["predictions"]["t+1"])],
                                                    "series": ["Forecast t+1"],
                                                    "show": [show_name],
                                                }
                                            )
                                        )
                                except Exception:
                                    continue

                            if rows:
                                cdf = pd.concat(rows, ignore_index=True)
                                max_w = cdf["week_start"].max()
                                cdf = cdf[cdf["week_start"] >= (max_w - pd.Timedelta(weeks=26))]

                                chart = (
                                    alt.Chart(cdf)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("week_start:T", title="Week"),
                                        y=alt.Y("value:Q", title="Interest (0–100)", scale=alt.Scale(domain=[0, 100])),
                                        color=alt.Color("show:N", title="Show"),
                                        strokeDash=alt.StrokeDash("series:N", title=None),
                                        tooltip=[
                                            alt.Tooltip("show:N", title="Show"),
                                            alt.Tooltip("series:N", title="Series"),
                                            alt.Tooltip("week_start:T", title="Week"),
                                            alt.Tooltip("value:Q", title="Interest", format=".1f"),
                                        ],
                                    )
                                    .properties(height=360)
                                )
                                st.altair_chart(chart, use_container_width=True)
                            else:
                                st.info("Not enough data to compare selected shows.")
                        else:
                            st.caption("Select at least 2 shows to compare.")

            else:
                df_show = panel[panel["id"] == selected_id].copy()
                if "week_start" in df_show.columns:
                    df_show = df_show.sort_values("week_start")

                if df_show.empty:
                    st.warning("No data available for this show.")
                else:
                    latest = df_show.iloc[-1]
                    X = build_model_input_lgbm(latest).iloc[0]

                    st.markdown("#### Feature snapshot (latest week)")
                    st.caption("These are the inputs actually sent into the LightGBM model for the most recent week.")
                    feat_df = pd.DataFrame({"Feature": X.index, "Value": X.values})
                    feat_df["abs_value"] = feat_df["Value"].abs()
                    feat_df = feat_df.sort_values("abs_value", ascending=False).drop(columns=["abs_value"])
                    st.dataframe(feat_df.head(40), use_container_width=True, hide_index=True)

                    if "interest" in df_show.columns:
                        st.markdown("#### Interest distribution")
                        hist = (
                            df_show[["week_start", "interest"]].dropna().copy()
                            if "week_start" in df_show.columns
                            else pd.DataFrame()
                        )
                        if not hist.empty:
                            hist["interest"] = pd.to_numeric(hist["interest"], errors="coerce")
                            chart = (
                                alt.Chart(hist.dropna())
                                .mark_bar()
                                .encode(
                                    x=alt.X("interest:Q", bin=alt.Bin(maxbins=20), title="Interest"),
                                    y=alt.Y("count():Q", title="Weeks"),
                                    tooltip=[alt.Tooltip("count():Q", title="Weeks")],
                                )
                                .properties(height=220)
                            )
                            st.altair_chart(chart, use_container_width=True)

    # ---------- MARKET SCANS (MOVER LISTS) ----------
    section("Catalog movers", "Scan top shows for predicted next-week movement (t+1).")

    movers = st.container(border=True)
    with movers:
        trending_all = compute_trending_all()
        if trending_all.empty:
            st.info("No trending data available.")
        else:
            f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.1, 1.5], vertical_alignment="center")
            with f1:
                min_last_interest = st.slider("Min current interest", 0.0, 100.0, 5.0, 1.0)
            with f2:
                direction = st.selectbox("Direction", ["Gainers", "Losers", "Both"])
            with f3:
                sort_by = st.selectbox("Sort by", ["delta_t+1", "predicted_t+1", "last_interest", "popularity"])
            with f4:
                rows_to_show = st.selectbox("Rows", [15, 25, 40], index=1)

            df_scan = trending_all.copy()
            df_scan = df_scan[df_scan["last_interest"] >= min_last_interest]

            if direction == "Gainers":
                df_scan = df_scan.sort_values(sort_by, ascending=False).head(rows_to_show)
            elif direction == "Losers":
                df_scan = df_scan.sort_values(sort_by, ascending=True).head(rows_to_show)
            else:
                top_g = df_scan.sort_values("delta_t+1", ascending=False).head(rows_to_show // 2)
                top_l = df_scan.sort_values("delta_t+1", ascending=True).head(rows_to_show // 2)
                df_scan = pd.concat([top_g, top_l], ignore_index=True)

            dl1, dl2 = st.columns([1, 6], vertical_alignment="center")
            with dl1:
                st.download_button(
                    "Download CSV",
                    data=df_scan.to_csv(index=False).encode("utf-8"),
                    file_name="streamwatch_movers.csv",
                    mime="text/csv",
                )
            with dl2:
                st.caption("Tip: sort by delta to see rising/falling shows; sort by popularity to sanity-check.")

            # ---- UI display table (professional headers) ----
            display_cols = ["name", "last_interest", "predicted_t+1", "delta_t+1"]
            display_df = _display_cols(df_scan, display_cols, MOVER_LABELS)
            delta_display_col = MOVER_LABELS["delta_t+1"]

            st.dataframe(
                style_delta_table(
                    display_df,
                    delta_col=delta_display_col,
                    format_map=MOVER_FORMAT_MAP,
                ),
                use_container_width=True,
                hide_index=True,
            )

            # ---- Chart: use renamed copy so axis/tooltips read professionally ----
            chart_df = df_scan.copy().sort_values("delta_t+1", ascending=False)
            chart_df_disp = chart_df.rename(columns=MOVER_LABELS)

            bar = (
                alt.Chart(chart_df_disp)
                .mark_bar()
                .encode(
                    x=alt.X(f"{delta_display_col}:Q", title="Net interest change (t+1 − latest)"),
                    y=alt.Y(f"{MOVER_LABELS['name']}:N", sort="-x", title="Show"),
                    color=alt.condition(
                        alt.datum[delta_display_col] >= 0, alt.value("#16a34a"), alt.value("#dc2626")
                    ),
                    tooltip=[
                        alt.Tooltip(f"{MOVER_LABELS['name']}:N", title="Show"),
                        alt.Tooltip(f"{MOVER_LABELS['last_interest']}:Q", title="Latest", format=".1f"),
                        alt.Tooltip(f"{MOVER_LABELS['predicted_t+1']}:Q", title="Forecast t+1", format=".1f"),
                        alt.Tooltip(f"{delta_display_col}:Q", title="Net change", format="+.1f"),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(bar, use_container_width=True)

    # ---------- ALL SHOWS OVERVIEW ----------
    with st.expander("All shows overview (ranked by popularity)", expanded=False):
        all_preds = compute_predictions_all()
        if all_preds.empty:
            st.info("No data available.")
        else:
            page_size = 50
            total_pages = int(np.ceil(len(all_preds) / page_size))
            col_page, col_info = st.columns([1, 3], vertical_alignment="center")

            with col_page:
                page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)

            start = (page - 1) * page_size
            end = start + page_size
            page_df = all_preds.iloc[start:end].copy()

            with col_info:
                st.caption(f"Showing {start + 1}–{min(end, len(all_preds))} of {len(all_preds)} shows.")

            display_cols = ["rank_popularity", "name", "last_interest", "predicted_t+1", "delta_t+1"]
            page_disp = _display_cols(page_df, display_cols, OVERVIEW_LABELS)
            delta_display_col = OVERVIEW_LABELS["delta_t+1"]

            st.dataframe(
                style_delta_table(
                    page_disp,
                    delta_col=delta_display_col,
                    format_map=OVERVIEW_FORMAT_MAP,
                ),
                use_container_width=True,
                hide_index=True,
            )


    # ---------- KEYWORD INTELLIGENCE ----------
    section(
        "Keyword intelligence",
        "Keyword flags (KW_*) derived from show descriptions/metadata.",
    )

    kw_box = st.container(border=True)
    with kw_box:
        kws = keyword_insights(panel, prefix="kw_")
        lift_df = kws["lift"].copy()
        cov_df = kws["coverage"].copy()

        def _pretty_kw(s: str) -> str:
            # kw_dark_comedy -> Dark comedy
            s = str(s)
            s = s.replace("kw_", "").replace("KW_", "")
            s = s.replace("_", " ").strip()
            return s[:1].upper() + s[1:] if s else s

        if lift_df.empty and cov_df.empty:
            st.info("No keyword columns found (expected columns like kw_<word>).")
        else:
            # Clean labels
            if not lift_df.empty:
                lift_df["keyword_readable"] = lift_df["keyword"].map(_pretty_kw)
            if not cov_df.empty:
                cov_df["keyword_readable"] = cov_df["keyword"].map(_pretty_kw)

            # Top-line KPIs
            k1, k2, k3 = st.columns(3, vertical_alignment="center")
            if not lift_df.empty:
                top_kw = lift_df.iloc[0]
                k1.metric("Strongest keyword signal", str(top_kw["keyword_readable"]))
                k2.metric("Lift (present − absent)", f"{top_kw['lift_present_minus_absent']:+.2f}")
                k3.metric("Coverage", f"{top_kw['coverage_%']:.1f}%")
            else:
                k1.metric("Strongest keyword signal", "N/A")
                k2.metric("Lift (present − absent)", "N/A")
                k3.metric("Coverage", "N/A")

            st.caption(
                "Interpretation: **Lift** compares mean interest when the keyword is present vs absent (association, not causality). "
                "**Coverage** is the % of rows where the keyword flag is on."
            )

            left, right = st.columns(2, gap="large")

            # ---------------- LEFT: Lift chart + styled table ----------------
            with left:
                st.markdown("#### Keywords most associated with higher interest")

                topN = 10
                chart_df = (
                    lift_df[["keyword_readable", "lift_present_minus_absent", "coverage_%"]]
                    .dropna()
                    .head(topN)
                    .copy()
                )
                chart_df["direction"] = np.where(chart_df["lift_present_minus_absent"] >= 0, "Up", "Down")

                lift_chart = (
                    alt.Chart(chart_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("lift_present_minus_absent:Q", title="Interest lift when keyword is present"),
                        y=alt.Y("keyword_readable:N", sort="-x", title=None),
                        color=alt.Color(
                            "direction:N",
                            scale=alt.Scale(domain=["Up", "Down"], range=["#16a34a", "#dc2626"]),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("keyword_readable:N", title="Keyword"),
                            alt.Tooltip("lift_present_minus_absent:Q", title="Interest lift", format="+.2f"),
                            alt.Tooltip("coverage_%:Q", title="Coverage (%)", format=".1f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(lift_chart, use_container_width=True)

                # Styled table (professional headers + color lift column)
                tbl = lift_df[
                    ["keyword_readable", "coverage_%", "mean_interest_present", "mean_interest_absent", "lift_present_minus_absent"]
                ].head(12).copy()
                tbl = tbl.rename(columns=KW_LIFT_TABLE_LABELS)

                def _color_lift(v):
                    try:
                        v = float(v)
                    except Exception:
                        return ""
                    if v > 0:
                        return "color: #16a34a; font-weight: 700;"
                    if v < 0:
                        return "color: #dc2626; font-weight: 700;"
                    return "color: rgba(255,255,255,0.72);"

                st.dataframe(
                    tbl.style
                    .format(
                        {
                            "Coverage (%)": "{:.1f}",
                            "Mean interest (keyword present)": "{:.2f}",
                            "Mean interest (keyword absent)": "{:.2f}",
                            "Lift (present − absent)": "{:+.2f}",
                        },
                        na_rep="—",
                    )
                    .applymap(_color_lift, subset=["Lift (present − absent)"]),
                    use_container_width=True,
                    hide_index=True,
                )

            # ---------------- RIGHT: Coverage donut + “high coverage” table ----------------
            with right:
                st.markdown("#### Keyword footprint (coverage across catalog)")

                # Donut chart showing coverage share among top coverage keywords
                topK = 6
                pie_df = cov_df[["keyword_readable", "coverage_%", "mean_interest_present"]].dropna().head(topK).copy()
                if not pie_df.empty:
                    pie_df["coverage_share"] = pie_df["coverage_%"] / pie_df["coverage_%"].sum()

                    donut = (
                        alt.Chart(pie_df)
                        .mark_arc(innerRadius=60, outerRadius=110)
                        .encode(
                            theta=alt.Theta("coverage_share:Q", title=None),
                            color=alt.Color("keyword_readable:N", title="Keyword"),
                            tooltip=[
                                alt.Tooltip("keyword_readable:N", title="Keyword"),
                                alt.Tooltip("coverage_%:Q", title="Coverage (%)", format=".1f"),
                                alt.Tooltip("mean_interest_present:Q", title="Mean interest (present)", format=".2f"),
                            ],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(donut, use_container_width=True)
                    st.caption("Donut shows relative coverage share among the most common keyword flags.")

                # Coverage table (professional headers)
                cov_tbl = cov_df[["keyword_readable", "coverage_%", "mean_interest_present"]].head(12).copy()
                cov_tbl = cov_tbl.rename(columns=KW_COV_TABLE_LABELS)

                st.dataframe(
                    cov_tbl.style.format(
                        {"Coverage (%)": "{:.1f}", "Mean interest (present)": "{:.2f}"},
                        na_rep="—",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

# ======================================================================================
# TAB 2 — MODEL OPS
# ======================================================================================
with tab_models:
    st.subheader("Model Ops")
    st.caption(
        "A lightweight monitoring console for training runs, performance across horizons, and data health. "
        "Designed to be interview-friendly: reproducible, debuggable, and explainable."
    )

    def kpi_card(col, label: str, value: str, delta: str | None = None, help_text: str | None = None):
        col.metric(label, value, delta=delta, help=help_text)

    mdf = metrics_df.copy() if metrics_df is not None else pd.DataFrame()
    has_metrics = (not mdf.empty) and {"family", "horizon"}.issubset(mdf.columns)

    controls = st.container(border=True)
    with controls:
        c1, c2, c3 = st.columns([2, 3, 2], vertical_alignment="center")

        families = sorted(mdf["family"].dropna().unique().tolist()) if has_metrics else ["lgbm"]
        selected_family = c1.selectbox("Model family", options=families, index=0, disabled=not has_metrics)

        df_fam = mdf[mdf["family"] == selected_family].copy() if has_metrics else pd.DataFrame()

        run_choices = []
        if not df_fam.empty and {"created_utc", "run_id"}.issubset(df_fam.columns):
            runs = (
                df_fam[["created_utc", "run_id", "week_start"]]
                .drop_duplicates()
                .sort_values("created_utc", ascending=False)
            )
            for _, r in runs.iterrows():
                run_choices.append(f"{str(r['created_utc'])} • {r['run_id']} • week={r.get('week_start','')}")

        selected_run = c2.selectbox(
            "Training run",
            options=run_choices if run_choices else ["(none yet)"],
            index=0,
            disabled=not bool(run_choices),
        )

        show_details = c3.toggle("Show run table", value=False)

    df_run = df_fam.copy()
    if selected_run and "•" in selected_run and "run_id" in df_run.columns:
        run_id = selected_run.split("•")[1].strip()
        df_run = df_run[df_run["run_id"].astype(str) == run_id]

    kpis = st.container(border=True)
    with kpis:
        c1, c2, c3, c4 = st.columns(4)

        if has_metrics and not df_run.empty:

            def _safe_mean(s):
                s = pd.to_numeric(s, errors="coerce").dropna()
                return float(s.mean()) if len(s) else np.nan

            def _safe_min(s):
                s = pd.to_numeric(s, errors="coerce").dropna()
                return float(s.min()) if len(s) else np.nan

            horizons_covered = (
                sorted(df_run["horizon"].dropna().unique().astype(int).tolist())
                if "horizon" in df_run.columns
                else []
            )
            rmse_best = _safe_min(df_run.get("rmse_val", pd.Series(dtype=float)))
            rmse_avg = _safe_mean(df_run.get("rmse_val", pd.Series(dtype=float)))
            r2_avg = _safe_mean(df_run.get("r2_val", pd.Series(dtype=float)))

            kpi_card(c1, "Horizons covered", str(horizons_covered) if horizons_covered else "N/A")
            kpi_card(c2, "Best val RMSE", f"{rmse_best:.3f}" if np.isfinite(rmse_best) else "N/A")
            kpi_card(c3, "Avg val RMSE", f"{rmse_avg:.3f}" if np.isfinite(rmse_avg) else "N/A")
            kpi_card(c4, "Avg val R²", f"{r2_avg:.3f}" if np.isfinite(r2_avg) else "N/A")
        else:
            kpi_card(c1, "Horizons covered", "1–4")
            kpi_card(c2, "Best val RMSE", "N/A")
            kpi_card(c3, "Avg val RMSE", "N/A")
            kpi_card(c4, "Avg val R²", "N/A")

    section("Performance by horizon", "Validation metrics across t+1…t+4 for the selected run.")

    if has_metrics and not df_run.empty:
        df_plot = df_run.dropna(subset=["horizon"]).copy()
        df_plot["horizon"] = df_plot["horizon"].astype(int)

        agg = (
            df_plot.groupby("horizon", as_index=False)
            .agg(
                rmse_val=("rmse_val", "mean"),
                r2_val=("r2_val", "mean"),
                rmse_train=("rmse_train", "mean"),
                r2_train=("r2_train", "mean"),
                n_train=("n_train", "max"),
                n_val=("n_val", "max"),
            )
            .sort_values("horizon")
        )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### Validation RMSE (lower is better)")
            st.line_chart(agg.set_index("horizon")["rmse_val"])
        with c2:
            st.markdown("#### Validation R² (higher is better)")
            st.line_chart(agg.set_index("horizon")["r2_val"])

        st.caption("Typically RMSE rises with horizon. If one horizon spikes, it can indicate weaker signal or staleness.")
    else:
        st.info("No metrics_history data loaded yet. Run the pipeline once to populate metrics.")

    if show_details:
        section("Run details", "Raw metrics for debugging & reproducibility.")
        detail = st.container(border=True)
        with detail:
            if has_metrics and not df_run.empty:
                st.dataframe(df_run, use_container_width=True)
                st.download_button(
                    "Download run metrics (CSV)",
                    data=df_run.to_csv(index=False).encode("utf-8"),
                    file_name=f"streamwatch_metrics_{selected_family}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No run details available.")

    section("Data health", "Freshness, coverage, and missingness checks (engineering maturity signal).")
    health = st.container(border=True)
    with health:
        c1, c2, c3, c4 = st.columns(4)

        min_week = _ensure_dt(panel["week_start"]).min() if "week_start" in panel.columns else pd.NaT
        max_week = _ensure_dt(panel["week_start"]).max() if "week_start" in panel.columns else pd.NaT
        n_rows = int(len(panel))
        n_shows = int(panel["id"].nunique()) if "id" in panel.columns else 0
        interest_missing = float(panel["interest"].isna().mean() * 100.0) if "interest" in panel.columns else np.nan

        kpi_card(c1, "Rows", f"{n_rows:,}")
        kpi_card(c2, "Shows", f"{n_shows:,}")
        kpi_card(c3, "Week range", f"{str(min_week)[:10]} → {str(max_week)[:10]}")
        kpi_card(c4, "Missing interest", f"{interest_missing:.1f}%" if np.isfinite(interest_missing) else "N/A")

        st.markdown("#### Missingness (top columns)")
        miss = panel.isna().mean().sort_values(ascending=False).head(12)
        miss_df = miss.reset_index()
        miss_df.columns = ["column", "missing_rate"]
        miss_df["missing_rate"] = (miss_df["missing_rate"] * 100.0).round(2)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    section("Reproducibility", "Artifact-driven configuration powering this UI.")
    repro = st.container(border=True)
    with repro:
        st.code(
            "\n".join(
                [
                    f"artifact_mode: {artifact_info.get('mode', 'unknown')}",
                    f"cache_root: {artifact_info.get('cache_root', '')}",
                    f"panel_clean_local: {str(DATA_PATH)}",
                    f"feature_columns_local: {str(FEATURE_COLS_PATH)}",
                    f"metrics_history_local: {str(METRICS_HISTORY_PATH) if METRICS_HISTORY_PATH else '(missing)'}",
                    f"lgbm_dir_local: {str(LGBM_DIR)}",
                    f"hgbr_dir_local: {str(HGBR_DIR)}",
                    "",
                    "serving_manifest:",
                    json.dumps(artifact_info.get("manifest", {}), indent=2),
                ]
            ),
            language="yaml",
        )

# ======================================================================================
# TAB 3 — ABOUT
# ======================================================================================
with tab_about:
    st.subheader("About StreamWatch")

    st.markdown(
        """
**StreamWatch** is a portfolio ETL + ML serving project that forecasts weekly **Google Trends interest (0–100)** for TV shows.
It’s designed to demonstrate **data engineering fundamentals**: reliable ingestion, deterministic transformations, artifact-driven deployments, and lightweight monitoring.
"""
    )

    st.markdown("### System overview")
    st.markdown(
        """
**Pipeline flow (high level):**
1. **Ingest**: pull show metadata + time series signals (e.g., trends, TMDB).
2. **Transform**: clean + join into a weekly panel and generate features (including keyword flags).
3. **Train**: fit multi-horizon models (**t+1 → t+4**) and evaluate on held-out validation.
4. **Publish**: write versioned artifacts to **GCS** (data + models + metrics) and update a **serving manifest**.
5. **Serve**: Streamlit reads the manifest, downloads the required artifacts, and renders forecasts + monitoring views.
"""
    )

    st.markdown("### What this app demonstrates")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
**Data Engineering**
- Artifact-driven serving (manifest as source of truth)
- Cloud storage integration (GCS)
- Caching + reproducible file paths
"""
        )
    with c2:
        st.markdown(
            """
**Analytics & ML**
- Multi-horizon forecasting (t+1…t+4)
- Explainability (global + local)
- Catalog scans for movers
"""
        )
    with c3:
        st.markdown(
            """
**Operational maturity**
- Metrics history (training runs)
- Data health checks (coverage/missingness)
- Debuggable UI for traceability
"""
        )

    st.markdown("### Where to look")
    st.markdown(
        """
- **GitHub repo** (code + formal writeup): https://github.com/respin5757/streamwatch  
- **Portfolio**: https://respin5757.github.io/personal-website/index.html  
- **LinkedIn**: https://www.linkedin.com/in/rafael-espinoza-2001/
"""
    )

    st.caption(
        "Note: Forecasts are directional signals based on historical patterns + metadata. "
        "This project prioritizes engineering correctness, reproducibility, and observability over hype."
    )
