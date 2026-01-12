# dags/streamwatch_weekly.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule

from streamwatch.pipelines.config import StreamWatchConfig
from streamwatch.pipelines.dates import week_start_from_ds

from streamwatch.pipelines.extract_tmdb import run as extract_tmdb
from streamwatch.pipelines.extract_trends import run as extract_trends
from streamwatch.pipelines.build_panel_features import run as build_panel
from streamwatch.pipelines.build_feature_panel import run as build_features
from streamwatch.pipelines.train_models import run as train_models
from streamwatch.pipelines.publish_artifacts import run as publish
from streamwatch.pipelines.validate_serving_contract import run as validate


DAG_ID = "streamwatch_weekly"


def _default_args() -> dict:
    return {
        "owner": "streamwatch",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
        # avoids two weekly runs overlapping and stepping on "overwrite" artifacts
        "max_active_runs": 1,
    }


with DAG(
    dag_id=DAG_ID,
    description="Weekly StreamWatch pipeline (TMDB + Trends -> panel -> features -> train -> publish -> validate)",
    default_args=_default_args(),
    start_date=datetime(2025, 1, 1),
    schedule="0 8 * * MON",  # Mondays 8:00am
    catchup=True,  # allows backfills by ds
    tags=["streamwatch"],
) as dag:

    @task
    def make_run_context(ds: str, run_id: str | None = None) -> Dict[str, str]:
        """
        ds: Airflow logical date string YYYY-MM-DD
        run_id: optional override (useful for ad-hoc). If not provided, derives from ds.
        """
        week_start = week_start_from_ds(ds)
        rid = run_id or f"airflow_{ds.replace('-', '')}"
        return {"run_date": ds, "week_start": week_start, "run_id": rid}

    @task
    def prepare_work_dir(ctx: Dict[str, str]) -> Dict[str, str]:
        cfg = StreamWatchConfig.from_env()
        work = cfg.work_dir / f"run_id={ctx['run_id']}" / f"week_start={ctx['week_start']}"
        work.mkdir(parents=True, exist_ok=True)
        return {**ctx, "work_dir": str(work)}

    @task(retries=2, retry_delay=timedelta(minutes=5))
    def task_extract_tmdb(ctx: Dict[str, str]) -> Dict[str, Any]:
        out = extract_tmdb(ctx["run_date"], ctx["run_id"], ctx["week_start"], out_dir=str(Path(ctx["work_dir"]) / "tmdb"))
        return out

    @task(retries=4, retry_delay=timedelta(minutes=15))
    def task_extract_trends(ctx: Dict[str, str], tmdb_out: Dict[str, Any]) -> Dict[str, Any]:
        work = Path(ctx["work_dir"])
        out = extract_trends(
            ctx["run_date"],
            ctx["run_id"],
            ctx["week_start"],
            tmdb_catalog_local_path=tmdb_out["tmdb_catalog_local_path"],
            out_trends_local_path=str(work / "trends.parquet"),
            cache_dir=str(work / "trends_cache"),
        )
        return out

    @task
    def task_build_panel(ctx: Dict[str, str], tmdb_out: Dict[str, Any], trends_out: Dict[str, Any]) -> Dict[str, Any]:
        work = Path(ctx["work_dir"])
        return build_panel(
            ctx["run_date"],
            ctx["run_id"],
            ctx["week_start"],
            tmdb_catalog_local_path=tmdb_out["tmdb_catalog_local_path"],
            trends_local_path=trends_out["trends_local_path"],
            out_panel_local_path=str(work / "panel.parquet"),
        )

    @task
    def task_build_features(ctx: Dict[str, str], panel_out: Dict[str, Any]) -> Dict[str, Any]:
        work = Path(ctx["work_dir"])
        return build_features(
            ctx["run_date"],
            ctx["run_id"],
            ctx["week_start"],
            in_panel_local_path=panel_out["panel_local_path"],
            out_panel_clean_local_path=str(work / "panel_clean.parquet"),
            tfidf_vocab_local_path=str(work / "tfidf_vocab.json"),
        )

    @task
    def task_train_models(ctx: Dict[str, str], panel_clean_out: Dict[str, Any]) -> Dict[str, Any]:
        return train_models(
            ctx["run_date"],
            ctx["run_id"],
            ctx["week_start"],
            panel_clean_local_path=panel_clean_out["panel_clean_local_path"],
            tfidf_vocab_local_path=panel_clean_out.get("tfidf_vocab_local_path"),
        )

    @task
    def task_publish(ctx: Dict[str, str], panel_clean_out: Dict[str, Any], model_out: Dict[str, Any]) -> Dict[str, Any]:
        return publish(
            ctx["run_date"],
            ctx["run_id"],
            ctx["week_start"],
            panel_clean_local_path=panel_clean_out["panel_clean_local_path"],
            feature_columns_local_path=model_out["feature_columns_local_path"],
            hgbr_remote_dir=model_out["hgbr_remote_dir"],
            lgbm_remote_dir=model_out["lgbm_remote_dir"],

        )

    @task
    def task_validate(ctx: Dict[str, str], panel_clean_out: Dict[str, Any], model_out: Dict[str, Any]) -> Dict[str, Any]:
        # Validate all HGBR models locally (fast, deterministic)
        return validate(
            panel_clean_local_path=panel_clean_out["panel_clean_local_path"],
            feature_columns_local_path=model_out["feature_columns_local_path"],
            model_dir_local_path=model_out["hgbr_local_dir"],
        )

    # Optional: end marker that always runs to help with logging/monitoring
    @task(trigger_rule=TriggerRule.ALL_DONE)
    def finalize(ctx: Dict[str, str], published: Dict[str, Any] | None = None) -> None:
        print("DONE:", ctx)
        if published:
            print("Published:", published)

    # Wiring
    ctx = make_run_context(ds="{{ ds }}")
    ctx2 = prepare_work_dir(ctx)

    tmdb = task_extract_tmdb(ctx2)
    trends = task_extract_trends(ctx2, tmdb)
    panel = task_build_panel(ctx2, tmdb, trends)
    feats = task_build_features(ctx2, panel)
    models = task_train_models(ctx2, feats)
    pub = task_publish(ctx2, feats, models)
    val = task_validate(ctx2, feats, models)
    finalize(ctx2, pub)
