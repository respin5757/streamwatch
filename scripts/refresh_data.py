# scripts/refresh_data.py
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from streamwatch.pipelines.extract_tmdb import run as extract_tmdb
from streamwatch.pipelines.extract_trends import run as extract_trends
from streamwatch.pipelines.build_panel_features import run as build_panel
from streamwatch.pipelines.build_feature_panel import run as build_features
from streamwatch.pipelines.train_models import run as train_models
from streamwatch.pipelines.publish_artifacts import run as publish
from streamwatch.pipelines.validate_serving_contract import run as validate
from streamwatch.pipelines.config import StreamWatchConfig


# ----------------------------
# Pretty helpers
# ----------------------------
def _now() -> float:
    return time.perf_counter()


def _fmt_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:0.1f}s"
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}m {s:0.0f}s"


def _hr() -> None:
    print("-" * 72)


def _header(title: str) -> None:
    _hr()
    print(f"▶ {title}")
    _hr()


def _kv(d: Dict[str, Any], keys: list[str]) -> str:
    parts = []
    for k in keys:
        if k in d:
            parts.append(f"{k}={d[k]}")
    return ", ".join(parts)


@dataclass
class StepResult:
    name: str
    seconds: float
    ok: bool
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _run_step(name: str, fn: Callable[[], Dict[str, Any]]) -> StepResult:
    _header(name)
    t0 = _now()
    try:
        out = fn()
        dt = _now() - t0
        print(f"✅ {name} completed in {_fmt_s(dt)}")
        return StepResult(name=name, seconds=dt, ok=True, meta=out)
    except KeyboardInterrupt:
        dt = _now() - t0
        print(f"\n🟡 {name} interrupted after {_fmt_s(dt)} (KeyboardInterrupt).")
        # Re-raise so you can stop cleanly
        raise
    except Exception as e:
        dt = _now() - t0
        print(f"❌ {name} failed after {_fmt_s(dt)}")
        print(f"   Error: {e}")
        return StepResult(name=name, seconds=dt, ok=False, error=str(e))


def _print_step_summary(step: StepResult) -> None:
    if not step.ok or not step.meta:
        return
    meta = step.meta

    if step.name.startswith("Extract TMDB"):
        print("   " + _kv(meta, ["tmdb_catalog_local_path", "n_shows", "n_rows", "n_cols"]))

    if step.name.startswith("Extract Google Trends"):
        # extract_trends returns these keys in the latest version:
        # attempted, used_cache, fetched, failed, rows, stopped_reason, trends_local_path
        print("   " + _kv(meta, ["trends_local_path", "attempted", "used_cache", "fetched", "failed", "rows", "stopped_reason"]))
        # show a few errors if present
        if "errors_sample" in meta and meta["errors_sample"]:
            print("   sample_errors:")
            for line in meta["errors_sample"][:5]:
                print(f"     - {line}")

    if step.name.startswith("Build panel.parquet"):
        print("   " + _kv(meta, ["panel_local_path", "n_rows", "n_cols"]))

    if step.name.startswith("Build panel_clean.parquet"):
        print("   " + _kv(meta, ["panel_clean_local_path", "n_rows", "n_cols"]))

    if step.name.startswith("Train models"):
        # your train_models returns metrics_records + feature_columns_local_path
        print("   " + _kv(meta, ["feature_columns_local_path"]))
        if "metrics_records" in meta and meta["metrics_records"]:
            # summarize by family
            fams = {}
            for r in meta["metrics_records"]:
                fams.setdefault(r.get("family"), 0)
                fams[r.get("family")] += 1
            print(f"   metrics_records: {sum(fams.values())} ({', '.join([f'{k}:{v}' for k,v in fams.items()])})")

    if step.name.startswith("Publish serving artifacts"):
        print("   " + _kv(meta, ["panel_clean_remote_path", "feature_columns_remote_path", "serving_manifest_remote_path"]))

    if step.name.startswith("Validate serving contract"):
        print("   " + str(meta))


# ----------------------------
# Main pipeline runner
# ----------------------------
def main(run_date: str, week_start: str, run_id: str, retrain: bool = True) -> None:
    cfg = StreamWatchConfig.from_env()

    work = cfg.work_dir / f"run_id={run_id}" / f"week_start={week_start}"
    work.mkdir(parents=True, exist_ok=True)

    print("\nStreamWatch Refresh")
    _hr()
    print(f"run_date   : {run_date}")
    print(f"week_start : {week_start}")
    print(f"run_id     : {run_id}")
    print(f"work_dir   : {work}")
    _hr()

    t_all = _now()
    results: list[StepResult] = []

    # Step 1: TMDB
    r1 = _run_step(
        "Extract TMDB catalog",
        lambda: extract_tmdb(run_date, run_id, week_start, out_dir=str(work / "tmdb")),
    )
    results.append(r1)
    _print_step_summary(r1)
    if not r1.ok:
        raise SystemExit(1)

    # Step 2: Trends
    r2 = _run_step(
        "Extract Google Trends",
        lambda: extract_trends(
            run_date,
            run_id,
            week_start,
            tmdb_catalog_local_path=r1.meta["tmdb_catalog_local_path"],
            out_trends_local_path=str(work / "trends.parquet"),
            # keep a stable cache path within the run workspace so reruns resume
            cache_dir=str(work / "trends_cache"),
        ),
    )
    results.append(r2)
    _print_step_summary(r2)
    if not r2.ok:
        raise SystemExit(1)

    # Step 3: Panel
    r3 = _run_step(
        "Build panel.parquet",
        lambda: build_panel(
            run_date,
            run_id,
            week_start,
            tmdb_catalog_local_path=r1.meta["tmdb_catalog_local_path"],
            trends_local_path=r2.meta["trends_local_path"],
            out_panel_local_path=str(work / "panel.parquet"),
        ),
    )
    results.append(r3)
    _print_step_summary(r3)
    if not r3.ok:
        raise SystemExit(1)

    # Step 4: Features
    r4 = _run_step(
        "Build panel_clean.parquet",
        lambda: build_features(
            run_date,
            run_id,
            week_start,
            in_panel_local_path=r3.meta["panel_local_path"],
            out_panel_clean_local_path=str(work / "panel_clean.parquet"),
            tfidf_vocab_local_path=str(work / "tfidf_vocab.json"),
        ),
    )
    results.append(r4)
    _print_step_summary(r4)
    if not r4.ok:
        raise SystemExit(1)

    # Step 5/6: Train + Publish + Validate
    if retrain:
        r5 = _run_step(
            "Train models (HGBR + LGBM) + append metrics history",
            lambda: train_models(
                run_date,
                run_id,
                week_start,
                panel_clean_local_path=r4.meta["panel_clean_local_path"],
                tfidf_vocab_local_path=r4.meta.get("tfidf_vocab_local_path"),
            ),
        )
        results.append(r5)
        _print_step_summary(r5)
        if not r5.ok:
            raise SystemExit(1)

        r6 = _run_step(
            "Publish serving artifacts to GCS",
            lambda: publish(
                run_date,
                run_id,
                week_start,
                panel_clean_local_path=r4.meta["panel_clean_local_path"],
                feature_columns_local_path=r5.meta["feature_columns_local_path"],
                hgbr_remote_dir=r5.meta["hgbr_remote_dir"],
                lgbm_remote_dir=r5.meta["lgbm_remote_dir"],

            ),
        )
        results.append(r6)
        _print_step_summary(r6)
        if not r6.ok:
            raise SystemExit(1)

        r7 = _run_step(
            "Validate serving contract",
            lambda: validate(
                panel_clean_local_path=r4.meta["panel_clean_local_path"],
                feature_columns_local_path=r5.meta["feature_columns_local_path"],
                model_dir_local_path=r5.meta["hgbr_local_dir"],
            ),
        )
        results.append(r7)
        _print_step_summary(r7)
        if not r7.ok:
            raise SystemExit(1)

    else:
        print("\nℹ️  --no-retrain supplied: skipping model training, publish, validate.")
        print(f"   panel_clean ready at: {r4.meta['panel_clean_local_path']}")

    total = _now() - t_all

    print("\nRun summary")
    _hr()
    for s in results:
        status = "OK " if s.ok else "ERR"
        print(f"{status}  {s.name:40s}  {_fmt_s(s.seconds)}")
    _hr()
    print(f"✅ Total: {_fmt_s(total)}")
    print(f"work_dir: {work}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-date", required=True, help="YYYY-MM-DD (Airflow ds)")
    p.add_argument("--week-start", required=True, help="YYYY-MM-DD (week partition)")
    p.add_argument("--run-id", required=True, help="Unique run id for versioning")
    p.add_argument("--no-retrain", action="store_true")
    args = p.parse_args()

    main(args.run_date, args.week_start, args.run_id, retrain=not args.no_retrain)
