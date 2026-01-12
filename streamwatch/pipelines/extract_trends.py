# streamwatch/pipelines/extract_trends.py
from __future__ import annotations

import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pytrends.request import TrendReq

# Silence noisy upstream warnings (log hygiene)
warnings.filterwarnings("ignore", category=FutureWarning, module="pytrends")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _make_pytrends() -> TrendReq:
    return TrendReq(hl="en-US", tz=360)


def _make_terms(name: str | None, original_name: str | None, networks: str | None) -> List[str]:
    base = (name or original_name or "").strip()
    if not base:
        return []
    nets = [n.strip() for n in (networks or "").split(",") if n.strip()]
    terms = [f"{base} tv series"]
    if nets:
        terms.append(f"{base} {nets[0]}")
        terms.append(f"{base} {nets[0]} tv series")

    # stable dedupe
    seen = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _is_429(err: Exception) -> bool:
    msg = str(err).lower()
    return ("code 429" in msg) or (" 429" in msg) or ("too many requests" in msg)


def _sleep(seconds: float) -> None:
    # jitter so you don't look like a bot
    time.sleep(max(0.0, seconds) + random.uniform(0, 0.75))


def _fetch_one_term(pytrends: TrendReq, term: str, *, timeframe: str, geo: str) -> Optional[pd.DataFrame]:
    pytrends.build_payload([term], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return None
    df = df.reset_index()
    # interest column is the single payload term
    interest_col = [c for c in df.columns if c not in ("date", "isPartial")][0]
    df = df.rename(columns={"date": "week_start", interest_col: "interest"})
    return df[["week_start", "interest"]]


def run(
    run_date: str,
    run_id: str,
    week_start: str,
    *,
    tmdb_catalog_local_path: str,
    out_trends_local_path: str,
    # optional
    cache_dir: str | None = None,
    timeframe: str = "today 5-y",
    geo: str = "",
    fail_on_too_many_failures: bool = False,
) -> Dict[str, Any]:
  
    out_path = Path(out_trends_local_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # budgets / throttles
    max_shows = _env_int("STREAMWATCH_TRENDS_MAX_SHOWS", 150)
    max_seconds = _env_int("STREAMWATCH_TRENDS_MAX_SECONDS", 900)  # 15 minutes default
    base_sleep = _env_float("STREAMWATCH_TRENDS_BASE_SLEEP", 2.0)

    max_retries = _env_int("STREAMWATCH_TRENDS_MAX_RETRIES", 4)
    backoff_base = _env_float("STREAMWATCH_TRENDS_BACKOFF_BASE", 15.0)
    backoff_cap = _env_float("STREAMWATCH_TRENDS_BACKOFF_CAP", 300.0)

    cooldown_429 = _env_float("STREAMWATCH_TRENDS_429_COOLDOWN", 120.0)
    reset_429_at = _env_int("STREAMWATCH_TRENDS_429_RESET_AT", 3)

    max_fails = _env_int("STREAMWATCH_TRENDS_MAX_FAILS", 200)

    # cache
    cache_root = Path(cache_dir) if cache_dir else out_path.parent / "trends_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    cat = pd.read_parquet(tmdb_catalog_local_path)[["id", "name", "original_name", "networks"]].copy()
    cat = cat.dropna(subset=["id"])
    cat["id"] = cat["id"].astype(int)

    start_ts = time.time()
    pytrends = _make_pytrends()

    rows: List[pd.DataFrame] = []
    errors: List[str] = []

    attempted = 0
    used_cache = 0
    fetched = 0
    failed = 0

    streak_429 = 0

    for idx, r in enumerate(cat.itertuples(index=False), start=1):
        if attempted >= max_shows:
            break
        if (time.time() - start_ts) > max_seconds:
            errors.append("time_budget_exceeded")
            break
        if failed >= max_fails:
            errors.append("too_many_failures")
            break

        show_id = int(r.id)
        attempted += 1

        cache_fp = cache_root / f"{show_id}.csv"

        # reuse cache if present
        if cache_fp.exists():
            try:
                tdf = pd.read_csv(cache_fp, parse_dates=["week_start"])
                if not tdf.empty:
                    tdf["id"] = show_id
                    rows.append(tdf[["id", "week_start", "interest"]])
                    used_cache += 1
                    _sleep(base_sleep * 0.15)
                    continue
            except Exception:
                # fall through to fetch
                pass

        terms = _make_terms(r.name, r.original_name, r.networks)
        if not terms:
            failed += 1
            errors.append(f"[{idx}] no_terms show_id={show_id}")
            _sleep(base_sleep)
            continue

        got: Optional[pd.DataFrame] = None
        last_err: Optional[str] = None

        for term in terms:
            for attempt in range(1, max_retries + 1):
                # budget check inside retries too
                if (time.time() - start_ts) > max_seconds:
                    errors.append("time_budget_exceeded")
                    break
                try:
                    got = _fetch_one_term(pytrends, term, timeframe=timeframe, geo=geo)
                    if got is None or got.empty:
                        last_err = f"empty term='{term}'"
                        break  # try next term
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)

                    if _is_429(e):
                        streak_429 += 1
                        # escalating cooldown
                        backoff = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
                        extra = min(backoff_cap, cooldown_429 * min(3, streak_429))
                        sleep_s = min(backoff_cap, backoff + extra)

                        errors.append(
                            f"[{idx}] 429 show_id={show_id} attempt={attempt} streak={streak_429} sleep={int(sleep_s)}"
                        )
                        _sleep(sleep_s)

                        if streak_429 >= reset_429_at:
                            pytrends = _make_pytrends()
                            streak_429 = 0
                        continue

                    # non-429 error
                    sleep_s = min(30.0, 2 ** (attempt - 1))
                    errors.append(f"[{idx}] err show_id={show_id} attempt={attempt} sleep={sleep_s}: {last_err}")
                    _sleep(sleep_s)

            if got is not None and not got.empty and last_err is None:
                break
            if "time_budget_exceeded" in errors[-1:]:
                break

        if got is None or got.empty:
            failed += 1
            errors.append(f"[{idx}] failed show_id={show_id} last_err={last_err}")
            _sleep(base_sleep)
            continue

        # success
        streak_429 = 0
        got = got.copy()
        got["id"] = show_id
        got = got[["id", "week_start", "interest"]]
        rows.append(got)
        fetched += 1

        # write per-show cache
        got[["week_start", "interest"]].to_csv(cache_fp, index=False)

        _sleep(base_sleep)

    if not rows:
        raise RuntimeError(
            f"No trends produced. attempted={attempted} failed={failed}. sample_errors={errors[:10]}"
        )

    trends = pd.concat(rows, ignore_index=True)
    trends["week_start"] = pd.to_datetime(trends["week_start"], errors="coerce")
    trends["id"] = pd.to_numeric(trends["id"], errors="coerce")
    trends["interest"] = pd.to_numeric(trends["interest"], errors="coerce")
    trends = trends.dropna(subset=["week_start", "id"])

    trends.to_parquet(out_path, index=False)

    result = {
        "run_date": run_date,
        "run_id": run_id,
        "week_start": week_start,
        "trends_local_path": str(out_path),
        "cache_dir": str(cache_root),
        "attempted": int(attempted),
        "used_cache": int(used_cache),
        "fetched": int(fetched),
        "failed": int(failed),
        "rows": int(trends.shape[0]),
        "errors_sample": errors[:25],
        "stopped_reason": errors[-1] if errors and errors[-1] in ("time_budget_exceeded", "too_many_failures") else None,
    }

    if fail_on_too_many_failures and failed > 0 and fetched == 0:
        raise RuntimeError(f"Trends fetch produced zero fetched shows. errors_sample={errors[:10]}")

    return result
