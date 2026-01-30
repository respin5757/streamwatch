# streamwatch/pipelines/extract_trends.py
from __future__ import annotations

"""
StreamWatch — Google Trends extractor (pytrends)

Purpose (in plain English):
- For each show in your TMDB catalog, query Google Trends for weekly interest over time.
- Persist results to a parquet file for downstream modeling.
- Use a per-show cache so re-runs (and CI/GitHub Actions) don’t re-hit the API for shows we already fetched.
- Be *very* defensive about rate limits (429) and slow/bottlenecked responses:
  - adaptive throttling
  - longer cool-downs
  - session resets
  - “cooldown mode” after repeated 429s
  - always write partial output if we got anything (so CI doesn’t waste progress)

Why GitHub Actions was failing:
- pytrends / Google Trends is rate-limited and sometimes slow.
- A tight loop + short backoffs can get you stuck in repeated 429s until you hit your time budget.

This version keeps your public interface the same, but improves the retry / sleep strategy
and allows much wider gaps automatically when 429s happen.
"""

import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    # NOTE: This is a fresh session object. Resetting it can help after repeated 429s.
    return TrendReq(hl="en-US", tz=360)


def _make_terms(name: str | None, original_name: str | None, networks: str | None) -> List[str]:
    base = (name or original_name or "").strip()
    if not base:
        return []
    nets = [n.strip() for n in (networks or "").split(",") if n.strip()]

    # NOTE: We try a few “search-y” variations. We *don’t* want tons of terms,
    # because that multiplies API calls and increases rate-limit risk.
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


def _is_timeoutish(err: Exception) -> bool:
    # pytrends bubbles different request exceptions; keep this broad.
    msg = str(err).lower()
    return ("timeout" in msg) or ("timed out" in msg) or ("readtimeout" in msg) or ("connection aborted" in msg)


def _sleep(seconds: float, *, jitter: float = 0.75) -> None:
    # jitter so you don't look like a bot (and to desync CI retries)
    time.sleep(max(0.0, seconds) + random.uniform(0, jitter))


def _fetch_one_term(pytrends: TrendReq, term: str, *, timeframe: str, geo: str) -> Optional[pd.DataFrame]:
    """
    Fetch weekly interest-over-time for a single term.
    Returns a df with columns: week_start, interest
    """
    pytrends.build_payload([term], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return None

    df = df.reset_index()
    # interest column is the single payload term
    interest_col = [c for c in df.columns if c not in ("date", "isPartial")][0]
    df = df.rename(columns={"date": "week_start", interest_col: "interest"})
    return df[["week_start", "interest"]]


@dataclass
class _ThrottleState:
    """
    A tiny adaptive throttler:
    - If we get repeated 429s, we enter “cooldown mode” with bigger sleeps.
    - If things are healthy, we slowly relax back down.
    """
    cooldown_mode: bool = False
    consecutive_429: int = 0
    consecutive_ok: int = 0

    def on_429(self) -> None:
        self.consecutive_429 += 1
        self.consecutive_ok = 0
        # enter cooldown mode quickly; pytrends can get stuck in 429 loops otherwise
        if self.consecutive_429 >= 2:
            self.cooldown_mode = True

    def on_ok(self) -> None:
        self.consecutive_ok += 1
        self.consecutive_429 = 0
        # after a few clean hits, exit cooldown mode
        if self.consecutive_ok >= 6:
            self.cooldown_mode = False


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

    # =========================
    # Budgets / throttles
    # =========================
    max_shows = _env_int("STREAMWATCH_TRENDS_MAX_SHOWS", 150)

    # IMPORTANT: GitHub Actions can be strict on time.
    max_seconds = _env_int("STREAMWATCH_TRENDS_MAX_SECONDS", 900)  # 15 minutes default

    # Base sleep between successful shows
    base_sleep = _env_float("STREAMWATCH_TRENDS_BASE_SLEEP", 2.0)

    # Retry knobs
    max_retries = _env_int("STREAMWATCH_TRENDS_MAX_RETRIES", 4)

    # Backoff behavior (we’re going to allow *much wider* gaps here)
    backoff_base = _env_float("STREAMWATCH_TRENDS_BACKOFF_BASE", 20.0)  
    backoff_cap = _env_float("STREAMWATCH_TRENDS_BACKOFF_CAP", 900.0)  # allow up to 15 min sleeps

    # 429 handling
    cooldown_429 = _env_float("STREAMWATCH_TRENDS_429_COOLDOWN", 240.0) 
    reset_429_at = _env_int("STREAMWATCH_TRENDS_429_RESET_AT", 2)  # reset session sooner

    # If the API is being *really* annoying, we hard-enter cooldown mode:
    cooldown_mode_multiplier = _env_float("STREAMWATCH_TRENDS_COOLDOWN_MULT", 2.5)
    cooldown_mode_floor = _env_float("STREAMWATCH_TRENDS_COOLDOWN_FLOOR", 60.0)

    # Fail-safety
    max_fails = _env_int("STREAMWATCH_TRENDS_MAX_FAILS", 200)

    # When we’re close to the time budget, stop starting new shows.
    # This prevents “start work → time budget exceeded mid-retry” churn.
    stop_buffer_seconds = _env_int("STREAMWATCH_TRENDS_STOP_BUFFER_SECONDS", 45)

    # =========================
    # Cache
    # =========================
    cache_root = Path(cache_dir) if cache_dir else out_path.parent / "trends_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Read catalog
    cat = pd.read_parquet(tmdb_catalog_local_path)[["id", "name", "original_name", "networks"]].copy()
    cat = cat.dropna(subset=["id"])
    cat["id"] = cat["id"].astype(int)

    start_ts = time.time()
    pytrends = _make_pytrends()
    throttle = _ThrottleState()

    rows: List[pd.DataFrame] = []
    errors: List[str] = []

    attempted = 0
    used_cache = 0
    fetched = 0
    failed = 0

    def _time_left() -> float:
        return max_seconds - (time.time() - start_ts)

    def _near_time_budget() -> bool:
        return _time_left() <= float(stop_buffer_seconds)

    # A small helper so “wider gaps” are automatic once we’re in cooldown mode.
    def _adaptive_base_sleep() -> float:
        if throttle.cooldown_mode:
            # In cooldown mode: bigger “normal” sleep between requests
            return max(cooldown_mode_floor, base_sleep * cooldown_mode_multiplier)
        return base_sleep

    # ============================================
    # Main loop
    # ============================================
    for idx, r in enumerate(cat.itertuples(index=False), start=1):
        if attempted >= max_shows:
            break
        if _near_time_budget():
            errors.append("time_budget_exceeded")
            break
        if failed >= max_fails:
            errors.append("too_many_failures")
            break

        show_id = int(r.id)
        attempted += 1

        cache_fp = cache_root / f"{show_id}.csv"

        # ============================================
        # Cache hit
        # ============================================
        if cache_fp.exists():
            try:
                tdf = pd.read_csv(cache_fp, parse_dates=["week_start"])
                if not tdf.empty:
                    tdf["id"] = show_id
                    rows.append(tdf[["id", "week_start", "interest"]])
                    used_cache += 1
                    # tiny sleep on cache hit so we don’t hammer filesystem/IO in CI
                    _sleep(_adaptive_base_sleep() * 0.10, jitter=0.25)
                    continue
            except Exception:
                # fall through to fetch
                pass

        # ============================================
        # Build term candidates
        # ============================================
        terms = _make_terms(r.name, r.original_name, r.networks)
        if not terms:
            failed += 1
            errors.append(f"[{idx}] no_terms show_id={show_id}")
            _sleep(_adaptive_base_sleep())
            continue

        got: Optional[pd.DataFrame] = None
        last_err: Optional[str] = None

        # ============================================
        # Try each term variant
        # ============================================
        for term in terms:
            # If we’re near the end, don’t start new term attempts.
            if _near_time_budget():
                errors.append("time_budget_exceeded")
                break

            # ============================================
            # Retry loop for this term
            # ============================================
            for attempt in range(1, max_retries + 1):
                if _near_time_budget():
                    errors.append("time_budget_exceeded")
                    break

                try:
                    got = _fetch_one_term(pytrends, term, timeframe=timeframe, geo=geo)

                    if got is None or got.empty:
                        # Not an error; just a bad term. Try next term.
                        last_err = f"empty term='{term}'"
                        break

                    # Success
                    last_err = None
                    throttle.on_ok()
                    break

                except Exception as e:
                    last_err = str(e)

                    # =========================
                    # 429: Rate limit
                    # =========================
                    if _is_429(e):
                        throttle.on_429()

                        # Core idea:
                        # - exponential backoff based on attempt
                        # - additional penalty based on consecutive 429s
                        # - if in cooldown mode, widen further
                        exp = backoff_base * (2 ** (attempt - 1))
                        streak_penalty = cooldown_429 * min(5, throttle.consecutive_429)
                        sleep_s = min(backoff_cap, exp + streak_penalty)

                        if throttle.cooldown_mode:
                            # “Wider gaps” when the API is upset:
                            sleep_s = min(backoff_cap, max(sleep_s, cooldown_mode_floor) * cooldown_mode_multiplier)

                        errors.append(
                            f"[{idx}] 429 show_id={show_id} attempt={attempt} "
                            f"consec_429={throttle.consecutive_429} cooldown={throttle.cooldown_mode} "
                            f"sleep={int(sleep_s)} term='{term}'"
                        )
                        _sleep(sleep_s)

                        # Reset session sooner after repeated 429s
                        if throttle.consecutive_429 >= reset_429_at:
                            pytrends = _make_pytrends()
                        continue

                    # =========================
                    # Timeout-ish: treat as “slow API day”
                    # =========================
                    if _is_timeoutish(e):
                        # We back off more aggressively than generic errors.
                        exp = backoff_base * (2 ** (attempt - 1))
                        sleep_s = min(backoff_cap, exp)

                        if throttle.cooldown_mode:
                            sleep_s = min(backoff_cap, max(sleep_s, cooldown_mode_floor))

                        errors.append(
                            f"[{idx}] timeout show_id={show_id} attempt={attempt} sleep={int(sleep_s)} term='{term}': {last_err}"
                        )
                        _sleep(sleep_s)
                        # If timeouts keep happening, resetting the session can help.
                        if attempt >= 2:
                            pytrends = _make_pytrends()
                        continue

                    # =========================
                    # Non-429 generic error
                    # =========================
                    sleep_s = min(60.0, 2 ** (attempt - 1))
                    if throttle.cooldown_mode:
                        sleep_s = max(sleep_s, cooldown_mode_floor * 0.5)

                    errors.append(f"[{idx}] err show_id={show_id} attempt={attempt} sleep={int(sleep_s)}: {last_err}")
                    _sleep(sleep_s)

            # If we got data, stop trying other terms.
            if got is not None and not got.empty and last_err is None:
                break

            # If we ran out of time, stop trying other terms too.
            if errors[-1:] == ["time_budget_exceeded"]:
                break

        # ============================================
        # If no data for show
        # ============================================
        if got is None or got.empty:
            failed += 1
            errors.append(f"[{idx}] failed show_id={show_id} last_err={last_err}")
            _sleep(_adaptive_base_sleep())
            continue

        # ============================================
        # Success path
        # ============================================
        got = got.copy()
        got["id"] = show_id
        got = got[["id", "week_start", "interest"]]
        rows.append(got)
        fetched += 1

        # write per-show cache (so next CI run doesn’t refetch)
        try:
            got[["week_start", "interest"]].to_csv(cache_fp, index=False)
        except Exception as e:
            errors.append(f"[{idx}] cache_write_failed show_id={show_id}: {e}")

        # Standard pacing between successful shows (adaptive)
        _sleep(_adaptive_base_sleep())

    # ============================================
    # Always persist what we have (partial output)
    # ============================================
    if not rows:
        # Nothing at all; this is a true failure.
        raise RuntimeError(
            f"No trends produced. attempted={attempted} failed={failed}. sample_errors={errors[:10]}"
        )

    trends = pd.concat(rows, ignore_index=True)
    trends["week_start"] = pd.to_datetime(trends["week_start"], errors="coerce")
    trends["id"] = pd.to_numeric(trends["id"], errors="coerce")
    trends["interest"] = pd.to_numeric(trends["interest"], errors="coerce")
    trends = trends.dropna(subset=["week_start", "id"])

    # Write parquet even if we stopped early due to time budget.
    # This makes GitHub Actions runs “incremental”: caches + partial parquet = progress, not waste.
    trends.to_parquet(out_path, index=False)

    stopped_reason = None
    if errors and errors[-1] in ("time_budget_exceeded", "too_many_failures"):
        stopped_reason = errors[-1]

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
        "stopped_reason": stopped_reason,
        "cooldown_mode": bool(throttle.cooldown_mode),
    }

    # Keep your old behavior: only hard-fail if you explicitly want it and we fetched nothing.
    if fail_on_too_many_failures and failed > 0 and fetched == 0:
        raise RuntimeError(f"Trends fetch produced zero fetched shows. errors_sample={errors[:10]}")

    return result
