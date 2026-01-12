# streamwatch/pipelines/extract_tmdb.py
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import requests
import pandas as pd
import numpy as np


BASE = "https://api.themoviedb.org/3"


def _headers() -> dict:
    token = os.getenv("TMDB_V4_TOKEN")
    if token:
        return {"accept": "application/json", "Authorization": f"Bearer {token}"}
    return {"accept": "application/json"}


def _fetch_top_page(page: int) -> dict:
    url = (
        f"{BASE}/discover/tv?"
        "include_adult=false&include_null_first_air_dates=false"
        "&language=en-US&sort_by=popularity.desc"
        f"&page={page}"
    )
    r = requests.get(url, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_details(tv_id: int) -> dict:
    url = f"{BASE}/tv/{tv_id}?append_to_response=credits,keywords,content_ratings,external_ids"
    r = requests.get(url, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def run(
    run_date: str,
    run_id: str,
    week_start: str,
    *,
    out_dir: str,
    max_shows: int = 500,
) -> Dict[str, Any]:
    """
    Local-first TMDB extract that writes:
      out_dir/tmdb_top_ids.csv
      out_dir/raw/tmdb/<id>.json
      out_dir/tmdb_catalog.parquet

    Returns small metadata.
    """
    out = Path(out_dir)
    raw_dir = out / "raw" / "tmdb"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) top ids
    out_csv = out / "tmdb_top_ids.csv"
    rows = []
    first = _fetch_top_page(1)
    total_pages = first.get("total_pages", 1)
    max_pages = min(total_pages, 25)

    for p in range(1, max_pages + 1):
        data = first if p == 1 else _fetch_top_page(p)
        for item in data.get("results", []):
            rows.append({
                "id": item["id"],
                "name": item.get("name"),
                "original_name": item.get("original_name"),
                "first_air_date": item.get("first_air_date"),
                "popularity": item.get("popularity"),
                "vote_average": item.get("vote_average"),
                "vote_count": item.get("vote_count"),
                "origin_country": ",".join(item.get("origin_country", [])),
                "language": item.get("original_language"),
            })
        time.sleep(0.25)

    seen, deduped = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
        if len(deduped) >= max_shows:
            break

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deduped[0].keys()))
        w.writeheader()
        w.writerows(deduped)

    # 2) details jsons
    ids = [int(r["id"]) for r in deduped]
    n_details = 0
    for i, tv_id in enumerate(ids, start=1):
        fp = raw_dir / f"{tv_id}.json"
        if fp.exists():
            continue
        try:
            data = _fetch_details(tv_id)
            fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            n_details += 1
            time.sleep(0.3)
        except requests.HTTPError:
            continue

    # 3) build catalog parquet
    catalog_rows = []
    for fp in raw_dir.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            genres = ",".join([g["name"] for g in data.get("genres", [])])
            networks = ",".join([n["name"] for n in data.get("networks", [])])
            credits = data.get("credits", {})
            keywords = data.get("keywords", {}).get("results", [])
            content_ratings = data.get("content_ratings", {}).get("results", [])
            external_ids = data.get("external_ids", {})

            cert = None
            if content_ratings:
                us = next((r for r in content_ratings if r.get("iso_3166_1") == "US"), None)
                cert = (us or content_ratings[0]).get("rating")

            catalog_rows.append({
                "id": data.get("id"),
                "name": data.get("name"),
                "original_name": data.get("original_name"),
                "first_air_date": data.get("first_air_date"),
                "last_air_date": data.get("last_air_date"),
                "status": data.get("status"),
                "type": data.get("type"),
                "in_production": data.get("in_production"),
                "number_of_seasons": data.get("number_of_seasons"),
                "number_of_episodes": data.get("number_of_episodes"),
                "episode_run_time_mean": (np.mean(data.get("episode_run_time")) if data.get("episode_run_time") else None),
                "popularity": data.get("popularity"),
                "vote_average": data.get("vote_average"),
                "vote_count": data.get("vote_count"),
                "num_cast": len(credits.get("cast", [])),
                "num_directors": sum(1 for c in credits.get("crew", []) if c.get("job") == "Director"),
                "keywords": ",".join([k["name"] for k in keywords]),
                "certification": cert,
                "imdb_id": external_ids.get("imdb_id"),
                "genres": genres,
                "networks": networks,
                "language": data.get("original_language"),
            })
        except Exception:
            continue

    cat_df = pd.DataFrame(catalog_rows).dropna(subset=["id"])
    cat_path = out / "tmdb_catalog.parquet"
    cat_df.to_parquet(cat_path, index=False)

    return {
        "run_date": run_date,
        "run_id": run_id,
        "week_start": week_start,
        "tmdb_catalog_local_path": str(cat_path),
        "n_top_ids": int(len(deduped)),
        "n_details_downloaded": int(n_details),
        "n_catalog_rows": int(cat_df.shape[0]),
    }
