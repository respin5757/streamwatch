# streamwatch/pipelines/dates.py
from __future__ import annotations

from datetime import datetime, timedelta

def week_start_from_ds(ds: str) -> str:
    """
    Given Airflow ds (YYYY-MM-DD), return Monday week_start (YYYY-MM-DD).
    """
    d = datetime.strptime(ds, "%Y-%m-%d").date()
    monday = d - timedelta(days=d.weekday())  # Monday=0
    return monday.isoformat()
