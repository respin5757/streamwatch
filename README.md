# StreamWatch  
Multi-horizon forecasting of TV show popularity (Google Trends) with a production-style data/ML pipeline.

[![StreamWatch Weekly Refresh](https://github.com/respin5757/streamwatch/actions/workflows/main.yml/badge.svg)](https://github.com/respin5757/streamwatch/actions/workflows/main.yml)

## Overview

StreamWatch forecasts weekly Google Trends interest (0–100) for TV shows **1–4 weeks ahead** using TMDB metadata + Google Trends time series. The main goal of this project is to demonstrate an end-to-end workflow: 

ingestion → feature panel → training → versioned artifacts → serving.

The Streamlit app is a **read-only serving layer**. It does not train models or rebuild data; it only loads published artifacts.

Live app: https://streamwatch.streamlit.app/  
Repo: https://github.com/respin5757/streamwatch

---

## What it does

Given historical weekly interest and show metadata, StreamWatch answers:

**“How popular will this show be 1–4 weeks from now?”**

Outputs:
- Forecasts for t+1, t+2, t+3, t+4
- Catalog scan (top gainers/losers)
- Metrics by horizon (RMSE/R², run history)
- Basic data health checks (coverage/missingness; lightweight drift signals)

---

## High-level architecture

TMDB API + Google Trends  
→ ingestion + normalization  
→ weekly panel (show_id × week_start)  
→ feature engineering + multi-horizon targets  
→ train per-horizon models  
→ publish artifacts to Google Cloud Storage  
→ Streamlit app loads artifacts and renders forecasts

<div align="center"> 
<img width="888" height="589" alt="Screenshot 2026-01-04 at 4 31 55 PM" src="https://github.com/user-attachments/assets/7f076757-4c7d-4561-a8a8-4e3aac73eb1c" />
</div>

---

## Orchestration approach (Airflow locally → GitHub Actions online)

I built the pipeline in two phases.

### 1) Local development: Dockerized Airflow DAG

For development and debugging, I orchestrated the pipeline with **Apache Airflow** running locally via Docker Compose. This made it easier to iterate with:
- task-level logs
- retries/fail-fast boundaries
- backfills for historical weeks
- clear separation of stages (not just a script chain)

Airflow local run:

    docker compose up airflow-init
    docker compose up

Airflow UI: http://localhost:8080

DAG location:
- dags/streamwatch_weekly.py

Conceptual flow:

    extract_tmdb
      ↓
    extract_trends
      ↓
    build_panel
      ↓
    build_features
      ↓
    train_models
      ↓
    publish_artifacts
      ↓
    validate_serving_contract

<img width="1353" height="454" alt="Screenshot 2026-01-04 at 4 21 35 PM" src="https://github.com/user-attachments/assets/0f399f0b-5738-4dbd-8deb-5339d0328c02" />


### 2) Online automation: GitHub Actions weekly refresh (free)

To keep the project running online without paying for managed orchestration, I moved the scheduled run to **GitHub Actions**. The workflow runs weekly and:
- pulls the latest TMDB + Trends data
- rebuilds the panel + features
- retrains models (t+1…t+4)
- writes versioned artifacts to a GCS bucket
- updates a serving manifest (atomic pointer) so the Streamlit app can safely load the latest run

The Airflow work is still useful because it forced a clean DAG design and clear stage boundaries; GitHub Actions is just the free scheduled runner.

<img width="1092" height="455" alt="Screenshot 2026-01-13 at 12 47 12 PM" src="https://github.com/user-attachments/assets/7565be2b-71fe-4456-a51c-a7ff9780ef10" />


---

## Data sources

- TMDB API (show metadata: genres, networks, ratings, popularity, etc.)
- Google Trends (weekly interest time series per show, normalized 0–100)

---

## Modeling

Target:
- Google Trends weekly interest (0–100)

Approach:
- separate regression model per horizon: **t+1, t+2, t+3, t+4**
- time-aware validation (no random split)

Primary model:
- LightGBM regressor

Features (examples):
- lagged interest
- rolling stats
- TMDB popularity/ratings
- encoded categoricals (status, language, type)
- keyword/network flags
<div align="center"> 
<img width="748" height="516" alt="Screenshot 2026-01-13 at 12 47 59 PM" src="https://github.com/user-attachments/assets/a93c531e-f584-4326-8cc7-1dba97e4e2bd" />
</div>

---

## Model evaluation and metrics

Metrics are logged per run and per forecast horizon (t+1…t+4):
- RMSE
- R²
- training and validation row counts

### Latest run (example)

| family | horizon | rmse_train | rmse_val | r2_train | r2_val | n_train | n_val | cutoff_week | run_id | run_date | week_start | created_utc | n_rows | n_cols | n_features | n_unique_weeks | n_unique_shows |
|--------|---------|------------|----------|----------|--------|---------|-------|-------------|--------|----------|------------|--------------|--------|--------|------------|----------------|----------------|
| lgbm | 1 | 8.387431735 | 16.647371576 | 0.7690226644 | 0.5323371384 | 30514 | 7592 | 2025-01-05 | manual_test_20251230 | 12/30/25 | 12/29/25 | 2026-01-08T03:14:22Z | 38252 | 274 | 259 | 262 | 146 |
| lgbm | 2 | 8.738684830 | 17.730467136 | 0.7494568847 | 0.4727693757 | 30514 | 7446 | 2025-01-05 | manual_test_20251230 | 12/30/25 | 12/29/25 | 2026-01-08T03:14:22Z | 38252 | 274 | 259 | 262 | 146 |
| lgbm | 3 | 8.842484096 | 18.413698990 | 0.7432307887 | 0.4346804168 | 30514 | 7300 | 2025-01-05 | manual_test_20251230 | 12/30/25 | 12/29/25 | 2026-01-08T03:14:22Z | 38252 | 274 | 259 | 262 | 146 |
| lgbm | 4 | 8.957980252 | 19.209889596 | 0.7365111395 | 0.3889348361 | 30514 | 7154 | 2025-01-05 | manual_test_20251230 | 12/30/25 | 12/29/25 | 2026-01-08T03:14:22Z | 38252 | 274 | 259 | 262 | 146 |



### Metrics visualization

<p align="center">
  <img width="1281" height="407" alt="Screenshot 2026-01-13 at 1 53 01 PM" src="https://github.com/user-attachments/assets/4222b917-83a3-4689-ba7b-7ac1d87a9b70">
</p>


---


## Artifact layout + serving contract

Artifacts are published to Google Cloud Storage, organized like:

    data/
      panel_clean.parquet
      feature_columns.json
      metrics_history.parquet
      serving_manifest.json  <-- source of truth

    models/
      lgbm/
        lgbm_interest_t+1.pkl
        lgbm_interest_t+2.pkl
        lgbm_interest_t+3.pkl
        lgbm_interest_t+4.pkl
      hgbr/
        hgbm_interest_t+1.pkl
        hgbm_interest_t+2.pkl
        hgbm_interest_t+3.pkl
        hgbm_interest_t+4.pkl

### Serving manifest

The Streamlit app only reads `serving_manifest.json` first. That file points to the exact artifact paths for a given run and is written last to avoid partial deploys.

Example:

    {
    "run_id": "manual_test_20251230",
    "run_date": "2025-12-30",
    "week_start": "2025-12-29",
    "panel_clean_remote_path": "data/panel_clean.parquet",
    "feature_columns_remote_path": "data/feature_columns.json",
    "metrics_history_remote_path": "data/metrics_history.parquet",
    "models": {
      "hgbr_remote_dir": "models/hgbr",
      "lgbm_remote_dir": "models/lgbm"
    }
  }

---

## Streamlit app (stateless serving)

The app:
- loads artifacts via the serving manifest
- generates forecasts for selected shows
- displays metrics and monitoring views

The app does not retrain models or write data.
<div align="center"> 
<img width="862" height="666" alt="Screenshot 2026-01-13 at 12 49 00 PM" src="https://github.com/user-attachments/assets/018d4c22-d923-48b9-bba8-a3d059fc13bd" />
</div>

---
## Getting started

### Prerequisites
- Python 3.10+
- Docker + Docker Compose (for local Airflow only)
- TMDB API key
- Google Cloud Storage bucket + credentials (for training and artifact upload)

### Run the Streamlit app locally (serving only)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open:
- http://localhost:8501

### Environment variables

Set these via a `.env` file, shell exports, or Streamlit secrets:

```bash
TMDB_API_KEY=your_tmdb_key
GCS_BUCKET=your_bucket_name
GCP_PROJECT=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

### Run the pipeline locally (Airflow, dev/debug)

```bash
docker compose up airflow-init
docker compose up
```

Airflow UI:
- http://localhost:8080

Trigger the DAG:
- `streamwatch_weekly`

### Run the pipeline online (GitHub Actions)

The production pipeline runs on a weekly schedule using GitHub Actions:
- Workflow: `.github/workflows/main.yml`
- Publishes versioned artifacts to GCS
- Updates the serving manifest atomically

---

## Tech stack

- Python (pandas, NumPy)
- LightGBM, scikit-learn
- HistGradientBoostingRegressor 
- Apache Airflow (local), Docker/Docker Compose
- GitHub Actions (weekly schedule)
- Google Cloud Storage
- Streamlit
- Parquet

---

## Future work

- managed Airflow / Composer
- alerting on metric degradation
- uncertainty intervals
- stronger drift checks

---

## Links

- GitHub: https://github.com/respin5757/streamwatch  
- Live app: https://streamwatch.streamlit.app/  
- Portfolio: https://respin5757.github.io/personal-website/index.html  
