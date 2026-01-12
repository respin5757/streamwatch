# streamwatch/pipelines/build_feature_panel.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def run(
    run_date: str,
    run_id: str,
    week_start: str,
    *,
    in_panel_local_path: str,
    out_panel_clean_local_path: str,
    tfidf_vocab_local_path: str,
    tfidf_max_features: int = 200,
) -> Dict:
    df = pd.read_parquet(in_panel_local_path)

    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["last_air_date"] = pd.to_datetime(df.get("last_air_date"), errors="coerce")
    df["first_air_date"] = pd.to_datetime(df.get("first_air_date"), errors="coerce")

    # freshness
    freshness_raw = (df["week_start"] - df["last_air_date"]).dt.days
    df["freshness_raw"] = freshness_raw
    df["freshness_days"] = df["freshness_raw"].clip(lower=0)
    df["freshness_future_flag"] = (df["freshness_raw"] < 0).astype(int)

    # certification
    if "certification" in df.columns and not df["certification"].dropna().empty:
        mode = df["certification"].mode().iloc[0]
        df["certification"] = df["certification"].fillna(mode)

    # runtime
    if "episode_run_time_mean" in df.columns and "genres" in df.columns:
        df["episode_run_time_mean"] = df.groupby("genres")["episode_run_time_mean"].transform(lambda x: x.fillna(x.median()))
        df["episode_run_time_mean"] = df["episode_run_time_mean"].fillna(df["episode_run_time_mean"].median())

    # genres multi-hot
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].fillna("").apply(lambda s: [g.strip() for g in str(s).split(",") if g.strip()])
        counts = Counter(g for lst in df["genres_list"] for g in lst)
        top = [g for g, _ in counts.most_common(15)]
        df["genres_filtered"] = df["genres_list"].apply(lambda lst: [g for g in lst if g in top])
        mlb = MultiLabelBinarizer()
        mat = mlb.fit_transform(df["genres_filtered"])
        gdf = pd.DataFrame(mat, columns=[f"genre__{g}" for g in mlb.classes_], index=df.index)
        df = pd.concat([df.drop(columns=["genres_list", "genres_filtered"]), gdf], axis=1)

    # networks multi-hot
    if "networks" in df.columns:
        df["networks_list"] = df["networks"].fillna("").apply(lambda s: [n.strip() for n in str(s).split(",") if n.strip()])
        counts = Counter(n for lst in df["networks_list"] for n in lst)
        top = [n for n, _ in counts.most_common(20)]
        df["networks_filtered"] = df["networks_list"].apply(lambda lst: [n for n in lst if n in top])
        mlb = MultiLabelBinarizer()
        mat = mlb.fit_transform(df["networks_filtered"])
        ndf = pd.DataFrame(mat, columns=[f"net__{n}" for n in mlb.classes_], index=df.index)
        df = pd.concat([df.drop(columns=["networks_list", "networks_filtered"]), ndf], axis=1)

    # TF-IDF keywords with stable vocab
    if "keywords" in df.columns:
        old_kw = [c for c in df.columns if c.startswith("kw__")]
        if old_kw:
            df = df.drop(columns=old_kw)

        df["keywords"] = df["keywords"].fillna("").astype(str)

        vocab_path = Path(tfidf_vocab_local_path)
        vocab = None
        if vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

        if vocab:
            tfidf = TfidfVectorizer(vocabulary=vocab)
            mat = tfidf.fit_transform(df["keywords"])
            names = tfidf.get_feature_names_out()
        else:
            tfidf = TfidfVectorizer(max_features=tfidf_max_features)
            mat = tfidf.fit_transform(df["keywords"])
            names = tfidf.get_feature_names_out()
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            vocab = {k: int(v) for k, v in tfidf.vocabulary_.items()}
            vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")


        kw = pd.DataFrame(mat.toarray(), columns=[f"kw__{w}" for w in names], index=df.index)
        df = pd.concat([df, kw], axis=1)

    # categoricals
    for c in ["status", "type", "language", "certification"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    out_path = Path(out_panel_clean_local_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "run_date": run_date,
        "run_id": run_id,
        "week_start": week_start,
        "panel_clean_local_path": str(out_path),
        "tfidf_vocab_local_path": str(tfidf_vocab_local_path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
    }
