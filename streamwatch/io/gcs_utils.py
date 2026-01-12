# streamwatch/io/gcs_utils.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional, Union

from google.cloud import storage
from google.oauth2 import service_account


PathLike = Union[str, Path]


def _get_client(project: str | None = None) -> storage.Client:
    """
    Auth priority:
      1) Streamlit secrets: st.secrets["STREAMWATCH_GCP_SA_JSON"]  (recommended on Streamlit Cloud)
      2) Env var: STREAMWATCH_GCP_SA_JSON (JSON string)
      3) ADC (local dev only) — but we fail fast on Streamlit Cloud to avoid metadata TransportError
    """
    info = None

    # 1) Streamlit secrets
    try:
        import streamlit as st

        if "STREAMWATCH_GCP_SA_JSON" in st.secrets:
            sa_obj = st.secrets["STREAMWATCH_GCP_SA_JSON"]

            # st.secrets nested tables aren't always plain dicts — force to dict
            info = dict(sa_obj)

    except Exception:
        pass

    # 2) Env var (JSON string)
    if info is None:
        sa_json = os.getenv("STREAMWATCH_GCP_SA_JSON")
        if sa_json and str(sa_json).strip():
            s = str(sa_json).strip()

            # If the JSON got wrapped in extra quotes, strip them
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                s = s[1:-1].strip()

            info = json.loads(s)

    # If we have SA info, use it
    if info is not None:
        creds = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=project or info.get("project_id"), credentials=creds)

    # 3) If running on Streamlit Cloud and we got here, ADC will crash -> raise a helpful error
    if os.getenv("STREAMLIT_CLOUD") == "true" or os.getenv("STREAMLIT_SERVER_HEADLESS") == "true":
        raise RuntimeError(
            "No service account credentials found. Set STREAMWATCH_GCP_SA_JSON in Streamlit Secrets "
            "as a TOML object/table (not a quoted JSON string)."
        )

    # local fallback
    return storage.Client(project=project) if project else storage.Client()



def _norm_prefix(prefix: str) -> str:
    return prefix.strip("/") if prefix else ""


def _blob_name(prefix: str, remote_path: str) -> str:
    p = _norm_prefix(prefix)
    rp = remote_path.lstrip("/")
    return f"{p}/{rp}" if p else rp


def blob_exists(remote_path: str, *, bucket: str, prefix: str = "", project: str | None = None) -> bool:
    client = _get_client(project)
    b = client.bucket(bucket)
    name = _blob_name(prefix, remote_path)
    return b.blob(name).exists(client)


# NOTE: intentionally positional-friendly for backward compatibility
def upload_file(
    local_path: PathLike,
    remote_path: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
    content_type: str | None = None,
) -> str:
    lp = Path(local_path)
    if not lp.exists():
        raise FileNotFoundError(f"Local file not found: {lp}")
    client = _get_client(project)
    b = client.bucket(bucket)
    name = _blob_name(prefix, remote_path)
    blob = b.blob(name)
    blob.upload_from_filename(str(lp), content_type=content_type)
    return f"gs://{bucket}/{name}"


def upload_bytes(
    data: bytes,
    remote_path: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
    content_type: str | None = None,
    if_generation_match: int | None = None,
) -> str:
    client = _get_client(project)
    b = client.bucket(bucket)
    name = _blob_name(prefix, remote_path)
    blob = b.blob(name)
    blob.upload_from_string(
        data,
        content_type=content_type,
        if_generation_match=if_generation_match,
    )
    return f"gs://{bucket}/{name}"


# NOTE: also positional-friendly
def download_file(
    remote_path: str,
    local_path: PathLike,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
) -> Path:
    out = Path(local_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = _get_client(project)
    b = client.bucket(bucket)
    name = _blob_name(prefix, remote_path)
    blob = b.blob(name)

    if not blob.exists(client):
        raise FileNotFoundError(f"GCS object not found: gs://{bucket}/{name}")

    blob.download_to_filename(str(out))
    return out


def upload_dir(
    local_dir: PathLike,
    remote_dir: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
    include_suffixes: Iterable[str] | None = None,
    content_type: str | None = None,
) -> list[str]:
    ld = Path(local_dir)
    if not ld.exists():
        raise FileNotFoundError(f"Local dir not found: {ld}")

    include = tuple(include_suffixes) if include_suffixes else None
    remote_dir = remote_dir.strip("/")

    uris: list[str] = []
    for p in ld.rglob("*"):
        if not p.is_file():
            continue
        if include and p.suffix not in include:
            continue
        rel = p.relative_to(ld).as_posix()
        uris.append(
            upload_file(
                p,
                f"{remote_dir}/{rel}",
                bucket=bucket,
                prefix=prefix,
                project=project,
                content_type=content_type,
            )
        )
    return uris


def download_dir(
    remote_dir: str,
    local_dir: PathLike,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
) -> list[Path]:
    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client(project)
    p = _norm_prefix(prefix)

    base = f"{p}/{remote_dir.strip('/')}".strip("/") if p else remote_dir.strip("/")
    if base and not base.endswith("/"):
        base += "/"

    downloaded: list[Path] = []
    for blob in client.list_blobs(bucket, prefix=base):
        rel = blob.name[len(base):] if base else blob.name
        if not rel or rel.endswith("/"):
            continue
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(out_path))
        downloaded.append(out_path)

    return downloaded


def get_blob_generation(remote_path: str, *, bucket: str, prefix: str = "", project: str | None = None) -> int | None:
    client = _get_client(project)
    b = client.bucket(bucket)
    name = _blob_name(prefix, remote_path)
    blob = b.blob(name)
    if not blob.exists(client):
        return None
    blob.reload(client=client)
    return int(blob.generation)


def write_json_to_gcs(
    payload: dict,
    remote_path: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
    if_generation_match: int | None = None,
) -> str:
    data = json.dumps(payload, indent=2).encode("utf-8")
    return upload_bytes(
        data,
        remote_path=remote_path,
        bucket=bucket,
        prefix=prefix,
        project=project,
        content_type="application/json",
        if_generation_match=if_generation_match,
    )


def atomic_update_json_pointer(
    payload: dict,
    pointer_path: str,
    bucket: str,
    prefix: str = "",
    project: str | None = None,
) -> str:
    
    gen = get_blob_generation(pointer_path, bucket=bucket, prefix=prefix, project=project)
    if gen is None:
        return write_json_to_gcs(
            payload,
            remote_path=pointer_path,
            bucket=bucket,
            prefix=prefix,
            project=project,
            if_generation_match=0,
        )
    return write_json_to_gcs(
        payload,
        remote_path=pointer_path,
        bucket=bucket,
        prefix=prefix,
        project=project,
        if_generation_match=gen,
    )
