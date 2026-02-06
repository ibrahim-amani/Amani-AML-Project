from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from amani_aml.core.config import settings

app = FastAPI(
    title="AMANI AML API",
    version="0.1.0",
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

NEW_DATA_DIR = settings.DATA_LAKE_DIR / "new_data"

PROVIDER = settings.PROVIDER_NAME
FULL_META_FILE = NEW_DATA_DIR / f"{PROVIDER}_meta.json"
DELTA_META_FILE = NEW_DATA_DIR / f"{PROVIDER}_delta_meta.json"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _require_file(path: Path) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_file(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            yield chunk


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_parse_iso_to_ms(iso_str: str) -> int:
    # meta["exported_at_iso"] غالباً فيه timezone already، بس نعمل fallback
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _resolve_full_from_meta() -> tuple[Path, Dict[str, Any]]:
    """
    Reads {provider}_meta.json and returns (full_gz_path, meta_obj).
    meta must include: file, sha256, exported_at_iso, records...
    """
    _require_file(FULL_META_FILE)
    meta = _read_json(FULL_META_FILE)

    fname = meta.get("file")
    if not isinstance(fname, str) or not fname.strip():
        raise HTTPException(status_code=500, detail="Invalid FULL meta: missing 'file'")

    full_path = NEW_DATA_DIR / fname
    _require_file(full_path)

    return full_path, meta


def _resolve_delta_from_meta() -> tuple[Path, Dict[str, Any]]:
    """
    Reads {provider}_delta_meta.json and returns (delta_gz_path, meta_obj).
    delta meta must include: file, sha256, exported_at_iso, records...
    """
    _require_file(DELTA_META_FILE)
    meta = _read_json(DELTA_META_FILE)

    fname = meta.get("file")
    if not isinstance(fname, str) or not fname.strip():
        raise HTTPException(status_code=500, detail="Invalid DELTA meta: missing 'file'")

    delta_path = NEW_DATA_DIR / fname
    _require_file(delta_path)

    return delta_path, meta


# -------------------------------------------------------------------
#  FULL DATA MANIFEST (initial load)
# -------------------------------------------------------------------

@app.get("/amani/meta")
def get_full_manifest():
    """
    Manifest for FULL import (initial load).
    Driven by {provider}_meta.json (file name is dynamic).
    """
    full_path, meta = _resolve_full_from_meta()

    exported_at_iso = meta.get("exported_at_iso")
    if not isinstance(exported_at_iso, str) or not exported_at_iso.strip():
        # fallback: use file mtime
        timestamp_ms = int(full_path.stat().st_mtime * 1000)
        exported_at_iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
    else:
        timestamp_ms = _safe_parse_iso_to_ms(exported_at_iso)

    # IMPORTANT: we trust meta["sha256"] as the contract checksum
    sha256 = meta.get("sha256")
    if not isinstance(sha256, str) or not sha256.strip():
        # fallback: compute (shouldn't happen)
        sha256 = _file_sha256(full_path)

    return JSONResponse(
        {
            "file": "http://localhost:8000/amani/file",
            "provider": PROVIDER,
            "sha256": sha256,
            "timestamp": timestamp_ms,
            "exported_at_iso": exported_at_iso,
            "records": meta.get("records"),
            "format": meta.get("format", "jsonl.gz"),
            "mode": "FULL",
            # helpful linkage
            "meta_file": str(FULL_META_FILE.name),
            "full_gz_file": str(full_path.name),
        }
    )


# -------------------------------------------------------------------
# FULL FILE DOWNLOAD
# -------------------------------------------------------------------

@app.get("/amani/file")
def download_full_export():
    """
    Streams the latest FULL export:
    {provider}_{YYYY-MM-DDTHH}_FULL.jsonl.gz
    resolved from {provider}_meta.json
    """
    full_path, _meta = _resolve_full_from_meta()

    return StreamingResponse(
        _stream_file(full_path),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="{full_path.name}"'
        },
    )


# -------------------------------------------------------------------
#  DELTA MANIFEST
# -------------------------------------------------------------------

@app.get("/amani/delta/meta")
def get_delta_manifest():
    """
    Manifest for DELTA updates (NEW + UPDATED).
    Driven by {provider}_delta_meta.json
    """
    delta_path, meta = _resolve_delta_from_meta()

    exported_at_iso = meta.get("exported_at_iso")
    if not isinstance(exported_at_iso, str) or not exported_at_iso.strip():
        timestamp_ms = int(delta_path.stat().st_mtime * 1000)
        exported_at_iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()
    else:
        timestamp_ms = _safe_parse_iso_to_ms(exported_at_iso)

    sha256 = meta.get("sha256")
    if not isinstance(sha256, str) or not sha256.strip():
        sha256 = _file_sha256(delta_path)

    return JSONResponse(
        {
            "file": "http://localhost:8000/amani/delta/file",
            "provider": PROVIDER,
            "sha256": sha256,
            "timestamp": timestamp_ms,
            "exported_at_iso": exported_at_iso,
            "records": meta.get("records"),
            "new_records": meta.get("new_records"),
            "updated_records": meta.get("updated_records"),
            "format": meta.get("format", "jsonl.gz"),
            "mode": "DELTA",
            # helpful linkage
            "meta_file": str(DELTA_META_FILE.name),
            "delta_gz_file": str(delta_path.name),
            "full_file": meta.get("full_file"),
            "state_file": meta.get("state_file"),
        }
    )


# -------------------------------------------------------------------
#  DELTA FILE DOWNLOAD
# -------------------------------------------------------------------

@app.get("/amani/delta/file")
def download_delta_export():
    """
    Streams the latest DELTA export:
    {provider}_{YYYY-MM-DDTHH}_DELTA.jsonl.gz
    resolved from {provider}_delta_meta.json
    """
    delta_path, _meta = _resolve_delta_from_meta()

    return StreamingResponse(
        _stream_file(delta_path),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="{delta_path.name}"'
        },
    )


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/")
def health():
    return {"ok": True, "provider": PROVIDER}
