
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from amani_aml.core.config import settings

app = FastAPI(
    title="AMANI AML API",
    version="1.1.0",
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

NEW_DATA_DIR = settings.DATA_LAKE_DIR / "new_data"

GOLDEN_FILE = NEW_DATA_DIR / "Golden_Export.jsonl.gz"
FULL_META_FILE = NEW_DATA_DIR / "AmaniAI_meta.json"

DELTA_FILE = NEW_DATA_DIR / "Golden_Export.delta.jsonl.gz"
DELTA_META_FILE = NEW_DATA_DIR / "AmaniAI_delta_meta.json"

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


def _load_delta_profiles() -> List[Dict[str, Any]]:
    """
    Load delta profiles from Golden_Export.delta.jsonl.gz
    """
    profiles: List[Dict[str, Any]] = []
    with gzip.open(DELTA_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                profiles.append(json.loads(line))
    return profiles


# -------------------------------------------------------------------
#  FULL DATA MANIFEST (initial load)
# -------------------------------------------------------------------

@app.get("/amani/meta")
def get_full_manifest():
    """
    Manifest for FULL import (initial load).
    """
    _require_file(GOLDEN_FILE)

    sha256 = _file_sha256(GOLDEN_FILE)
    timestamp_ms = int(GOLDEN_FILE.stat().st_mtime * 1000)

    return JSONResponse(
        {
            "file": "http://localhost:8000/amani/file",
            "sha256": sha256,
            "timestamp": timestamp_ms,
            "exported_at_iso": datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat(),
            "mode": "FULL",
        }
    )


# -------------------------------------------------------------------
# FULL FILE DOWNLOAD
# -------------------------------------------------------------------

@app.get("/amani/file")
def download_full_export():
    """
    Streams Golden_Export.jsonl.gz
    """
    _require_file(GOLDEN_FILE)

    return StreamingResponse(
        _stream_file(GOLDEN_FILE),
        media_type="application/gzip",
        headers={
            "Content-Disposition": 'attachment; filename="Golden_Export.jsonl.gz"'
        },
    )


# -------------------------------------------------------------------
#  DELTA MANIFEST
# -------------------------------------------------------------------

@app.get("/amani/delta/meta")
def get_delta_manifest():
    """
    Manifest for DELTA updates (NEW + UPDATED).
    """
    _require_file(DELTA_FILE)
    _require_file(DELTA_META_FILE)

    meta = _read_json(DELTA_META_FILE)

    timestamp_ms = int(
        datetime.fromisoformat(meta["exported_at_iso"])
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )

    return JSONResponse(
        {
            "file": "http://localhost:8000/amani/delta/file",
            "sha256": meta["sha256"],
            "timestamp": timestamp_ms,
            "records": meta["records"],
            "new_records": meta["new_records"],
            "updated_records": meta["updated_records"],
            "mode": "DELTA",
        }
    )


# -------------------------------------------------------------------
#  DELTA FILE DOWNLOAD
# -------------------------------------------------------------------

@app.get("/amani/delta/file")
def download_delta_export():
    """
    Streams Golden_Export.delta.jsonl.gz
    """
    _require_file(DELTA_FILE)

    return StreamingResponse(
        _stream_file(DELTA_FILE),
        media_type="application/gzip",
        headers={
            "Content-Disposition": 'attachment; filename="Golden_Export.delta.jsonl.gz"'
        },
    )


# -------------------------------------------------------------------
#  UPDATE ENDPOINT (used by _get_update_batch)
# -------------------------------------------------------------------

@app.get("/amani/update/{timestamp}")
def get_updates_since(timestamp: str):
    """
    Incremental update endpoint.

    If the client timestamp is older than the DELTA export timestamp,
    return NEW + UPDATED profiles.
    Otherwise, return empty updates.
    """
    try:
        client_ts = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    _require_file(DELTA_META_FILE)

    meta = _read_json(DELTA_META_FILE)

    delta_ts = int(
        datetime.fromisoformat(meta["exported_at_iso"])
        .replace(tzinfo=timezone.utc)
        .timestamp() * 1000
    )

    if client_ts >= delta_ts or meta["records"] == 0:
        return JSONResponse(
            {
                "profiles": [],
                "timestamp": delta_ts,
                "mode": "DELTA",
            }
        )

    # Return delta profiles
    _require_file(DELTA_FILE)
    profiles = _load_delta_profiles()

    return JSONResponse(
        {
            "profiles": profiles,
            "timestamp": delta_ts,
            "mode": "DELTA",
            "new_records": meta["new_records"],
            "updated_records": meta["updated_records"],
        }
    )


# -------------------------------------------------------------------
# Health
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}
