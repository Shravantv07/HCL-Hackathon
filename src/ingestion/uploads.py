"""
Automatic ingestion and registration of user-uploaded PDFs.

    ingest_upload(content: bytes, filename: str) -> record

Flow: validate → de-duplicate by SHA-256 → save to data/uploads/ → extract a title →
index_paper() (text chunks + figures into Chroma and figures_index.json) → record the
result in data/uploads/registry.json.

The registry is the upload counterpart of paper_catalogue.csv: the catalogue stays the
curated list of team papers, while uploads register themselves here. Nothing has to be
edited by hand. Registry record:
    {paper_id, title, filename, sha256, status: "indexing" | "indexed" | "failed",
     pages, text_chunks, figures, error, uploaded_at, indexed_at}
"""

import csv
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pymupdf as fitz

from src.config import PROJECT_ROOT, get_settings
from src.utils.security import validate_upload

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
REGISTRY_PATH = UPLOAD_DIR / "registry.json"
CATALOGUE_PATH = PROJECT_ROOT / "paper_catalogue.csv"

_lock = threading.Lock()  # serialises registry writes and indexing


class UploadError(ValueError):
    """The uploaded file cannot be ingested (invalid, empty, or indexing failed)."""


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def load_registry() -> Dict[str, Dict[str, Any]]:
    """paper_id -> record. Missing or unreadable registry -> {} (never raises)."""
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_record(record: Dict[str, Any]) -> None:
    """Insert/replace one record atomically (write temp file, then rename)."""
    registry = load_registry()
    registry[record["paper_id"]] = record
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, REGISTRY_PATH)


def indexed_uploads() -> Dict[str, str]:
    """paper_id -> title for every successfully indexed upload."""
    return {pid: r.get("title") or pid for pid, r in load_registry().items()
            if r.get("status") == "indexed"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _catalogue_paper_for(sha256: str) -> Optional[Dict[str, str]]:
    """If the file is one of the team papers in data/papers, return its catalogue row."""
    papers_dir = get_settings().papers_path
    if not CATALOGUE_PATH.exists():
        return None
    with open(CATALOGUE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = papers_dir / (row.get("local_filename") or "").strip()
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
                return row
    return None


def extract_title(pdf_path: Path, fallback: str) -> str:
    """PDF metadata title if meaningful, else the largest-font text on page 1, else `fallback`."""
    try:
        with fitz.open(str(pdf_path)) as doc:
            meta_title = (doc.metadata or {}).get("title", "").strip()
            if len(meta_title) > 8 and not re.search(r"untitled|microsoft word|\.pdf$|^doi", meta_title, re.I):
                return meta_title[:200]
            if doc.page_count:
                spans = [(span["size"], span["text"].strip())
                         for block in doc[0].get_text("dict")["blocks"]
                         for line in block.get("lines", []) for span in line["spans"]
                         if len(span["text"].strip()) > 3]
                if spans:
                    biggest = max(size for size, _ in spans)
                    title = " ".join(t for size, t in spans if size >= biggest - 0.5)
                    if 8 < len(title) < 250:
                        return re.sub(r"\s+", " ", title)
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def ingest_upload(content: bytes, filename: str) -> Dict[str, Any]:
    """
    Save, index and register an uploaded PDF. Returns the registry record plus
    `duplicate: bool`. Raises UploadError if the file is invalid or indexing fails
    (the failure is also recorded in the registry).
    """
    if not content or not content.startswith(b"%PDF-"):
        raise UploadError("Invalid PDF file.")
    sha256 = hashlib.sha256(content).hexdigest()

    # Already one of the team's catalogue papers: reuse it instead of indexing a copy.
    row = _catalogue_paper_for(sha256)
    if row:
        return {"paper_id": row["id"].strip(), "title": row.get("title", "").strip() or filename,
                "filename": filename, "sha256": sha256, "status": "indexed",
                "source": "catalogue", "duplicate": True}

    paper_id = f"u{sha256[:12]}"  # deterministic: the same file always maps to the same id
    with _lock:
        existing = load_registry().get(paper_id)
        if existing and existing.get("status") == "indexed":
            return {**existing, "duplicate": True}

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path = UPLOAD_DIR / f"{paper_id}.pdf"
        pdf_path.write_bytes(content)
        try:
            validate_upload(str(pdf_path))
        except ValueError as e:
            pdf_path.unlink(missing_ok=True)
            raise UploadError(str(e)) from e

        title = extract_title(pdf_path, fallback=Path(filename).stem)
        record = {"paper_id": paper_id, "title": title, "filename": filename, "sha256": sha256,
                  "status": "indexing", "pages": None, "text_chunks": None, "figures": None,
                  "error": None, "uploaded_at": _now(), "indexed_at": None, "source": "upload"}
        _save_record(record)

        try:
            from src.ingestion.indexer import index_paper  # heavy import, only when needed
            summary = index_paper(str(pdf_path), paper_id, title)
            if summary.get("status") == "indexed" and not summary.get("text_chunks"):
                raise UploadError("No extractable text was found in this PDF (is it a scanned image?).")
        except Exception as e:
            record.update(status="failed", error=str(e))
            _save_record(record)
            raise e if isinstance(e, UploadError) else UploadError(f"Indexing failed: {e}") from e

        record.update(status="indexed", indexed_at=_now(), pages=summary.get("pages"),
                      text_chunks=summary.get("text_chunks"), figures=summary.get("figures"))
        _save_record(record)
        return {**record, "duplicate": False}
