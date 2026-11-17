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


def remove_from_index(paper_id: str) -> int:
    """
    Delete everything indexing wrote for `paper_id`: its Chroma chunks, its entries in
    figures_index.json and the extracted figure images. Returns the number of chunks
    removed. Safe to call when nothing was written.
    """
    import chromadb  # local import: only needed for cleanup

    cfg = get_settings()
    removed = 0
    try:
        collection = chromadb.PersistentClient(path=str(cfg.chroma_path)).get_collection(cfg.chroma_collection)
        ids = collection.get(where={"paper_id": paper_id})["ids"]
        if ids:
            collection.delete(ids=ids)
        removed = len(ids)
    except Exception:
        pass  # no collection yet -> nothing to remove

    try:
        fig_index = json.loads(cfg.figures_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        fig_index = None
    if isinstance(fig_index, dict):
        stale = [k for k, v in fig_index.items() if isinstance(v, dict) and v.get("paper_id") == paper_id]
        for key in stale:
            image_path = fig_index.pop(key).get("image_path")
            if image_path and Path(image_path).is_file():
                Path(image_path).unlink()
        if stale:
            tmp = cfg.figures_index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(fig_index, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cfg.figures_index_path)
    return removed


def delete_upload(paper_id: str) -> int:
    """
    Remove a user upload completely: index data, registry entry and saved file.
    Raises KeyError if `paper_id` is not a registered upload. Returns chunks removed.
    """
    with _lock:
        registry = load_registry()
        if paper_id not in registry:
            raise KeyError(paper_id)
        removed = remove_from_index(paper_id)
        registry.pop(paper_id)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, REGISTRY_PATH)
        (UPLOAD_DIR / f"{paper_id}.pdf").unlink(missing_ok=True)
        return removed


def text_to_pdf(text: str, title: str) -> bytes:
    """Render plain text / Markdown source as a simple multi-page PDF (keeps line breaks)."""
    import html
    import io

    body = "".join(f"<p>{html.escape(line) or '&nbsp;'}</p>" for line in text.splitlines())
    story = fitz.Story(html=f"<h2>{html.escape(title)}</h2>{body}",
                       user_css="p { font-size: 10pt; margin: 0 0 2pt 0; } h2 { font-size: 14pt; }")
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    page_rect = fitz.paper_rect("a4")
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(page_rect + (54, 54, -54, -54))
        story.draw(device)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def ingest_text_upload(content: bytes, filename: str) -> Dict[str, Any]:
    """Ingest a .txt / .md upload by converting it to PDF and using the normal PDF pipeline."""
    text = content.decode("utf-8", errors="replace").strip()
    if not text:
        raise UploadError("The text file is empty.")
    return ingest_upload(text_to_pdf(text, Path(filename).stem), filename)


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

        def fail(message: str, cause: Optional[Exception] = None) -> UploadError:
            remove_from_index(paper_id)  # never leave a failed paper half-indexed
            record.update(status="failed", error=message)
            _save_record(record)
            err = UploadError(message)
            err.__cause__ = cause
            return err

        # 1. Extract text (with OCR for image-only pages) BEFORE anything is indexed.
        from src.ingestion.pdf_parser import parse_pdf, tesseract_available
        try:
            pages = parse_pdf(str(pdf_path), paper_id)
        except Exception as e:
            raise fail(f"Could not read this PDF: {e}", e)
        if not sum(p["word_count"] for p in pages):
            note = "" if tesseract_available() else " (Tesseract is not installed on the server)"
            raise fail(f"No readable text was found in this PDF, even with OCR{note}.")

        # 2. Index the already-parsed pages (text chunks + figures).
        try:
            from src.ingestion.indexer import index_paper  # heavy import, only when needed
            summary = index_paper(str(pdf_path), paper_id, title, pages=pages)
        except Exception as e:
            raise fail(f"Indexing failed: {e}", e)

        record.update(status="indexed", indexed_at=_now(), pages=summary.get("pages"),
                      text_chunks=summary.get("text_chunks"), figures=summary.get("figures"))
        _save_record(record)
        return {**record, "duplicate": False}
