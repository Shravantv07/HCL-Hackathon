"""
Integration layer between the chains and Members 1/2.

This is the ONLY module in src/chains that knows how teammate outputs look.
When Member 2's retriever (or Member 1's figure index) changes shape, update
the functions here; the chains themselves consume normalized dicts only.

Normalized passage (field names follow Member 1's chunk dicts, src/ingestion/indexer.py):
    {
        chunk_id: str,          # "{paper_id}__{chunk_type}__{chunk_index}" (indexer convention)
        paper_id: str,
        title: str,             # paper title; falls back to paper_catalogue.csv, then paper_id
        page: int | None,
        section: str,
        chunk_type: str,        # "text" | "table" | "figure"
        figure_ref: str,
        image_path: str,
        text: str,
        score: float | None,    # retrieval score: `score`, or Member 2's `rrf_score`
    }

Member 2's interface (src/retrieval/retriever.py):
    retrieve(query, top_k=6, paper_ids=None) -> [{"id", <chunk metadata>, "text", "rrf_score"}]
Note: `paper_ids` currently filters only the dense (Chroma) half of the hybrid search;
BM25 hits from other papers can still appear, so per-paper callers re-filter here.
"""

import csv
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "6"))
CATALOGUE_PATH = os.getenv("PAPER_CATALOGUE", "./paper_catalogue.csv")
FIGURES_INDEX_PATH = os.getenv("FIGURES_INDEX_PATH", "./data/figures_index.json")  # written by indexer.py
FIGURES_PATH = os.getenv("FIGURES_PATH", "./data/figures")  # same variable the indexer uses


class SourceError(RuntimeError):
    """Raised when passages or figures cannot be obtained from teammate components."""


@lru_cache(maxsize=1)
def _catalogue_titles() -> Dict[str, str]:
    """paper_id -> title from paper_catalogue.csv (empty dict if unavailable)."""
    path = Path(CATALOGUE_PATH)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["id"].strip(): row.get("title", "").strip()
            for row in csv.DictReader(f)
            if row.get("id", "").strip() and row.get("title", "").strip()
        }


def paper_title(paper_id: str) -> str:
    return _catalogue_titles().get(paper_id, "")


@lru_cache(maxsize=1)
def known_paper_ids() -> Optional[frozenset]:
    """All paper ids in paper_catalogue.csv, or None if the catalogue is unavailable."""
    path = Path(CATALOGUE_PATH)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return frozenset(row["id"].strip() for row in csv.DictReader(f) if row.get("id", "").strip())


def normalize_passage(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one retrieved record into the normalized passage format.

    Accepts Member 1's flat chunk dict (optionally with a score) or a Chroma-style
    record with `id` / `document` / `metadata`. Raises ValueError if the record has
    no text or no paper_id, because such a passage cannot be cited.
    """
    meta = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k != "metadata"}}

    text = (meta.get("text") or meta.get("document") or "").strip()
    paper_id = str(meta.get("paper_id") or "").strip()
    if not text:
        raise ValueError("retrieved record has no text")
    if not paper_id:
        raise ValueError("retrieved record has no paper_id")

    chunk_type = meta.get("chunk_type") or "text"
    chunk_id = meta.get("chunk_id") or meta.get("id")
    if not chunk_id and meta.get("chunk_index") is not None:
        chunk_id = f"{paper_id}__{chunk_type}__{meta['chunk_index']}"

    page = meta.get("page")
    try:
        page = int(page) if page not in (None, "") else None
    except (TypeError, ValueError):
        page = None

    score = meta.get("score")
    if score is None:
        score = meta.get("rrf_score")  # Member 2's hybrid retriever
    return {
        "chunk_id": str(chunk_id) if chunk_id else "",
        "paper_id": paper_id,
        "title": (meta.get("title") or "").strip() or paper_title(paper_id) or paper_id,
        "page": page,
        "section": meta.get("section") or "",
        "chunk_type": chunk_type,
        "figure_ref": meta.get("figure_ref") or "",
        "image_path": meta.get("image_path") or "",
        "text": text,
        "score": float(score) if isinstance(score, (int, float)) else None,
    }


def get_passages(
    question: str, top_k: Optional[int] = None, paper_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve raw passage records for a question through Member 2's retriever.

    `paper_ids` restricts the search to those papers: it is passed to retrieve(), and
    because retrieve() currently filters only its dense half, records from other papers
    are also dropped here. Records are otherwise returned as-is; chains normalize them
    with normalize_passage() so malformed records can be reported as warnings.
    Raises SourceError if retrieval fails. Never substitutes other data.
    """
    try:
        from src.retrieval.retriever import retrieve  # imported lazily: Member 2's module
        records = retrieve(question, top_k=top_k or TOP_K,
                           paper_ids=list(paper_ids) if paper_ids else None)
    except Exception as e:
        raise SourceError(f"Retrieval failed: {e}") from e

    records = list(records or [])
    if paper_ids:
        allowed = {str(p) for p in paper_ids}
        records = [r for r in records
                   if isinstance(r, dict)
                   and str(r.get("paper_id") or (r.get("metadata") or {}).get("paper_id") or "") in allowed]
    return records


def resolve_image_path(path: str) -> str:
    """
    Return a usable local path for a stored figure path.

    Member 1 stores absolute paths from the machine that ran extraction, which break
    on other machines. If the stored path does not exist, fall back to the same file
    name inside FIGURES_PATH. Returns the original path if neither exists (the caller
    reports it as missing).
    """
    if not path or Path(path).is_file():
        return path
    name = re.split(r"[\\/]", path)[-1]  # handles Windows and POSIX separators
    local = Path(FIGURES_PATH) / name
    return str(local) if name and local.is_file() else path


def normalize_figure(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a figure reference into the normalized figure format:
        {fig_id, paper_id, title, page, caption, nearby_text, image_path,
         image_bytes, mime_type, width, height}

    Accepts Member 1's figures_index.json records (fig_id, image_path, caption,
    nearby_text, page, paper_id, width, height), figure chunks returned by
    retrieval (figure_ref, image_path, text, ...), or a direct upload
    ({"image_bytes": ..., "mime_type": ...} plus any metadata). Missing metadata
    stays empty / None; it is never guessed.
    """
    meta = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k != "metadata"}}
    paper_id = str(meta.get("paper_id") or "").strip()
    caption = (meta.get("caption") or "").strip()
    nearby = (meta.get("nearby_text") or "").strip()
    if not caption and not nearby and meta.get("chunk_type") == "figure":
        nearby = (meta.get("text") or meta.get("document") or "").strip()  # chunk = caption + nearby text

    page = meta.get("page")
    try:
        page = int(page) if page not in (None, "") else None
    except (TypeError, ValueError):
        page = None

    return {
        "fig_id": str(meta.get("fig_id") or meta.get("figure_ref") or "").strip(),
        "paper_id": paper_id,
        "title": (meta.get("title") or "").strip() or (paper_title(paper_id) or paper_id if paper_id else ""),
        "page": page,
        "caption": caption,
        "nearby_text": nearby,
        "image_path": resolve_image_path(str(meta.get("image_path") or "").strip()),
        "image_bytes": meta.get("image_bytes"),
        "mime_type": meta.get("mime_type") or "",
        "width": meta.get("width"),
        "height": meta.get("height"),
    }


def get_figure(fig_id: str) -> Dict[str, Any]:
    """Look up one figure in Member 1's figures_index.json. Raises SourceError."""
    path = Path(FIGURES_INDEX_PATH)
    if not path.exists():
        raise SourceError(
            f"Figure index not found at {path}. Run ingestion (src/ingestion/indexer.py) first."
        )
    try:
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SourceError(f"Could not read figure index {path}: {e}") from e
    record = index.get(fig_id) if isinstance(index, dict) else None
    if not record:
        raise SourceError(f"Figure '{fig_id}' is not in the figure index.")
    return normalize_figure(record)


def find_figures(
    question: str, top_k: Optional[int] = None, paper_ids: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve figure chunks relevant to a question (via Member 2's retriever), in rank
    order, enriched from the figure index when possible. Raises SourceError.

    retrieve() cannot filter by chunk type, so text chunks compete for the same slots;
    the search is widened (3 x TOP_K by default) to leave room for figure chunks.
    """
    figures = []
    for record in get_passages(question, top_k=top_k or TOP_K * 3, paper_ids=paper_ids):
        meta = {**record.get("metadata", {}), **record}
        if meta.get("chunk_type") != "figure":
            continue
        figure = normalize_figure(record)
        if figure["fig_id"]:
            try:
                figure = {**figure, **{k: v for k, v in get_figure(figure["fig_id"]).items() if v}}
            except SourceError:
                pass  # the retrieved chunk already carries image_path and text
        figures.append(figure)
    return figures


def get_paper_passages(
    paper_id: str, queries: List[str], top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve passages belonging to one paper, one retrieval per query.

    The search is restricted with retrieve(paper_ids=[paper_id]). Because that filter
    currently applies only to the dense half of Member 2's hybrid search, results are
    also re-checked here and other papers' chunks dropped (top_k is widened a little to
    compensate). Results from the different queries are interleaved by rank so a context
    budget does not drop a whole query's evidence. Returns normalized passages;
    raises SourceError.
    """
    per_query: List[List[Dict[str, Any]]] = []
    for query in queries:
        matches = []
        for record in get_passages(query, top_k=top_k or TOP_K * 2, paper_ids=[paper_id]):
            try:
                p = normalize_passage(record)
            except (ValueError, AttributeError, TypeError):
                continue
            if p["paper_id"] == paper_id:
                matches.append(p)
        per_query.append(matches)

    interleaved, longest = [], max((len(m) for m in per_query), default=0)
    for rank in range(longest):
        interleaved.extend(m[rank] for m in per_query if rank < len(m))
    return interleaved
