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
        score: float | None,    # best available ranking score (see _score below)
    }

Member 2's interface (src/retrieval/retriever.py):
    Retriever().retrieve(query, top_k=None, paper_ids=None, rerank=None)
        -> [{"id", "text", <chunk metadata>, "score", "rrf_score", "rerank_score"?}]
    `score` means different things per source (cosine distance for dense hits, BM25 score
    for keyword hits), so the reranker / fusion scores are preferred.
One Retriever is created lazily and reused: building it loads every chunk and the BM25
index, so it must not be rebuilt per query (restart the app after re-indexing).
Paths come from src/config.py, so they resolve against the project root.
"""

import csv
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import PROJECT_ROOT, get_settings

_settings = get_settings()
TOP_K = _settings.top_k
CATALOGUE_PATH = os.getenv("PAPER_CATALOGUE") or str(PROJECT_ROOT / "paper_catalogue.csv")
FIGURES_INDEX_PATH = str(_settings.figures_index_path)  # written by indexer.py
FIGURES_PATH = str(_settings.figures_path)              # where the indexer saves images

_retriever = None  # shared Retriever instance, created on first use


class SourceError(RuntimeError):
    """Raised when passages or figures cannot be obtained from teammate components."""


def _get_retriever():
    """Create Member 2's Retriever once; a failed attempt is retried on the next call."""
    global _retriever
    if _retriever is None:
        from src.retrieval.retriever import Retriever  # imported lazily: Member 2's module
        _retriever = Retriever()
    return _retriever


def _is_transient(e: Exception) -> bool:
    """Network hiccups (SSL EOF, connection reset, timeouts, 5xx/429 from the embedding API)."""
    text = f"{type(e).__name__} {e}".lower()
    code = getattr(e, "code", None)
    return (code in {408, 429, 500, 502, 503, 504}
            or any(s in text for s in ("ssl", "eof", "connection", "timeout", "timed out", "deadline",
                                       "temporarily", "remoteprotocol")))


def _explain_retrieval_error(e: Exception) -> str:
    """Turn common setup failures into actionable messages."""
    text = str(e)
    if isinstance(e, ModuleNotFoundError) and "sentence_transformers" in text:
        return ("Reranker dependency missing: run `pip install sentence-transformers`, "
                "or set RERANK_ENABLED=false in .env.")
    if "does not exist" in text.lower() or type(e).__name__ == "NotFoundError":
        return (f"Search index not found at {_settings.chroma_path}. Build it with: "
                "python -m src.ingestion.indexer --catalogue paper_catalogue.csv")
    return f"Retrieval failed: {e}"


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


def _uploaded_titles() -> Dict[str, str]:
    """paper_id -> title of indexed user uploads (read live: uploads change at runtime)."""
    try:
        from src.ingestion.uploads import indexed_uploads
        return indexed_uploads()
    except Exception:
        return {}


def paper_title(paper_id: str) -> str:
    return _catalogue_titles().get(paper_id) or _uploaded_titles().get(paper_id, "")


@lru_cache(maxsize=1)
def _catalogue_ids() -> Optional[frozenset]:
    path = Path(CATALOGUE_PATH)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return frozenset(row["id"].strip() for row in csv.DictReader(f) if row.get("id", "").strip())


def known_paper_ids() -> Optional[frozenset]:
    """
    All registered paper ids: paper_catalogue.csv plus indexed uploads.
    None only if neither source is available (callers then skip id validation).
    """
    catalogue, uploads = _catalogue_ids(), frozenset(_uploaded_titles())
    if catalogue is None and not uploads:
        return None
    return (catalogue or frozenset()) | uploads


def find_mentioned_papers(text: str) -> List[str]:
    """
    Registered papers (catalogue + uploads) named in `text`, in order of first mention.
    Matches the paper id, full title, short title (before ':') and the short title
    without a trailing version number (e.g. "LLaVA-OneVision" for "LLaVA-OneVision-1.5").
    """
    titles = {**_uploaded_titles(), **_catalogue_titles()}
    lowered = (text or "").lower()
    found = []
    for pid, title in titles.items():
        short = title.split(":")[0].strip()
        aliases = {pid, title, short, re.sub(r"[-\s]v?\d+(\.\d+)*$", "", short)}
        positions = [m.start() for a in aliases if len(a.strip()) >= 4
                     for m in [re.search(rf"(?<![\w-]){re.escape(a.strip().lower())}(?![\w-])", lowered)] if m]
        if re.search(rf"\b{re.escape(pid)}\b", lowered):
            positions.append(lowered.index(pid))
        if positions:
            found.append((min(positions), pid))
    return [pid for _, pid in sorted(found)]


def refresh_retriever() -> None:
    """
    Drop the shared Retriever so the next query rebuilds it. Call after indexing new
    papers: the Retriever builds its BM25 index from the chunks present at creation.
    """
    global _retriever
    _retriever = None


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

    # Prefer comparable scores: reranker, then fusion; raw `score` is source-specific.
    score = next((meta[k] for k in ("rerank_score", "rrf_score", "score")
                  if isinstance(meta.get(k), (int, float))), None)
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
    question: str, top_k: Optional[int] = None, paper_ids: Optional[List[str]] = None,
    rerank: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve raw passage records for a question through Member 2's retriever.

    `paper_ids` restricts the search to those papers: it is passed to the retriever, and
    any record from another paper is also dropped here as a safety check (attribution
    must never depend on a teammate's filter). Records are otherwise returned as-is; chains normalize them
    with normalize_passage() so malformed records can be reported as warnings.
    Raises SourceError if retrieval fails. Never substitutes other data.
    """
    for attempt in range(2):
        try:
            records = _get_retriever().retrieve(
                query=question, top_k=top_k or TOP_K,
                paper_ids=list(paper_ids) if paper_ids else None,
                rerank=rerank,  # None = RERANK_ENABLED from config
            )
            break
        except Exception as e:
            # The query embedding is a network call; a dropped connection (SSL EOF,
            # reset, timeout) is worth one retry. Setup errors are not.
            if attempt == 0 and _is_transient(e):
                time.sleep(1.5)
                continue
            raise SourceError(_explain_retrieval_error(e)) from e

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
    paper_id: str, queries: List[str], top_k: Optional[int] = None, rerank: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieve passages belonging to one paper, one retrieval per query.

    The search is restricted with paper_ids=[paper_id] (and re-checked in get_passages);
    top_k defaults to 2 x TOP_K per query. Results from the different queries are interleaved by rank so a context
    budget does not drop a whole query's evidence. Returns normalized passages;
    raises SourceError.
    """
    per_query: List[List[Dict[str, Any]]] = []
    for query in queries:
        matches = []
        # No cross-encoder here: comparison/synthesis run 4-7 queries per paper, and reranking
        # them on CPU dominated their runtime; the search is already scoped to one paper.
        for record in get_passages(query, top_k=top_k or TOP_K * 2, paper_ids=[paper_id], rerank=rerank):
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
