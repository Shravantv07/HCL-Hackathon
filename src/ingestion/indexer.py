"""
End-to-end ingestion pipeline:
  parse_pdf → chunk_pages → extract_figures → chunk_figures → embed → FAISS

The index is stored as two files:
  data/faiss_index/index.faiss   – the FAISS vector index
  data/faiss_index/metadata.json – chunk metadata aligned by position

Run for a single paper:
    python -m src.ingestion.indexer --pdf data/papers/paper_01.pdf --id p01 --title "My Paper Title"

Run for all papers listed in paper_catalogue.csv:
    python -m src.ingestion.indexer --catalogue paper_catalogue.csv
"""

import os
import json
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import faiss
import google.generativeai as genai
from dotenv import load_dotenv

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.figure_extractor import extract_figures
from src.ingestion.chunker import chunk_pages, chunk_figures

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")
FIGURES_PATH    = os.getenv("FIGURES_PATH", "./data/figures")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
FIGURES_INDEX_PATH = "./data/figures_index.json"

FAISS_INDEX_FILE    = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_FILE       = os.path.join(FAISS_INDEX_DIR, "metadata.json")
INDEXED_IDS_FILE    = os.path.join(FAISS_INDEX_DIR, "indexed_ids.json")

# Gemini text-embedding-004 outputs 768-dimensional vectors
EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Gemini embedding
# ---------------------------------------------------------------------------

def _embed(texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
    """
    Embed a list of texts using Gemini and return an (N, EMBEDDING_DIM) float32 array.
    Processes in batches of 100 (Gemini batch limit).
    """
    genai.configure(api_key=GEMINI_API_KEY)
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type=task_type,
        )
        all_embeddings.extend(result["embedding"])

    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------

def _load_index() -> faiss.Index:
    """Load existing FAISS index from disk, or create a new flat cosine index."""
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    if Path(FAISS_INDEX_FILE).exists():
        return faiss.read_index(FAISS_INDEX_FILE)
    # IndexFlatIP with normalised vectors == cosine similarity
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _save_index(index: faiss.Index) -> None:
    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)


def _load_metadata() -> List[Dict[str, Any]]:
    """Load chunk metadata list (position-aligned with FAISS index)."""
    if Path(METADATA_FILE).exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_metadata(metadata: List[Dict[str, Any]]) -> None:
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _load_indexed_ids() -> set:
    if Path(INDEXED_IDS_FILE).exists():
        with open(INDEXED_IDS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_indexed_ids(ids: set) -> None:
    with open(INDEXED_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(ids), f)


def _already_indexed(paper_id: str) -> bool:
    return paper_id in _load_indexed_ids()


# ---------------------------------------------------------------------------
# Figure index (JSON sidecar)
# ---------------------------------------------------------------------------

def _load_figures_index() -> Dict[str, Any]:
    path = Path(FIGURES_INDEX_PATH)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_figures_index(index: Dict[str, Any]) -> None:
    Path(FIGURES_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(FIGURES_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Core: add chunks to FAISS
# ---------------------------------------------------------------------------

def _add_chunks_to_index(chunks: List[Dict[str, Any]]) -> None:
    """Embed chunks and add them to the FAISS index + metadata store."""
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    print(f"[indexer]   Embedding {len(texts)} chunks ...")
    embeddings = _embed(texts)

    # Normalise to unit length so IndexFlatIP gives cosine similarity
    faiss.normalize_L2(embeddings)

    index    = _load_index()
    metadata = _load_metadata()

    index.add(embeddings)
    metadata.extend(chunks)

    _save_index(index)
    _save_metadata(metadata)


# ---------------------------------------------------------------------------
# Single-paper ingestion
# ---------------------------------------------------------------------------

def index_paper(pdf_path: str, paper_id: str, paper_title: str) -> Dict[str, Any]:
    """
    Full pipeline for one paper. Skips if already indexed.
    Returns a summary dict with counts.
    """
    if _already_indexed(paper_id):
        print(f"[indexer] {paper_id} already indexed — skipping.")
        return {"paper_id": paper_id, "status": "skipped"}

    print(f"[indexer] Parsing {pdf_path} ...")
    pages = parse_pdf(pdf_path, paper_id)
    print(f"[indexer]   {len(pages)} pages extracted")

    text_chunks = chunk_pages(
        pages,
        paper_title=paper_title,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    print(f"[indexer]   {len(text_chunks)} text/table chunks")

    print(f"[indexer] Extracting figures ...")
    figures = extract_figures(pdf_path, paper_id, FIGURES_PATH)
    print(f"[indexer]   {len(figures)} figures extracted")

    fig_chunks = chunk_figures(figures, paper_title=paper_title)
    print(f"[indexer]   {len(fig_chunks)} figure chunks")

    all_chunks = text_chunks + fig_chunks
    _add_chunks_to_index(all_chunks)

    # Mark paper as indexed
    indexed_ids = _load_indexed_ids()
    indexed_ids.add(paper_id)
    _save_indexed_ids(indexed_ids)

    # Save figure metadata sidecar
    fig_index = _load_figures_index()
    for fig in figures:
        fig_index[fig["fig_id"]] = fig
    _save_figures_index(fig_index)

    print(f"[indexer] ✓ Done: {paper_id}")
    return {
        "paper_id": paper_id,
        "status": "indexed",
        "pages": len(pages),
        "text_chunks": len(text_chunks),
        "figures": len(figures),
        "fig_chunks": len(fig_chunks),
        "total_chunks": len(all_chunks),
    }


# ---------------------------------------------------------------------------
# Batch ingestion from catalogue
# ---------------------------------------------------------------------------

def index_from_catalogue(catalogue_path: str, papers_dir: str = "./data/papers") -> None:
    """Read paper_catalogue.csv and index every paper whose PDF file exists."""
    with open(catalogue_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        paper_id = row.get("id", "").strip()
        filename = row.get("local_filename", "").strip()
        title    = row.get("title", filename).strip()

        if not paper_id or not filename:
            continue

        pdf_path = str(Path(papers_dir) / filename)
        if not Path(pdf_path).exists():
            print(f"[indexer] Skipping {paper_id}: file not found ({pdf_path})")
            continue

        try:
            result = index_paper(pdf_path, paper_id, title)
            print(f"[indexer] {result}")
        except Exception as e:
            print(f"[indexer] ERROR indexing {paper_id}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest research papers into ResearchIQ")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",       help="Path to a single PDF file")
    group.add_argument("--catalogue", help="Path to paper_catalogue.csv for batch ingestion")
    parser.add_argument("--id",         help="Paper ID (required with --pdf)")
    parser.add_argument("--title",      help="Paper title (optional with --pdf)", default="")
    parser.add_argument("--papers-dir", default="./data/papers")

    args = parser.parse_args()

    if args.pdf:
        if not args.id:
            parser.error("--id is required when using --pdf")
        index_paper(args.pdf, args.id, args.title)
    else:
        index_from_catalogue(args.catalogue, args.papers_dir)
