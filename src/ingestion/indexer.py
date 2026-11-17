"""
End-to-end ingestion pipeline:
  parse_pdf → chunk_pages → extract_figures → chunk_figures → embed → ChromaDB

Run for a single paper:
    python -m src.ingestion.indexer --pdf data/papers/paper_01.pdf --id p01 --title "My Paper Title"

Run for all papers listed in paper_catalogue.csv:
    python -m src.ingestion.indexer --catalogue paper_catalogue.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from google import genai
from google.genai import types

from src.config import get_settings
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.figure_extractor import extract_figures
from src.ingestion.chunker import chunk_pages, chunk_figures

# ---------------------------------------------------------------------------
# Config (loaded centrally from .env / environment)
# ---------------------------------------------------------------------------

_settings          = get_settings()
CHROMA_PATH        = str(_settings.chroma_path)
FIGURES_PATH       = str(_settings.figures_path)
PAPERS_PATH        = str(_settings.papers_path)
CHUNK_SIZE         = _settings.chunk_size
CHUNK_OVERLAP      = _settings.chunk_overlap
EMBEDDING_MODEL    = _settings.embedding_model
GEMINI_API_KEY     = _settings.gemini_api_key
COLLECTION_NAME    = _settings.chroma_collection
FIGURES_INDEX_PATH = str(_settings.figures_index_path)


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def _get_collection():
    """Return (or create) the Chroma collection with Gemini embeddings."""

    class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self):
            api_key = get_settings().require_gemini_api_key()
            self._client = genai.Client(api_key=api_key)

        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            response = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                # One Content per chunk: a plain list of strings is treated as ONE
                # multi-part input by gemini-embedding-2 and returns a single embedding.
                contents=[types.Content(parts=[types.Part(text=t)]) for t in input],
            )
            return [e.values for e in response.embeddings]

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _already_indexed(collection, paper_id: str) -> bool:
    """Check if any chunks for this paper_id already exist in the collection."""
    results = collection.get(where={"paper_id": paper_id}, limit=1)
    return len(results["ids"]) > 0


def _upsert_chunks(collection, chunks: List[Dict[str, Any]]) -> None:
    """Embed and upsert a batch of chunks into Chroma."""
    if not chunks:
        return

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        chunk_id = f"{c['paper_id']}__{c['chunk_type']}__{c['chunk_index']}"
        ids.append(chunk_id)
        documents.append(c["text"])

        # Chroma metadata values must be str | int | float | bool
        meta = {
            "paper_id":    c["paper_id"],
            "title":       c.get("title", ""),
            "page":        c.get("page", 0),
            "section":     c.get("section", ""),
            "chunk_type":  c.get("chunk_type", "text"),
            "chunk_index": c.get("chunk_index", 0),
            "figure_ref":  c.get("figure_ref") or "",
            "image_path":  c.get("image_path") or "",
            "token_count": c.get("token_count", 0),
        }
        metadatas.append(meta)

    # Upsert in batches of 100 to avoid request size limits
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


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
# Single-paper ingestion
# ---------------------------------------------------------------------------

def index_paper(pdf_path: str, paper_id: str, paper_title: str) -> Dict[str, Any]:
    """
    Full pipeline for one paper. Skips if already indexed.
    Returns a summary dict with counts.
    """
    collection = _get_collection()

    if _already_indexed(collection, paper_id):
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
    print(f"[indexer] Embedding and indexing {len(all_chunks)} chunks ...")
    _upsert_chunks(collection, all_chunks)

    # Save figure metadata to JSON sidecar
    fig_index = _load_figures_index()
    for fig in figures:
        fig_index[fig["fig_id"]] = fig
    _save_figures_index(fig_index)

    print(f"[indexer] ✓ Done: {paper_id}")
    return {
        "paper_id":    paper_id,
        "status":      "indexed",
        "pages":       len(pages),
        "text_chunks": len(text_chunks),
        "figures":     len(figures),
        "fig_chunks":  len(fig_chunks),
    }


# ---------------------------------------------------------------------------
# Batch ingestion from catalogue
# ---------------------------------------------------------------------------

def index_from_catalogue(catalogue_path: str, papers_dir: str = PAPERS_PATH) -> None:
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
    parser.add_argument("--papers-dir", default=PAPERS_PATH)

    args = parser.parse_args()

    if args.pdf:
        if not args.id:
            parser.error("--id is required when using --pdf")
        index_paper(args.pdf, args.id, args.title)
    else:
        index_from_catalogue(args.catalogue, args.papers_dir)
