"""
Embeds chunks and stores them in ChromaDB.
Run once per paper; subsequent runs skip already-indexed papers.
"""
from typing import List, Dict, Any


def index_chunks(chunks: List[Dict[str, Any]], chroma_path: str) -> None:
    """Embed chunks and upsert into the Chroma collection."""
    # TODO: implement
    raise NotImplementedError


def index_paper(pdf_path: str, paper_id: str, paper_title: str, chroma_path: str, figures_dir: str) -> None:
    """End-to-end: parse → chunk → extract figures → index a single paper."""
    # TODO: wire parse_pdf → chunk_pages → extract_figures → index_chunks
    raise NotImplementedError
