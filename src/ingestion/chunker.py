"""
Splits extracted page text into overlapping chunks with metadata.
Chunk types: text | figure | table
"""
from typing import List, Dict, Any


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 512, overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Chunk page text into overlapping segments.

    Each chunk dict:
        { paper_id, title, page, section, chunk_type, text, figure_ref }
    """
    # TODO: implement
    raise NotImplementedError
