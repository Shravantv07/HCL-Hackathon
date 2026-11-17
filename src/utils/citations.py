"""
Formats citation references from chunk metadata.
"""
from typing import Dict, Any


def format_citation(chunk: Dict[str, Any]) -> str:
    """
    Returns a human-readable citation string from a chunk's metadata.
    Example: '[Retrieval-Augmented Generation Survey, p.4]'
    """
    title = chunk.get("title", "Unknown Paper")
    page = chunk.get("page", "?")
    fig_ref = chunk.get("figure_ref")
    if fig_ref:
        return f"[{title}, {fig_ref}, p.{page}]"
    return f"[{title}, p.{page}]"
