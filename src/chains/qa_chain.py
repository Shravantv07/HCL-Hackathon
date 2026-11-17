"""
Core Q&A chain: retrieve → prompt → LLM → answer + citations.
"""
from typing import Dict, Any


def answer(question: str) -> Dict[str, Any]:
    """
    Returns:
        { answer: str, citations: list[{ paper, page, fig_ref }], chunks_used: list }
    """
    # TODO: implement
    raise NotImplementedError
