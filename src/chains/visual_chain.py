"""
Visual Q&A chain: retrieve figure → send image + question to GPT-4o vision.
"""
from typing import Dict, Any


def answer_visual(question: str, fig_id: str = None) -> Dict[str, Any]:
    """
    Returns:
        { answer: str, image_path: str, source: { paper, page, fig_id } }
    """
    # TODO: implement
    raise NotImplementedError
