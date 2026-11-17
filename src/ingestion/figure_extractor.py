"""
Figure and table extraction from PDFs.
Saves each image to data/figures/ and pairs it with its caption.
"""
import fitz
from pathlib import Path
from typing import List, Dict, Any


def extract_figures(pdf_path: str, paper_id: str, figures_dir: str) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF and locate their captions.

    Returns a list of dicts:
        { paper_id, page, fig_id, image_path, caption, nearby_text }
    """
    # TODO: implement
    raise NotImplementedError
