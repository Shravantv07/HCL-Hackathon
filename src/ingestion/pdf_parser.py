"""
PDF text extraction with page-level metadata.
Uses PyMuPDF as primary parser; falls back to OCR for image-only pages.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any


def parse_pdf(pdf_path: str, paper_id: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks from a PDF, one entry per page.

    Returns a list of dicts:
        { paper_id, page, text, is_ocr }
    """
    # TODO: implement
    raise NotImplementedError
