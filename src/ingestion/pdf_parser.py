"""
PDF text extraction with page-level metadata.
Uses PyMuPDF as primary parser; falls back to OCR for image-only pages.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz
import pytesseract
from PIL import Image

from src.config import get_settings


def _page_to_image(page: fitz.Page, dpi: Optional[int] = None) -> Image.Image:
    """Render a PDF page to a PIL image for OCR."""
    effective_dpi = dpi if dpi is not None else get_settings().ocr_dpi
    mat = fitz.Matrix(effective_dpi / 72, effective_dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _is_mostly_empty(text: str, min_chars: Optional[int] = None) -> bool:
    """Return True if the extracted text is too short to be useful."""
    threshold = min_chars if min_chars is not None else get_settings().ocr_min_chars
    return len(text.strip()) < threshold


def parse_pdf(pdf_path: str, paper_id: str) -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF.

    Strategy:
      1. Use PyMuPDF to extract text blocks (handles multi-column via sort=True).
      2. If a page yields fewer than 50 characters, fall back to Tesseract OCR.

    Returns a list of page dicts:
        {
            paper_id: str,
            page: int,          # 1-indexed
            text: str,
            is_ocr: bool,
            word_count: int
        }
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Dict[str, Any]] = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Primary: PyMuPDF text extraction
            # sort=True reads left-to-right, top-to-bottom — better for multi-column
            raw_text = page.get_text("text", sort=True)

            is_ocr = False
            if _is_mostly_empty(raw_text):
                # Fallback: render page to image and run OCR
                try:
                    img = _page_to_image(page)
                    raw_text = pytesseract.image_to_string(img)
                    is_ocr = True
                except Exception:
                    raw_text = ""  # OCR unavailable; keep empty

            cleaned = raw_text.strip()
            pages.append(
                {
                    "paper_id": paper_id,
                    "page": page_num,
                    "text": cleaned,
                    "is_ocr": is_ocr,
                    "word_count": len(cleaned.split()),
                }
            )

    return pages


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 3:
        print("Usage: python pdf_parser.py <pdf_path> <paper_id>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1], sys.argv[2])
    for p in result:
        print(f"Page {p['page']:3d} | words: {p['word_count']:5d} | ocr: {p['is_ocr']}")
