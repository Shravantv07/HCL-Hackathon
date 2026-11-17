"""
PDF text extraction with page-level metadata.
Uses PyMuPDF as primary parser; falls back to OCR for image-only pages.
"""

import io
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz
import pytesseract
from PIL import Image

from src.config import get_settings

# Default install locations, used when tesseract is not on PATH (common on Windows).
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    """
    Locate the Tesseract binary (TESSERACT_CMD env var, PATH, then default install
    folders), point pytesseract at it, and report whether OCR can run.
    """
    candidates = [os.getenv("TESSERACT_CMD"), shutil.which("tesseract"), *_TESSERACT_CANDIDATES]
    for cmd in candidates:
        if cmd and Path(cmd).is_file():
            pytesseract.pytesseract.tesseract_cmd = cmd
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                continue
    return False


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
      2. If a page yields fewer than OCR_MIN_CHARS characters, fall back to Tesseract OCR.
         If OCR is unavailable or fails, a warning is printed and the page keeps
         whatever text PyMuPDF found (it is never silently blanked).

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
    ocr_missing_pages: List[int] = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Primary: PyMuPDF text extraction
            # sort=True reads left-to-right, top-to-bottom — better for multi-column
            raw_text = page.get_text("text", sort=True)

            is_ocr = False
            ocr_engine = None
            # A scanned / image-only page has (almost) no text layer at all. Pages with a little
            # real text (e.g. a full-page figure plus its caption) must not trigger Gemini OCR:
            # figures are described separately during indexing.
            no_text_layer = len(raw_text.strip()) < 5
            if _is_mostly_empty(raw_text):
                # Fallback 1: render page to image and run Tesseract OCR
                img = None
                if tesseract_available():
                    try:
                        img = _page_to_image(page)
                        ocr_text = pytesseract.image_to_string(img)
                        if not _is_mostly_empty(ocr_text):
                            raw_text, is_ocr, ocr_engine = ocr_text, True, "tesseract"
                    except Exception as e:
                        print(f"[pdf_parser] WARNING: OCR failed on {path.name} page {page_num}: {e}")

                # Fallback 2: Gemini vision, for scanned pages Tesseract could not read
                if not is_ocr and no_text_layer:
                    from src.ingestion.vision import ocr_page
                    try:
                        img = img or _page_to_image(page)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        gemini_text = ocr_page(buf.getvalue())
                    except Exception:
                        gemini_text = None
                    if gemini_text and not _is_mostly_empty(gemini_text):
                        raw_text, is_ocr, ocr_engine = gemini_text, True, "gemini"
                    elif not tesseract_available():
                        ocr_missing_pages.append(page_num)

            cleaned = raw_text.strip()
            pages.append(
                {
                    "paper_id": paper_id,
                    "page": page_num,
                    "text": cleaned,
                    "is_ocr": is_ocr,
                    "ocr_engine": ocr_engine,   # None | "tesseract" | "gemini"
                    "word_count": len(cleaned.split()),
                }
            )

    if ocr_missing_pages:
        print(f"[pdf_parser] WARNING: {path.name}: page(s) {ocr_missing_pages} have no text layer "
              "and need OCR, but Tesseract was not found. Install it or set TESSERACT_CMD.")
    return pages


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 3:
        print("Usage: python pdf_parser.py <pdf_path> <paper_id>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1], sys.argv[2])
    for p in result:
        print(f"Page {p['page']:3d} | words: {p['word_count']:5d} | ocr: {p['is_ocr']}")
