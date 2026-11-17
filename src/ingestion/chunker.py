"""
Splits extracted page text into overlapping token-aware chunks with metadata.

Chunk types:
  - "text"   : body paragraph chunks
  - "figure" : caption + nearby_text for a figure (from figure_extractor)
  - "table"  : detected table text + caption

Each chunk carries full metadata so retrieval results can be cited precisely.
"""

import re
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Simple token counter (word-based approximation; good enough for chunking)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count: words + punctuation splits."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(\d+\.?\s+[A-Z]|Abstract|Introduction|Background|Related Work|"
    r"Method(ology)?|Approach|Experiment|Result|Discussion|Conclusion|"
    r"Reference|Appendix)",
    re.IGNORECASE | re.MULTILINE,
)


def _detect_section(text: str) -> str:
    """Return the first heading-like line found in a text block, or ''."""
    match = _HEADING_RE.search(text)
    if match:
        return match.group(0).strip()[:80]
    return ""


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(r"^table\s*\d+", re.IGNORECASE)


def _looks_like_table(text: str) -> bool:
    return bool(_TABLE_RE.search(text))


# ---------------------------------------------------------------------------
# Core chunker
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: List[Dict[str, Any]],
    paper_title: str = "",
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Split page-level text into overlapping chunks.

    Args:
        pages:       Output of pdf_parser.parse_pdf()
        paper_title: Human-readable paper title for metadata
        chunk_size:  Target chunk size in tokens
        overlap:     Token overlap between consecutive chunks

    Returns a list of chunk dicts:
        {
            paper_id: str,
            title: str,
            page: int,
            section: str,
            chunk_type: "text" | "table",
            chunk_index: int,
            text: str,
            figure_ref: None,       # always None for text/table chunks
            token_count: int,
        }
    """
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        paper_id = page["paper_id"]
        page_num = page["page"]
        text = page["text"]

        if not text.strip():
            continue

        section = _detect_section(text)
        chunk_type = "table" if _looks_like_table(text) else "text"

        words = text.split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            window = words[start:end]
            chunk_text = " ".join(window)

            chunks.append(
                {
                    "paper_id": paper_id,
                    "title": paper_title,
                    "page": page_num,
                    "section": section,
                    "chunk_type": chunk_type,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "figure_ref": None,
                    "token_count": _count_tokens(chunk_text),
                }
            )
            chunk_index += 1

            if end == len(words):
                break
            start = end - overlap  # slide back by overlap tokens

    return chunks


def chunk_figures(figures: List[Dict[str, Any]], paper_title: str = "") -> List[Dict[str, Any]]:
    """
    Create one chunk per extracted figure using its caption + nearby text.

    Args:
        figures:     Output of figure_extractor.extract_figures()
        paper_title: Human-readable paper title for metadata

    Returns a list of figure chunk dicts:
        {
            paper_id, title, page, section, chunk_type="figure",
            chunk_index, text, figure_ref, image_path, token_count
        }
    """
    chunks: List[Dict[str, Any]] = []

    for i, fig in enumerate(figures):
        combined = f"{fig['caption']} {fig['nearby_text']}".strip()
        if not combined:
            combined = f"Figure on page {fig['page']} of {paper_title}"

        chunks.append(
            {
                "paper_id": fig["paper_id"],
                "title": paper_title,
                "page": fig["page"],
                "section": "",
                "chunk_type": "figure",
                "chunk_index": i,
                "text": combined,
                "figure_ref": fig["fig_id"],
                "image_path": fig["image_path"],
                "token_count": _count_tokens(combined),
            }
        )

    return chunks


if __name__ == "__main__":
    import sys

    # Quick smoke-test: pipe in a couple of fake pages
    fake_pages = [
        {
            "paper_id": "p01",
            "page": 1,
            "text": "Abstract This paper presents a novel approach to retrieval-augmented generation. " * 20,
            "is_ocr": False,
            "word_count": 100,
        },
        {
            "paper_id": "p01",
            "page": 2,
            "text": "Table 1 shows accuracy results across all benchmarks. " * 10,
            "is_ocr": False,
            "word_count": 50,
        },
    ]
    text_chunks = chunk_pages(fake_pages, paper_title="Test Paper", chunk_size=50, overlap=10)
    print(f"Text chunks produced: {len(text_chunks)}")
    for c in text_chunks:
        print(f"  [{c['chunk_type']}] p.{c['page']} | tokens:{c['token_count']} | {c['text'][:60]!r}")
