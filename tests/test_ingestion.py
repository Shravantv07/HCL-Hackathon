"""Tests for the document processing pipeline."""
import pytest
from pathlib import Path
from src.ingestion.chunker import chunk_pages, chunk_figures, _count_tokens, _detect_section


# ---------------------------------------------------------------------------
# chunker tests
# ---------------------------------------------------------------------------

def _make_pages(text: str, paper_id: str = "p01", page: int = 1):
    return [{"paper_id": paper_id, "page": page, "text": text, "is_ocr": False, "word_count": len(text.split())}]


def test_chunk_pages_produces_chunks():
    pages = _make_pages("word " * 600, paper_id="p01")
    chunks = chunk_pages(pages, paper_title="Test Paper", chunk_size=100, overlap=20)
    assert len(chunks) > 1


def test_chunk_overlap():
    pages = _make_pages("word " * 300, paper_id="p01")
    chunks = chunk_pages(pages, chunk_size=100, overlap=20)
    # Each non-last chunk should be ~100 tokens; overlap means next starts 80 tokens later
    assert all(c["token_count"] <= 110 for c in chunks)


def test_chunk_metadata_fields():
    pages = _make_pages("Some text about retrieval augmented generation. " * 20, paper_id="p02")
    chunks = chunk_pages(pages, paper_title="RAG Paper")
    for c in chunks:
        assert "paper_id" in c
        assert "page" in c
        assert "chunk_type" in c
        assert "text" in c
        assert c["paper_id"] == "p02"
        assert c["title"] == "RAG Paper"


def test_chunk_type_table_detection():
    pages = _make_pages("Table 1 shows the results of our experiments across datasets.", paper_id="p03")
    chunks = chunk_pages(pages)
    assert any(c["chunk_type"] == "table" for c in chunks)


def test_empty_page_produces_no_chunks():
    pages = _make_pages("", paper_id="p04")
    chunks = chunk_pages(pages)
    assert chunks == []


def test_chunk_figures_produces_figure_chunks():
    fake_figures = [
        {
            "paper_id": "p01",
            "page": 3,
            "fig_id": "p01_p3_fig1",
            "image_path": "/data/figures/p01_p3_fig1.png",
            "caption": "Figure 1: Architecture overview.",
            "nearby_text": "The model consists of an encoder and decoder.",
            "width": 400,
            "height": 300,
        }
    ]
    chunks = chunk_figures(fake_figures, paper_title="My Paper")
    assert len(chunks) == 1
    c = chunks[0]
    assert c["chunk_type"] == "figure"
    assert c["figure_ref"] == "p01_p3_fig1"
    assert "Architecture overview" in c["text"]


def test_detect_section():
    text = "Introduction\nRetrieval-augmented generation (RAG) has become..."
    assert _detect_section(text).lower().startswith("intro")


def test_count_tokens():
    assert _count_tokens("hello world foo") == 3
    assert _count_tokens("") == 0
