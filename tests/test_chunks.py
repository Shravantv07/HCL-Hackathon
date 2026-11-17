"""Integration tests for PDF parsing, figure extraction, and chunking."""
import io
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image

from src.ingestion.chunker import chunk_figures, chunk_pages
from src.ingestion.figure_extractor import extract_figures
from src.ingestion.pdf_parser import parse_pdf


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal multi-page PDF with text and an embedded image in tmp_path."""
    pdf_path = tmp_path / "sample_paper.pdf"
    doc = fitz.open()

    # Page 1: body text (use insert_textbox so text wraps onto multiple lines)
    page1 = doc.new_page()
    page1.insert_textbox(
        fitz.Rect(72, 72, 520, 720),
        "Introduction\n"
        + ("Multimodal large language models process text and images. " * 25),
    )

    # Page 2: embedded raster image (>= 80x80) + caption below it
    page2 = doc.new_page()
    img = Image.new("RGB", (120, 120), color=(30, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page2.insert_image(fitz.Rect(72, 72, 192, 192), stream=buf.getvalue())
    page2.insert_text((72, 210), "Figure 1: Sample architecture diagram.")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_parse_and_chunk_pages(sample_pdf: Path):
    pages = parse_pdf(str(sample_pdf), "p_test")
    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[0]["paper_id"] == "p_test"
    assert pages[0]["word_count"] > 0

    chunks = chunk_pages(pages, paper_title="Sample Paper", chunk_size=50, overlap=10)
    assert len(chunks) >= 2
    assert all(c["paper_id"] == "p_test" for c in chunks)
    assert all(c["title"] == "Sample Paper" for c in chunks)
    assert chunks[0]["page"] == 1


def test_extract_and_chunk_figures(sample_pdf: Path, tmp_path: Path):
    figures_dir = tmp_path / "figures"
    figs = extract_figures(str(sample_pdf), "p_test", str(figures_dir))

    assert len(figs) == 1
    fig = figs[0]
    assert fig["paper_id"] == "p_test"
    assert fig["page"] == 2
    assert fig["fig_id"] == "p_test_p2_fig1"
    assert Path(fig["image_path"]).exists()
    assert "Figure 1" in fig["caption"]

    fig_chunks = chunk_figures(figs, paper_title="Sample Paper")
    assert len(fig_chunks) == 1
    assert fig_chunks[0]["chunk_type"] == "figure"
    assert fig_chunks[0]["figure_ref"] == "p_test_p2_fig1"
    assert "Figure 1" in fig_chunks[0]["text"]
