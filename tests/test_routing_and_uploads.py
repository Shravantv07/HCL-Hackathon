"""
Deterministic tests for request routing, paper-name detection and upload helpers.

No Gemini call is made: routing tests replace the chain functions with recorders
(they check WHICH chain is called with WHICH papers, not what a model would say).
"""
import pymupdf
import pytest

from backend import dispatch
from src.chains import comparison_chain, sources
from src.chains.visual_chain import _named_figure
from src.ingestion.uploads import text_to_pdf

TITLES = {
    "p01": "LLaVA-OneVision-1.5: Efficient Vision-Language Models via Progressive Training",
    "p04": "ChartMind: A Comprehensive Benchmark for Complex Chart Understanding",
    "p06": "LongTableBench: Towards Long-Context Understanding for Tables",
    "p09": "Table as a Modality for Large Language Models",
}


@pytest.fixture
def catalogue(monkeypatch):
    monkeypatch.setattr(sources, "_catalogue_titles", lambda: TITLES)
    monkeypatch.setattr(sources, "_uploaded_titles", lambda: {"u123": "MultiChartQA: Multi-Chart Benchmark"})


# --- paper-name detection ----------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("Compare ChartMind and LongTableBench", ["p04", "p06"]),
    ("How does LLaVA-OneVision differ from ChartMind?", ["p01", "p04"]),       # version suffix dropped
    ("p09 versus p04", ["p09", "p04"]),                                        # ids, in mention order
    ("Table as a Modality for Large Language Models vs MultiChartQA", ["p09", "u123"]),  # uploads too
    ("What is retrieval-augmented generation?", []),
])
def test_find_mentioned_papers(catalogue, question, expected):
    assert sources.find_mentioned_papers(question) == expected


# --- routing ---------------------------------------------------------------------------

@pytest.fixture
def recorder(monkeypatch, catalogue):
    calls = []

    def rec(name, result):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result
        return fn

    ok = {"status": "ok", "answer": "x", "citations": [], "warnings": []}
    monkeypatch.setattr(dispatch, "resolve_followup", lambda q, h: q)  # no LLM rewrite
    monkeypatch.setattr(dispatch.qa_chain, "answer", rec("qa", ok))
    monkeypatch.setattr(dispatch.comparison_chain, "compare",
                        rec("compare", {**ok, "papers": [], "table": [], "citations": []}))
    monkeypatch.setattr(dispatch.synthesis_chain, "synthesize",
                        rec("synthesize", {**ok, "synthesis": {"text": "s"}, "papers": [], "disclaimer": "d"}))
    monkeypatch.setattr(dispatch.visual_chain, "answer_visual", rec("visual", {**ok, "source": {}}))
    monkeypatch.setattr(dispatch.visual_chain, "answer_about_figure", rec("image", {**ok, "source": {}}))
    monkeypatch.setattr(dispatch, "_catalogue_ids", lambda: frozenset(TITLES))
    return calls


def test_comparison_uses_papers_named_in_question(recorder):
    out = dispatch.route_question("Compare ChartMind and LongTableBench")
    assert out["route"] == "comparison"
    assert recorder[0][0] == "compare" and recorder[0][1][0] == ["p04", "p06"]


def test_comparison_with_one_paper_falls_back_to_qa(recorder):
    out = dispatch.route_question("Compare ChartMind with other work")
    assert out["route"] == "factual" and recorder[0][0] == "qa"
    assert any("at least two papers" in w for w in out["warnings"])


def test_selected_papers_take_priority_for_comparison(recorder):
    dispatch.route_question("Compare these papers", paper_ids=["p01", "p09"])
    assert recorder[0][0] == "compare" and recorder[0][1][0] == ["p01", "p09"]


def test_synthesis_defaults_to_catalogue(recorder):
    out = dispatch.route_question("What research gaps exist across all papers?")
    assert out["route"] == "synthesis"
    assert recorder[0][0] == "synthesize" and recorder[0][1][0] == sorted(TITLES)


def test_attached_image_is_analysed_directly(recorder, tmp_path):
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG fake bytes are fine: the chain is replaced by a recorder")
    out = dispatch.route_question("What does this show?", image_paths=[img])
    assert out["route"] == "visual" and recorder[0][0] == "image"


def test_factual_question_goes_to_qa_with_scope(recorder):
    dispatch.route_question("What is the training pipeline?", paper_ids=["p01"])
    assert recorder[0][0] == "qa" and recorder[0][2]["paper_ids"] == ["p01"]


def test_error_result_is_shown_as_warning_answer(recorder, monkeypatch):
    monkeypatch.setattr(dispatch.qa_chain, "answer",
                        lambda *a, **k: {"status": "error", "error": "Retrieval failed: boom", "citations": []})
    out = dispatch.route_question("Anything?")
    assert out["status"] == "error" and out["answer"].startswith("⚠️ Retrieval failed")


# --- small helpers ---------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("(AI-generated suggestion, not established fact) No video benchmarks [S2].", "No video benchmarks [S2]."),
    ("AI-generated suggestion: no multilingual data.", "no multilingual data."),
    ("Neither paper covers video.", "Neither paper covers video."),
])
def test_strip_gap_label(text, expected):
    assert comparison_chain.strip_gap_label(text) == expected


def test_named_figure_is_preferred():
    candidates = [{"caption": "Figure 2 Overall architecture"}, {"caption": "Figure 1 Performance"},
                  {"caption": "Table 1: Results"}]
    assert _named_figure("What does Figure 1 show?", candidates) is candidates[1]
    assert _named_figure("Explain table 1", candidates) is candidates[2]
    assert _named_figure("What is shown?", candidates) is None
    assert _named_figure("Figure 7?", candidates) is None


def test_text_upload_becomes_multipage_pdf():
    pdf = text_to_pdf("Line one\n\n" + "multimodal models " * 3000, "notes")
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    assert doc.page_count > 1
    assert "Line one" in doc[0].get_text() and "notes" in doc[0].get_text()
