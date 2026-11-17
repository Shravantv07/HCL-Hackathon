"""
Live end-to-end tests against the real index, retriever and Gemini.

Skipped unless RUN_LIVE_TESTS=1 (they cost API calls and need a built index):
    RUN_LIVE_TESTS=1 pytest -m integration tests/test_live_pipeline.py
"""
import os

import pytest

from src.config import get_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run live tests"),
    pytest.mark.skipif(not get_settings().chroma_path.exists(), reason="index not built"),
]


def test_qa_answers_from_the_right_paper():
    from backend.dispatch import route_question
    out = route_question("What training pipeline does LLaVA-OneVision-1.5 adopt?")
    assert out["status"] == "ok" and out["route"] == "factual"
    assert "three-stage" in out["answer"].lower()
    assert "p01" in {s["paper_id"] for s in out["sources"]}


def test_unanswerable_question_is_not_answered_from_thin_air():
    from backend.dispatch import route_question
    out = route_question("What word error rate does M3DocVQA report for speech recognition?")
    text = out["answer"].lower()
    assert out["status"] == "insufficient_evidence" or "not" in text


def test_comparison_attributes_cells_to_their_own_paper():
    from src.chains.comparison_chain import compare_papers
    out = compare_papers("p04", "p06", "results")
    assert out["status"] == "ok"
    for row in out["table"]:
        for cell in row["cells"]:
            assert all(c["paper_id"] == cell["paper_id"] for c in cell["citations"])


def test_text_upload_is_indexed_searchable_and_deletable():
    from backend.dispatch import route_question
    from src.chains.sources import refresh_retriever
    from src.ingestion.uploads import delete_upload, ingest_text_upload

    rec = ingest_text_upload(b"TEST FILE - not research. The codeword is CEDAR-OWL-31.", "live_test_note.md")
    try:
        refresh_retriever()
        out = route_question("What is the codeword?", paper_ids=[rec["paper_id"]])
        assert "CEDAR-OWL-31" in out["answer"]
    finally:
        delete_upload(rec["paper_id"])
        refresh_retriever()
