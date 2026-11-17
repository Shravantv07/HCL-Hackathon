"""Tests for centralized configuration loading, validation, and path resolution."""
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, get_settings, resolve_project_path
from src.retrieval.router import QueryRouter
from src.utils.security import validate_upload


def test_default_settings(monkeypatch):
    for key in (
        "GEMINI_API_KEY",
        "CHROMA_PATH",
        "CHROMA_COLLECTION",
        "FIGURES_PATH",
        "FIGURES_INDEX_PATH",
        "PAPERS_PATH",
        "EVAL_QUESTIONS_PATH",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "OCR_DPI",
        "OCR_MIN_CHARS",
        "MAX_FILE_SIZE_MB",
        "TOP_K",
        "RERANK_ENABLED",
        "RERANKER_MODEL",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = get_settings()
    assert cfg.gemini_api_key is None
    assert cfg.chroma_collection == "research_papers"
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 32
    assert cfg.ocr_dpi == 200
    assert cfg.ocr_min_chars == 50
    assert cfg.max_file_size_mb == 50
    assert cfg.top_k == 6
    assert cfg.rerank_enabled is True
    assert cfg.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.embedding_model == "gemini-embedding-2"
    assert cfg.llm_model == "gemini-3.8-pro"


def test_missing_required_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = get_settings()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        cfg.require_gemini_api_key()


def test_invalid_integer_settings(monkeypatch):
    monkeypatch.setenv("TOP_K", "not_an_int")
    with pytest.raises(ValueError, match="Invalid integer for environment variable TOP_K"):
        get_settings()

    monkeypatch.setenv("TOP_K", "0")
    with pytest.raises(ValueError, match="TOP_K must be >= 1"):
        get_settings()


def test_invalid_chunk_overlap(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        get_settings()


def test_boolean_parsing(monkeypatch):
    for false_val in ("false", "0", "no", "off", "FALSE"):
        monkeypatch.setenv("RERANK_ENABLED", false_val)
        assert get_settings().rerank_enabled is False

    for true_val in ("true", "1", "yes", "on", "TRUE"):
        monkeypatch.setenv("RERANK_ENABLED", true_val)
        assert get_settings().rerank_enabled is True

    monkeypatch.setenv("RERANK_ENABLED", "maybe")
    with pytest.raises(ValueError, match="Invalid boolean for environment variable RERANK_ENABLED"):
        get_settings()


def test_path_resolution_independent_of_cwd(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHROMA_PATH", "./data/chroma_index")
    cfg = get_settings()
    assert cfg.chroma_path == (PROJECT_ROOT / "data" / "chroma_index").resolve()

    abs_dir = str((tmp_path / "custom_chroma").resolve())
    assert resolve_project_path(abs_dir) == Path(abs_dir)


def test_modules_use_central_config(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TOP_K", "4")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")

    class RecordingRetriever:
        def __init__(self):
            self.last_call = None

        def retrieve(self, query, top_k=None, paper_ids=None, rerank=None):
            self.last_call = {"top_k": top_k, "rerank": rerank}
            return []

    rec = RecordingRetriever()
    router = QueryRouter(retriever=rec)
    router.route("Test query")
    assert rec.last_call == {"top_k": 4, "rerank": False}

    big_pdf = tmp_path / "big.pdf"
    big_pdf.write_bytes(b"%PDF-1.4" + b"0" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="Maximum is 1 MB"):
        validate_upload(str(big_pdf))
