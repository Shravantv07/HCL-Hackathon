"""
Centralized configuration loader for ResearchIQ.

Loads environment variables from `<project_root>/.env`, parses typed values
with clear validation errors, and resolves relative paths against the project
root so the application works from any working directory.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _get_str(name: str, default: str) -> str:
    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    return val.strip()


def _get_int(name: str, default: int, min_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer for environment variable {name}: {raw!r}"
            ) from exc

    if min_value is not None and value < min_value:
        raise ValueError(
            f"Environment variable {name} must be >= {min_value}, got {value}."
        )
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean for environment variable {name}: {raw!r}. "
        "Expected one of true/false, 1/0, yes/no, on/off."
    )


def resolve_project_path(raw_path: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if it is not already absolute."""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    # API & Models
    gemini_api_key: Optional[str]
    embedding_model: str
    llm_model: str

    # Paths
    chroma_path: Path
    figures_path: Path
    figures_index_path: Path
    papers_path: Path
    eval_questions_path: Path

    # Ingestion
    chunk_size: int
    chunk_overlap: int
    ocr_dpi: int
    ocr_min_chars: int

    # Retrieval & Reranking
    chroma_collection: str
    top_k: int
    rerank_enabled: bool
    reranker_model: str

    # Security / Upload
    max_file_size_mb: int

    def require_gemini_api_key(self) -> str:
        """Return GEMINI_API_KEY or raise a clear ValueError if missing."""
        key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key or not key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required but not set. "
                "Add it to your .env file or environment."
            )
        return key.strip()


def get_settings() -> Settings:
    """Build and validate a Settings instance from the current environment."""
    chunk_size = _get_int("CHUNK_SIZE", 256, min_value=1)
    chunk_overlap = _get_int("CHUNK_OVERLAP", 32, min_value=0)
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be less than CHUNK_SIZE ({chunk_size})."
        )

    raw_key = os.getenv("GEMINI_API_KEY")
    api_key = raw_key.strip() if raw_key and raw_key.strip() else None

    return Settings(
        gemini_api_key=api_key,
        embedding_model=_get_str("EMBEDDING_MODEL", "gemini-embedding-2"),
        llm_model=_get_str("LLM_MODEL", "gemini-3.8-pro"),
        chroma_path=resolve_project_path(_get_str("CHROMA_PATH", "./data/chroma_index")),
        figures_path=resolve_project_path(_get_str("FIGURES_PATH", "./data/figures")),
        figures_index_path=resolve_project_path(
            _get_str("FIGURES_INDEX_PATH", "./data/figures_index.json")
        ),
        papers_path=resolve_project_path(_get_str("PAPERS_PATH", "./data/papers")),
        eval_questions_path=resolve_project_path(
            _get_str("EVAL_QUESTIONS_PATH", "./evaluation/eval_questions.csv")
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_dpi=_get_int("OCR_DPI", 200, min_value=72),
        ocr_min_chars=_get_int("OCR_MIN_CHARS", 50, min_value=0),
        chroma_collection=_get_str("CHROMA_COLLECTION", "research_papers"),
        top_k=_get_int("TOP_K", 6, min_value=1),
        rerank_enabled=_get_bool("RERANK_ENABLED", True),
        reranker_model=_get_str(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", 50, min_value=1),
    )


settings = get_settings()
