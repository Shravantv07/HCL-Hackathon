from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.retrieval.embeddings import EmbeddingGenerator


def test_embedding_generator_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        EmbeddingGenerator()


def test_embedding_generator_empty_query(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with patch("src.retrieval.embeddings.genai.Client"):
        generator = EmbeddingGenerator()
    with pytest.raises(ValueError, match="empty"):
        generator.embed_query("   ")


def test_embedding_generator_embed_query_unit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    fake_response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
    )
    with patch("src.retrieval.embeddings.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.embed_content.return_value = fake_response
        generator = EmbeddingGenerator()
        embedding = generator.embed_query("  How do multimodal LLMs work?  ")

    assert embedding == [0.1, 0.2, 0.3]
    mock_client_cls.return_value.models.embed_content.assert_called_once()


@pytest.mark.integration
def test_query_embedding_live():
    generator = EmbeddingGenerator()
    query = "How do multimodal LLMs process visual information?"
    embedding = generator.embed_query(query)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, (int, float)) for value in embedding)
