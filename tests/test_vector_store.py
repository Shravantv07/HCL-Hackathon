
import pytest

from src.retrieval.vector_store import VectorStore


class FakeEmbeddingGenerator:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class FakeCollection:

    def get(self, include=None):
        return {
            "ids": ["paper1__text__0", "paper1__text__1"],
            "documents": [
                "Transformers use attention.",
                "Multimodal models process images and text."
            ],
            "metadatas": [
                {"paper_id": "paper1", "page": 1},
                {"paper_id": "paper1", "page": 2}
            ]
        }

    def query(self, **kwargs):
        self.last_query = kwargs

        return {
            "ids": [["paper1__text__0"]],
            "documents": [["Transformers use attention."]],
            "metadatas": [[{"paper_id": "paper1", "page": 1}]],
            "distances": [[0.15]]
        }


@pytest.fixture
def vector_store():
    return VectorStore(
        embedding_generator=FakeEmbeddingGenerator(),
        collection=FakeCollection()
    )


def test_get_all_chunks(vector_store):
    chunks = vector_store.get_all_chunks()

    assert len(chunks) == 2
    assert chunks[0]["id"] == "paper1__text__0"
    assert chunks[0]["text"] == "Transformers use attention."
    assert chunks[0]["page"] == 1


def test_semantic_search(vector_store):
    results = vector_store.semantic_search(
        "What are transformers?",
        top_k=3
    )

    assert len(results) == 1
    assert results[0]["id"] == "paper1__text__0"
    assert results[0]["score"] == 0.15


def test_paper_filter(vector_store):
    vector_store.semantic_search(
        "What are transformers?",
        paper_ids=["paper1"]
    )

    assert vector_store.collection.last_query["where"] == {
        "paper_id": "paper1"
    }


def test_empty_query(vector_store):
    with pytest.raises(ValueError):
        vector_store.semantic_search("")


def test_invalid_top_k(vector_store):
    with pytest.raises(ValueError):
        vector_store.semantic_search("transformers", top_k=0)
