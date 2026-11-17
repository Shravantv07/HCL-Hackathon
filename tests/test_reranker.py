
from src.retrieval.reranker import Reranker


class FakeCrossEncoder:

    def predict(self, pairs):
        # Scores correspond to the order of input chunks.
        return [0.2, 0.9, 0.5]


def test_reranking():

    chunks = [
        {"id": "A", "text": "Chunk about image processing"},
        {"id": "B", "text": "Chunk about multimodal vision encoders"},
        {"id": "C", "text": "Chunk about language models"}
    ]

    reranker = Reranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "How do multimodal models process images?",
        chunks,
        top_k=3
    )

    assert len(results) == 3

    # B has the highest fake relevance score.
    assert results[0]["id"] == "B"
    assert results[0]["rerank_score"] == 0.9

    print("Reranking successful!")


def test_top_k():

    reranker = Reranker(model=FakeCrossEncoder())

    chunks = [
        {"id": "A", "text": "Text A"},
        {"id": "B", "text": "Text B"},
        {"id": "C", "text": "Text C"}
    ]

    results = reranker.rerank(
        "test query",
        chunks,
        top_k=2
    )

    assert len(results) == 2


def test_empty_chunks():

    reranker = Reranker(model=FakeCrossEncoder())

    results = reranker.rerank(
        "test query",
        [],
        top_k=5
    )

    assert results == []


if __name__ == "__main__":
    test_reranking()
    test_top_k()
    test_empty_chunks()
