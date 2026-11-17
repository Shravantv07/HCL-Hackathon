
from src.retrieval.retriever import Retriever


class FakeVectorStore:

    def get_all_chunks(self):
        return [
            {
                "id": "A",
                "text": "Multimodal models process images.",
                "paper_id": "p01"
            },
            {
                "id": "B",
                "text": "Vision encoders process image patches.",
                "paper_id": "p02"
            }
        ]

    def semantic_search(self, query, top_k, paper_ids=None):
        return [
            {
                "id": "A",
                "text": "Multimodal models process images.",
                "paper_id": "p01"
            },
            {
                "id": "B",
                "text": "Vision encoders process image patches.",
                "paper_id": "p02"
            }
        ]


class FakeKeywordSearcher:

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, top_k, paper_ids=None):
        return [
            {
                "id": "B",
                "text": "Vision encoders process image patches.",
                "paper_id": "p02"
            },
            {
                "id": "A",
                "text": "Multimodal models process images.",
                "paper_id": "p01"
            }
        ]


class FakeReranker:

    def rerank(self, query, chunks, top_k):
        results = [chunk.copy() for chunk in chunks]

        results.reverse()

        for result in results:
            result["rerank_score"] = 0.9

        return results[:top_k]


def test_retriever_without_reranking():

    retriever = Retriever(
        vector_store=FakeVectorStore(),
        keyword_searcher=FakeKeywordSearcher([]),
    )

    results = retriever.retrieve(
        "How do multimodal models process images?",
        top_k=2,
        rerank=False
    )

    assert len(results) == 2
    assert all("rrf_score" in result for result in results)

    print("Hybrid retrieval pipeline successful!")


def test_retriever_with_reranking():

    retriever = Retriever(
        vector_store=FakeVectorStore(),
        keyword_searcher=FakeKeywordSearcher([]),
        reranker=FakeReranker()
    )

    results = retriever.retrieve(
        "How do multimodal models process images?",
        top_k=2,
        rerank=True
    )

    assert len(results) == 2
    assert "rerank_score" in results[0]

    print("Retrieval with reranking successful!")


def test_empty_query():

    retriever = Retriever(
        vector_store=FakeVectorStore(),
        keyword_searcher=FakeKeywordSearcher([])
    )

    try:
        retriever.retrieve("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_retriever_without_reranking()
    test_retriever_with_reranking()
    test_empty_query()
