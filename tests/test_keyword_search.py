
from src.retrieval.keyword_search import KeywordSearcher


def sample_chunks():
    return [
        {
            "id": "chunk_1",
            "text": "Multimodal large language models process images and text together.",
            "paper_id": "paper_1",
            "page": 2
        },
        {
            "id": "chunk_2",
            "text": "Vision transformers divide images into patches for visual processing.",
            "paper_id": "paper_2",
            "page": 4
        },
        {
            "id": "chunk_3",
            "text": "Retrieval augmented generation uses external documents to improve answers.",
            "paper_id": "paper_3",
            "page": 7
        }
    ]


def test_keyword_search():
    searcher = KeywordSearcher(sample_chunks())

    results = searcher.search(
        "How do multimodal models process images?",
        top_k=2
    )

    assert len(results) <= 2
    assert len(results) > 0
    assert results[0]["id"] == "chunk_1"
    assert "score" in results[0]

    print("Keyword search successful!")
    for result in results:
        print(result["id"], result["score"])


def test_paper_filter():
    searcher = KeywordSearcher(sample_chunks())

    results = searcher.search(
        "images visual processing",
        paper_ids=["paper_2"]
    )

    assert all(
        result["paper_id"] == "paper_2"
        for result in results
    )

    print("Paper filtering successful!")


def test_empty_query():
    searcher = KeywordSearcher(sample_chunks())

    try:
        searcher.search("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_keyword_search()
    test_paper_filter()
    test_empty_query()
