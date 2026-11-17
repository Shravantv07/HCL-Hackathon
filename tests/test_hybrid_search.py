
from src.retrieval.hybrid_search import HybridSearcher


def test_rrf_combination():

    dense = [
        {"id": "A", "text": "Chunk A"},
        {"id": "B", "text": "Chunk B"},
        {"id": "C", "text": "Chunk C"}
    ]

    sparse = [
        {"id": "B", "text": "Chunk B"},
        {"id": "A", "text": "Chunk A"},
        {"id": "D", "text": "Chunk D"}
    ]

    searcher = HybridSearcher()

    results = searcher.search(
        dense,
        sparse,
        top_k=4
    )

    # A and B appear in both rankings.
    assert len(results) == 4
    assert results[0]["id"] == "A"
    assert results[1]["id"] == "B"

    # Both must have contributions from both rankings.
    assert results[0]["rrf_score"] > 0
    assert results[1]["rrf_score"] > 0

    assert results[0]["rrf_score"] == (
        1 / 61 + 1 / 62
    )

    print("RRF combination successful!")


def test_top_k():

    searcher = HybridSearcher()

    results = searcher.search(
        [{"id": "A"}, {"id": "B"}],
        [{"id": "C"}],
        top_k=2
    )

    assert len(results) == 2


def test_empty_results():

    searcher = HybridSearcher()

    results = searcher.search([], [])

    assert results == []


if __name__ == "__main__":
    test_rrf_combination()
    test_top_k()
    test_empty_results()
