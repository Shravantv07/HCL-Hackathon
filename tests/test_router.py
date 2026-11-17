"""Unit and integration tests for src/retrieval/router.py."""
from unittest.mock import patch

import pytest

from src.retrieval.retriever import Retriever
from src.retrieval.router import QueryRouter, classify_query


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self._results = results if results is not None else []
        self._error = error
        self.calls = []

    def retrieve(self, query, top_k=6, paper_ids=None, rerank=True):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "paper_ids": paper_ids,
                "rerank": rerank,
            }
        )
        if self._error is not None:
            raise self._error
        if paper_ids:
            return [r for r in self._results if r.get("paper_id") in paper_ids][:top_k]
        return self._results[:top_k]


def _sample_chunks():
    return [
        {
            "id": "p01__text__0",
            "paper_id": "p01",
            "title": "LLaVA-OneVision-1.5",
            "page": 3,
            "chunk_type": "text",
            "text": "Progressive training improves vision-language alignment.",
            "rerank_score": 0.91,
        },
        {
            "id": "p02__figure__0",
            "paper_id": "p02",
            "title": "VisuLogic",
            "page": 5,
            "chunk_type": "figure",
            "figure_ref": "p02_p5_fig1",
            "image_path": "data/figures/p02_p5_fig1.png",
            "text": "Figure 1: Multi-step visual reasoning pipeline.",
            "rerank_score": 0.85,
        },
    ]


def test_factual_query_routing():
    fake = FakeRetriever(results=_sample_chunks())
    router = QueryRouter(retriever=fake, default_top_k=6)

    out = router.route("What is progressive training in LLaVA-OneVision?")

    assert out["route"] == "factual"
    assert out["query"] == "What is progressive training in LLaVA-OneVision?"
    assert fake.calls[0]["top_k"] == 6
    assert out["metadata"]["result_count"] == 2
    assert out["metadata"]["warnings"] == []


def test_comparison_and_synthesis_query_routing():
    fake = FakeRetriever(results=_sample_chunks())
    router = QueryRouter(retriever=fake, default_top_k=6)

    comp = router.route("Compare LLaVA-OneVision versus VisuLogic on reasoning")
    assert comp["route"] == "comparison"
    assert fake.calls[0]["top_k"] == 8
    assert set(comp["paper_ids"]) == {"p01", "p02"}

    synth = router.route("Synthesize common themes and research gaps across all papers")
    assert synth["route"] == "synthesis"
    assert fake.calls[1]["top_k"] == 10


def test_visual_query_routing():
    fake = FakeRetriever(results=_sample_chunks())
    router = QueryRouter(retriever=fake)

    out = router.route("What does Figure 1 show about the reasoning pipeline?")

    assert out["route"] == "visual"
    assert out["metadata"]["has_visual_candidates"] is True
    assert any("Visual understanding is not implemented" in w for w in out["metadata"]["warnings"])


def test_paper_specific_filtering():
    fake = FakeRetriever(results=_sample_chunks())
    router = QueryRouter(retriever=fake)

    out = router.route("Explain the training method", paper_ids=["p01"])

    assert fake.calls[0]["paper_ids"] == ["p01"]
    assert out["requested_paper_ids"] == ["p01"]
    assert out["paper_ids"] == ["p01"]
    assert len(out["results"]) == 1
    assert out["results"][0]["paper_id"] == "p01"


def test_empty_query_handling():
    router = QueryRouter(retriever=FakeRetriever())

    with pytest.raises(ValueError, match="empty"):
        classify_query("   ")

    with pytest.raises(ValueError, match="empty"):
        router.route("")


def test_no_retrieval_results():
    router = QueryRouter(retriever=FakeRetriever(results=[]))

    out = router.route("Unanswerable question about quantum gravity")

    assert out["results"] == []
    assert out["paper_ids"] == []
    assert out["metadata"]["result_count"] == 0
    assert any("No matching evidence" in w for w in out["metadata"]["warnings"])


def test_invalid_route_override():
    router = QueryRouter(retriever=FakeRetriever())

    with pytest.raises(ValueError, match="Unsupported route"):
        router.route("Valid query", route_override="unknown_mode")


def test_retriever_unavailable_or_failure():
    # Case 1: Lazy Retriever() initialization fails (e.g. missing ChromaDB)
    with patch("src.retrieval.router.Retriever", side_effect=RuntimeError("No ChromaDB")):
        router = QueryRouter(retriever=None)
        with pytest.raises(RuntimeError, match="Retriever unavailable"):
            router.route("Test query")

    # Case 2: Injected retriever raises unexpected error during retrieve()
    failing = FakeRetriever(error=OSError("disk error"))
    router2 = QueryRouter(retriever=failing)
    with pytest.raises(RuntimeError, match="Retrieval failed"):
        router2.route("Test query")


def test_metadata_and_citations_preserved():
    fake = FakeRetriever(results=_sample_chunks())
    router = QueryRouter(retriever=fake)

    out = router.route("Show Figure 1")
    r0, r1 = out["results"]

    assert r0["paper_id"] == "p01"
    assert r0["page"] == 3
    assert r0["citation"] == "[LLaVA-OneVision-1.5, p.3]"

    assert r1["paper_id"] == "p02"
    assert r1["page"] == 5
    assert r1["figure_ref"] == "p02_p5_fig1"
    assert r1["image_path"] == "data/figures/p02_p5_fig1.png"
    assert r1["citation"] == "[VisuLogic, p02_p5_fig1, p.5]"


def test_integration_with_real_retriever():
    chunks = _sample_chunks()

    class FakeVS:
        def get_all_chunks(self):
            return chunks

        def semantic_search(self, query, top_k, paper_ids=None):
            if paper_ids:
                return [c for c in chunks if c["paper_id"] in paper_ids][:top_k]
            return chunks[:top_k]

    class FakeKS:
        def search(self, query, top_k, paper_ids=None):
            if paper_ids:
                return [c for c in chunks if c["paper_id"] in paper_ids][:top_k]
            return list(reversed(chunks))[:top_k]

    class FakeRR:
        def rerank(self, query, chunks, top_k):
            out = [i.copy() for i in chunks]
            for idx, item in enumerate(out):
                item["rerank_score"] = 1.0 - idx * 0.1
            return out[:top_k]

    real_retriever = Retriever(
        vector_store=FakeVS(),
        keyword_searcher=FakeKS(),
        reranker=FakeRR(),
    )
    router = QueryRouter(retriever=real_retriever, default_top_k=2)

    out = router.route("What does Figure 1 show?", paper_ids=["p02"], rerank=True)
    assert out["route"] == "visual"
    assert out["paper_ids"] == ["p02"]
    assert len(out["results"]) == 1
    assert out["results"][0]["id"] == "p02__figure__0"
    assert "rerank_score" in out["results"][0]
    assert "citation" in out["results"][0]
