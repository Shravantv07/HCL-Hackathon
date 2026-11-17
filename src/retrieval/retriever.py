
from typing import List, Dict, Any, Optional

from src.config import get_settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_search import KeywordSearcher
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker


class Retriever:
    """
    Coordinates semantic search, keyword search,
    hybrid fusion, and optional reranking.
    """

    def __init__(
        self,
        vector_store=None,
        keyword_searcher=None,
        hybrid_searcher=None,
        reranker=None
    ):

        self.vector_store = vector_store or VectorStore()

        # Build BM25 from all chunks already stored in ChromaDB.
        self.keyword_searcher = (
            keyword_searcher
            or KeywordSearcher(self.vector_store.get_all_chunks())
        )

        self.hybrid_searcher = hybrid_searcher or HybridSearcher()

        # Load the actual cross-encoder only when reranking is requested.
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        paper_ids: Optional[List[str]] = None,
        rerank: Optional[bool] = None
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        cfg = get_settings()
        if top_k is None:
            top_k = cfg.top_k
        if rerank is None:
            rerank = cfg.rerank_enabled

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Retrieve more candidates than the final requested count.
        fetch_k = top_k * 4

        # 1. Dense retrieval.
        dense_results = self.vector_store.semantic_search(
            query=query,
            top_k=fetch_k,
            paper_ids=paper_ids
        )

        # 2. Sparse retrieval.
        sparse_results = self.keyword_searcher.search(
            query=query,
            top_k=fetch_k,
            paper_ids=paper_ids
        )

        # 3. Combine rankings using RRF.
        hybrid_results = self.hybrid_searcher.search(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=fetch_k
        )

        # 4. Optional cross-encoder reranking.
        if rerank and hybrid_results:

            if self.reranker is None:
                self.reranker = Reranker()

            final_results = self.reranker.rerank(
                query=query,
                chunks=hybrid_results,
                top_k=top_k
            )

        else:
            final_results = hybrid_results[:top_k]

        return final_results
