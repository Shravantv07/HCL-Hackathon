
from typing import List, Dict, Any, Optional

<<<<<<< Updated upstream
import chromadb
from rank_bm25 import BM25Okapi
from google import genai
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./data/chroma_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "research_papers"
=======
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_search import KeywordSearcher
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
>>>>>>> Stashed changes


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

<<<<<<< Updated upstream
def _get_embedding(text: str) -> List[float]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return response.embeddings[0].values
=======
        # Build BM25 from all chunks already stored in ChromaDB.
        self.keyword_searcher = (
            keyword_searcher
            or KeywordSearcher(self.vector_store.get_all_chunks())
        )
>>>>>>> Stashed changes

        self.hybrid_searcher = hybrid_searcher or HybridSearcher()

        # Load the actual cross-encoder only when reranking is requested.
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        paper_ids: Optional[List[str]] = None,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:

<<<<<<< Updated upstream
    class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self):
            self._client = genai.Client(api_key=GEMINI_API_KEY)

        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            response = self._client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=input,
            )
            return [e.values for e in response.embeddings]
=======
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

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
>>>>>>> Stashed changes

        else:
            final_results = hybrid_results[:top_k]

<<<<<<< Updated upstream
    # Load all chunks for BM25, retaining their Chroma IDs
    # Note: "ids" is always returned by ChromaDB and must not be in include=[]
    all_data = _collection.get(include=["documents", "metadatas"])
    ids   = all_data.get("ids") or []
    docs  = all_data.get("documents") or []
    metas = all_data.get("metadatas") or []

    _all_chunks = [
        {"id": cid, **meta, "text": doc}
        for cid, doc, meta in zip(ids, docs, metas)
    ]

    # ID-to-position mapping for reliable lookup (avoids duplicate-text bugs)
    _chunk_positions = {chunk["id"]: idx for idx, chunk in enumerate(_all_chunks)}

    corpus_tokens = [c["text"].lower().split() for c in _all_chunks]
    _bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(dense_ids: List[int], sparse_ids: List[int], k: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


# ---------------------------------------------------------------------------
# Public retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 6,
    paper_ids: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return the top_k most relevant chunks for a query.

    Args:
        query:     Natural language question.
        top_k:     Number of results to return.
        paper_ids: Optional list of paper IDs to restrict search to.

    Each result is the chunk metadata dict with an extra 'rrf_score' key.
    """
    _load_assets()

    if not _all_chunks:
        return []

    fetch_k = min(top_k * 4, len(_all_chunks))

    # --- Dense retrieval via Chroma ---
    where_filter = {"paper_id": {"$in": paper_ids}} if paper_ids else None
    query_embedding = _get_embedding(query)

    chroma_results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chroma_ids   = chroma_results["ids"][0]
    chroma_metas = chroma_results["metadatas"][0]
    chroma_docs  = chroma_results["documents"][0]

    # Map Chroma result back to index positions using unique chunk ID
    dense_indices = []
    for cid in chroma_ids:
        pos = _chunk_positions.get(cid)
        if pos is not None:
            dense_indices.append(pos)

    # --- Sparse retrieval (BM25) ---
    sparse_indices = []
    if _bm25:
        query_tokens  = query.lower().split()
        bm25_scores   = _bm25.get_scores(query_tokens)
        # Respect paper_ids filter: zero out scores for chunks outside the filter
        if paper_ids:
            for i, chunk in enumerate(_all_chunks):
                if chunk.get("paper_id") not in paper_ids:
                    bm25_scores[i] = 0.0
        sparse_indices = list(np.argsort(bm25_scores)[::-1][:fetch_k])

    # --- RRF merge ---
    merged = _rrf(dense_indices, sparse_indices)

    # --- Build results ---
    results = []
    for rank, idx in enumerate(merged[:top_k]):
        chunk = dict(_all_chunks[idx])
        chunk["rrf_score"] = 1.0 / (60 + rank + 1)
        results.append(chunk)

    return results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"
    hits = retrieve(q, top_k=5)
    print(f"\nTop {len(hits)} results for: {q!r}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h.get('paper_id')} p.{h.get('page')}] score={h['rrf_score']:.4f}")
        print(f"   {h['text'][:120]}...\n")
=======
        return final_results
>>>>>>> Stashed changes
