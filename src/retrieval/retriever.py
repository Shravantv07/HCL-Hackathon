"""
Hybrid retriever: dense (ChromaDB cosine) + sparse (BM25) merged via Reciprocal Rank Fusion.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH", "./data/chroma_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "research_papers"


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_collection = None
_all_chunks: List[Dict[str, Any]] = None
_chunk_positions: Dict[str, int] = None
_bm25: BM25Okapi = None


def _get_embedding(text: str) -> List[float]:
    genai.configure(api_key=GEMINI_API_KEY)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


def _load_assets():
    global _collection, _all_chunks, _chunk_positions, _bm25

    if _collection is not None:
        return

    genai.configure(api_key=GEMINI_API_KEY)

    class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=input,
                task_type="retrieval_document",
            )
            return result["embedding"] if isinstance(input, str) else result["embedding"]

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    # Load all chunks for BM25, retaining their Chroma IDs
    all_data = _collection.get(include=["documents", "metadatas", "ids"])
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
