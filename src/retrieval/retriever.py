"""
Hybrid retriever: dense (FAISS cosine) + sparse (BM25) merged via Reciprocal Rank Fusion.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

import faiss
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_FILE    = os.path.join(FAISS_INDEX_DIR, "metadata.json")


# ---------------------------------------------------------------------------
# Load index + metadata (cached in module-level globals after first load)
# ---------------------------------------------------------------------------

_faiss_index: faiss.Index = None
_metadata: List[Dict[str, Any]] = None
_bm25: BM25Okapi = None
_corpus_tokens: List[List[str]] = None


def _load_assets():
    global _faiss_index, _metadata, _bm25, _corpus_tokens

    if _faiss_index is not None:
        return  # already loaded

    if not Path(FAISS_INDEX_FILE).exists():
        raise FileNotFoundError(
            "FAISS index not found. Run the indexer first:\n"
            "  python -m src.ingestion.indexer --catalogue paper_catalogue.csv"
        )

    _faiss_index = faiss.read_index(FAISS_INDEX_FILE)

    with open(METADATA_FILE, encoding="utf-8") as f:
        _metadata = json.load(f)

    # Build BM25 corpus from chunk texts
    _corpus_tokens = [doc["text"].lower().split() for doc in _metadata]
    _bm25 = BM25Okapi(_corpus_tokens)


# ---------------------------------------------------------------------------
# Embedding query
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> np.ndarray:
    genai.configure(api_key=GEMINI_API_KEY)
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    vec = np.array(result["embedding"], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf(dense_ids: List[int], sparse_ids: List[int], k: int = 60) -> List[int]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns a re-ranked list of indices.
    """
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

    Each result is the original chunk dict with two extra keys:
        dense_score: float  (cosine similarity)
        rrf_score:   float  (reciprocal rank fusion score)
    """
    _load_assets()

    n_total = len(_metadata)
    if n_total == 0:
        return []

    fetch_k = min(top_k * 4, n_total)  # fetch more candidates before RRF

    # --- Dense retrieval ---
    query_vec = _embed_query(query)
    distances, dense_indices = _faiss_index.search(query_vec, fetch_k)
    dense_ids    = [int(i) for i in dense_indices[0] if i >= 0]
    dense_scores = {int(i): float(d) for i, d in zip(dense_indices[0], distances[0]) if i >= 0}

    # --- Sparse retrieval (BM25) ---
    query_tokens  = query.lower().split()
    bm25_scores   = _bm25.get_scores(query_tokens)
    sparse_ids    = list(np.argsort(bm25_scores)[::-1][:fetch_k])

    # --- Reciprocal Rank Fusion ---
    merged = _rrf(dense_ids, sparse_ids)

    # --- Filter by paper_ids if requested ---
    if paper_ids:
        merged = [i for i in merged if _metadata[i].get("paper_id") in paper_ids]

    # --- Build result list ---
    results = []
    for idx in merged[:top_k]:
        chunk = dict(_metadata[idx])
        chunk["dense_score"] = dense_scores.get(idx, 0.0)
        chunk["rrf_score"]   = 1.0 / (60 + merged.index(idx) + 1)
        results.append(chunk)

    return results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is retrieval augmented generation?"
    hits = retrieve(q, top_k=5)
    print(f"\nTop {len(hits)} results for: {q!r}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h['paper_id']} p.{h['page']}] score={h['rrf_score']:.4f}")
        print(f"   {h['text'][:120]}...\n")
