"""
Hybrid retriever: dense (Chroma) + sparse (BM25) merged via Reciprocal Rank Fusion.
"""
from typing import List, Dict, Any


def retrieve(query: str, top_k: int = 6, chroma_path: str = "./data/chroma_index") -> List[Dict[str, Any]]:
    """
    Return top_k chunks most relevant to query.
    Each result includes the original chunk dict plus a retrieval score.
    """
    # TODO: implement dense + BM25 + RRF
    raise NotImplementedError
