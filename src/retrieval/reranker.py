
from typing import Any, Dict, List, Optional

from src.config import get_settings


class Reranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    """

    def __init__(
        self,
        model=None,
        model_name: Optional[str] = None
    ):
        # Model injection makes testing possible without downloading it.
        if model is None:
            from sentence_transformers import CrossEncoder

            resolved_model = model_name or get_settings().reranker_model
            self.model = CrossEncoder(resolved_model)
        else:
            self.model = model

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 6
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not chunks:
            return []

        pairs = [
            [query, chunk["text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        results = []

        for chunk, score in zip(chunks, scores):
            result = chunk.copy()
            result["rerank_score"] = float(score)
            results.append(result)

        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return results[:top_k]
