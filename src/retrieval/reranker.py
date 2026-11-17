
from typing import List, Dict, Any


class Reranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    """

    def __init__(
        self,
        model=None,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        # Model injection makes testing possible without downloading it.
        if model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
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
