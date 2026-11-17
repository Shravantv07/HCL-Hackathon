
from typing import Dict, List, Any


class HybridSearcher:
    """
    Combines dense and sparse retrieval results using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, rrf_k: int = 60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.rrf_k = rrf_k

    def search(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scores = {}
        chunks = {}

        rankings = [dense_results, sparse_results]

        for ranking in rankings:

            # Avoid counting the same chunk twice in one ranking.
            seen = set()

            for rank, chunk in enumerate(ranking, start=1):

                chunk_id = chunk["id"]

                if chunk_id in seen:
                    continue

                seen.add(chunk_id)

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

                # Preserve all original chunk information.
                chunks[chunk_id] = chunk.copy()

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True
        )

        results = []

        for chunk_id in ranked_ids[:top_k]:

            result = chunks[chunk_id]
            result["rrf_score"] = scores[chunk_id]

            results.append(result)

        return results
