
import re
from rank_bm25 import BM25Okapi


class BM25Search:

    def __init__(self, chunks):
        """
        chunks: List of dictionaries containing:
            id
            document
            metadata
        """

        self.chunks = chunks

        # Tokenize every document
        self.tokenized_corpus = [
            self._tokenize(chunk["document"])
            for chunk in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text):
        """
        Tokenize text while preserving technical terms
        such as GPT-4, ViT-L/14 and CLIP.
        """
        return re.findall(
            r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*",
            text.lower()
        )

    def search(self, query, top_k=10):

        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-scoring documents
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):

            results.append({
                "id": self.chunks[idx]["id"],
                "document": self.chunks[idx]["document"],
                "metadata": self.chunks[idx]["metadata"],
                "bm25_score": float(scores[idx]),
                "rank": rank
            })

        return results
