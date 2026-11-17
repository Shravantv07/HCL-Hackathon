
import re
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi


class KeywordSearcher:
    """
    Performs keyword-based retrieval using BM25.
    """

    def __init__(self, chunks: List[Dict]):
        """
        Args:
            chunks: List of dictionaries containing:
                    id, text, and metadata such as paper_id and page.
        """

        self.chunks = chunks

        # Tokenize every chunk and build the BM25 index.
        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = (
            BM25Okapi(self.tokenized_corpus)
            if self.tokenized_corpus
            else None
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Convert text into lowercase tokens.
        Preserves technical terms containing hyphens,
        underscores, or slashes.
        """

        return re.findall(
            r"[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)*",
            text.lower()
        )

    def search(
        self,
        query: str,
        top_k: int = 6,
        paper_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks using BM25.

        Args:
            query: User's question.
            top_k: Maximum number of results.
            paper_ids: Optional list of paper IDs to restrict search.

        Returns:
            Ranked list of chunks with their BM25 scores.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        results = []

        for index, score in enumerate(scores):

            chunk = self.chunks[index]

            # Apply paper filtering consistently.
            if paper_ids is not None:
                if chunk.get("paper_id") not in paper_ids:
                    continue

            # Ignore chunks with no keyword relevance.
            if score <= 0:
                continue

            result = chunk.copy()
            result["score"] = float(score)

            results.append(result)

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]
