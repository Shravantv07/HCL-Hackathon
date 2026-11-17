
from typing import Dict, List, Optional

import chromadb

from src.config import get_settings
from src.retrieval.embeddings import EmbeddingGenerator


class VectorStore:
    """
    Handles semantic retrieval and chunk access through ChromaDB.
    """

    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        collection=None
    ):
        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )

        # Allow fake collections to be injected during testing.
        if collection is not None:
            self.collection = collection
        else:
            cfg = get_settings()
            client = chromadb.PersistentClient(path=str(cfg.chroma_path))

            # Connect to the existing collection.
            # Do not silently create an empty collection.
            self.collection = client.get_collection(
                name=cfg.chroma_collection
            )

    @staticmethod
    def _build_filter(
        paper_ids: Optional[List[str]]
    ) -> Optional[Dict]:

        if not paper_ids:
            return None

        if len(paper_ids) == 1:
            return {"paper_id": paper_ids[0]}

        return {"paper_id": {"$in": paper_ids}}

    def get_all_chunks(self) -> List[Dict]:
        """
        Fetch all indexed chunks for BM25 indexing.
        """

        data = self.collection.get(
            include=["documents", "metadatas"]
        )

        chunks = []

        ids = data.get("ids", [])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        for i, chunk_id in enumerate(ids):
            chunks.append({
                "id": chunk_id,
                "text": documents[i],
                **(metadatas[i] or {})
            })

        return chunks

    def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        paper_ids: Optional[List[str]] = None
    ) -> List[Dict]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is None:
            top_k = get_settings().top_k

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_generator.embed_query(query)

        n_results = top_k
        if hasattr(self.collection, "count"):
            total = self.collection.count()
            if total == 0:
                return []
            n_results = min(top_k, total)

        search_args = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        where_filter = self._build_filter(paper_ids)

        if where_filter is not None:
            search_args["where"] = where_filter

        data = self.collection.query(**search_args)

        results = []

        ids = data["ids"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]
        distances = data["distances"][0]

        for i, chunk_id in enumerate(ids):
            results.append({
                "id": chunk_id,
                "text": documents[i],
                **(metadatas[i] or {}),
                "score": float(distances[i])
            })

        return results
