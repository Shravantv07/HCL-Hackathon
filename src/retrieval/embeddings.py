from typing import List

from google import genai

from src.config import get_settings


class EmbeddingGenerator:

    def __init__(self):
        cfg = get_settings()
        self.api_key = cfg.require_gemini_api_key()
        self.model_name = cfg.embedding_model
        self.client = genai.Client(api_key=self.api_key)

    def embed_query(self, query: str) -> List[float]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query.strip()
        )

        return result.embeddings[0].values
