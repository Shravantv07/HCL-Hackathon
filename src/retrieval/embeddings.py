
import os
from typing import List

from google import genai
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv(
            "EMBEDDING_MODEL",
            "gemini-embedding-2"
        )

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")

        self.client = genai.Client(api_key=self.api_key)

    def embed_query(self, query: str) -> List[float]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=query.strip()
        )

        return result.embeddings[0].values
