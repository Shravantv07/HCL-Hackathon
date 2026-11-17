
from src.retrieval.embeddings import EmbeddingGenerator


def test_query_embedding():

    generator = EmbeddingGenerator()

    query = "How do multimodal LLMs process visual information?"

    embedding = generator.embed_query(query)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, (int, float)) for value in embedding)

    print("Embedding generated successfully!")
    print("Dimensions:", len(embedding))
    print("First five values:", embedding[:5])


if __name__ == "__main__":
    test_query_embedding()
