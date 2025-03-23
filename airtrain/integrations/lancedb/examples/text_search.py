import asyncio
import numpy as np
from typing import List
import tempfile
import os
from sentence_transformers import SentenceTransformer

from airtrain.integrations.lancedb.database_service import LanceDBService
from airtrain.integrations.lancedb.credentials import LanceDBCredentials
from airtrain.integrations.lancedb.schemas import (
    TableConfig,
    VectorData,
    SearchConfig,
    IndexConfig,
)


async def main():
    """Example demonstrating text embedding search with LanceDB"""

    # Initialize the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sample text data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast orange fox leaps across a sleepy canine",
        "The lazy dog sleeps in the sun",
        "A cat chases a mouse in the garden",
        "Birds fly in the blue sky",
        "Fish swim in the deep ocean",
        "The sun sets behind the mountains",
        "Stars twinkle in the night sky",
        "Rain falls on the green grass",
        "Snow covers the winter landscape",
    ]

    # Create embeddings for the texts
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Create a temporary directory for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize credentials with temporary directory
        credentials = LanceDBCredentials(
            database_uri=os.path.join(temp_dir, "text_search.db"),
            create_if_not_exists=True,
        )

        # Initialize service
        service = LanceDBService(credentials)

        # Create a table for text embeddings
        table_config = TableConfig(
            name="text_embeddings",
            vector_dim=384,  # Dimension of all-MiniLM-L6-v2 embeddings
            schema={
                "metadata": "json",
            },
        )
        await service.create_table(table_config)

        # Prepare vectors with text metadata
        vectors: List[VectorData] = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append(
                VectorData(vector=embedding.tolist(), metadata={"id": i, "text": text})
            )

        # Add vectors to the table
        result = await service.add_vectors("text_embeddings", vectors)
        print(f"Added {result.affected_rows} text embeddings")

        # Create an index for faster search
        index_config = IndexConfig(
            table_name="text_embeddings", index_type="hnsw", metric="cosine"
        )
        await service.create_index(index_config)

        # Perform text similarity search
        query_text = "Animals in nature"
        query_embedding = model.encode([query_text])[0]

        search_config = SearchConfig(
            table_name="text_embeddings",
            query_vector=query_embedding.tolist(),
            metric="cosine",
            k=3,
        )

        results = await service.search_vectors(search_config)

        print(f"\nSearch Results for query: '{query_text}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Text: {result.metadata['text']}")
            print(
                f"  Similarity Score: {1 - result.distance:.4f}"
            )  # Convert distance to similarity

        # Clean up
        await service.delete_table("text_embeddings")
        print("\nTable deleted successfully")


if __name__ == "__main__":
    asyncio.run(main())
