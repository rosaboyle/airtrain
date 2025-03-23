import asyncio
import os
from typing import List
import tempfile
from dotenv import load_dotenv
import pyarrow as pa

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, parent_dir)

from airtrain.integrations.openai.skills import (
    OpenAIEmbeddingsSkill,
    OpenAIEmbeddingsInput,
)
from airtrain.integrations.lancedb.database_service import LanceDBService
from airtrain.integrations.lancedb.credentials import LanceDBCredentials
from airtrain.integrations.lancedb.schemas import (
    TableConfig,
    VectorData,
    SearchConfig,
    IndexConfig,
)

# Load environment variables (for OpenAI API key)
load_dotenv()


async def main():
    """Example demonstrating OpenAI embeddings with LanceDB"""
    try:
        # Initialize the OpenAI embeddings skill
        embeddings_skill = OpenAIEmbeddingsSkill()

        # Sample documents
        documents = [
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
            "Machine learning algorithms process data",
            "Neural networks learn patterns",
            "Artificial intelligence transforms industries",
            "Data scientists analyze trends",
            "Deep learning models recognize images",
        ]

        print("Generating embeddings...")
        # Generate embeddings for the documents
        embeddings_input = OpenAIEmbeddingsInput(
            texts=documents,
            model="text-embedding-3-large",  # Using the latest model
            dimensions=1024,  # Using smaller dimensions for testing
        )
        embeddings_output = await embeddings_skill.process_async(embeddings_input)
        print(f"Generated {len(embeddings_output.embeddings)} embeddings")

        # Use a fixed directory for testing
        db_path = os.path.join("test_lancedb", "semantic_search.db")

        # Initialize LanceDB credentials and service
        credentials = LanceDBCredentials(database_uri=db_path)
        service = LanceDBService(credentials)

        print("Creating table...")
        # Create a table for document embeddings
        table_config = TableConfig(
            name="document_embeddings",
            vector_dim=1024,  # Using smaller dimensions
            mode="overwrite",  # Overwrite if exists
        )
        await service.create_table(table_config)

        print("Adding vectors to table...")
        # Prepare vectors with document metadata
        vectors: List[VectorData] = []
        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings_output.embeddings)
        ):
            vectors.append(
                VectorData(vector=embedding, metadata={"id": i, "text": doc})
            )

        # Add vectors to the table
        result = await service.add_vectors("document_embeddings", vectors)
        print(f"Added {result.affected_rows} document embeddings")

        print("Creating search index...")
        # Create an index for faster search
        index_config = IndexConfig(
            table_name="document_embeddings",
            index_type="ivf_pq",  # Using IVF-PQ index
            metric="cosine",
            params={
                "num_partitions": 128,  # Smaller number for our small dataset
                "num_sub_vectors": 64,
                "max_iterations": 50,
            },
        )
        # await service.create_index(index_config)

        # Perform semantic search
        query_text = "Computer technology and AI"
        print(f"\nPerforming search for: {query_text}")

        query_embedding_input = OpenAIEmbeddingsInput(
            texts=query_text, model="text-embedding-3-large", dimensions=1024
        )
        query_embedding_output = await embeddings_skill.process_async(
            query_embedding_input
        )

        search_config = SearchConfig(
            table_name="document_embeddings",
            query_vector=query_embedding_output.embeddings[0],
            metric="cosine",
            k=5,
        )

        results = await service.search_vectors(search_config)

        print(f"\nSearch Results for query: '{query_text}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Text: {result.metadata['text']}")
            print(f"  Similarity Score: {1 - result.distance:.4f}")

        # Example of hybrid search (combining vector search with metadata filtering)
        print("\nPerforming hybrid search (filtered for 'learning')...")
        hybrid_search_config = SearchConfig(
            table_name="document_embeddings",
            query_vector=query_embedding_output.embeddings[0],
            metric="cosine",
            k=5,
            filter_expr="metadata.text LIKE '%learning%'",  # Only return documents containing 'learning'
        )

        hybrid_results = await service.search_vectors(hybrid_search_config)

        print(f"\nHybrid Search Results (filtered for 'learning'):")
        for i, result in enumerate(hybrid_results, 1):
            print(f"\nResult {i}:")
            print(f"  Text: {result.metadata['text']}")
            print(f"  Similarity Score: {1 - result.distance:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
