import asyncio
import numpy as np
from typing import List
import tempfile
import os

from airtrain.integrations.lancedb.database_service import LanceDBService
from airtrain.integrations.lancedb.credentials import LanceDBCredentials
from airtrain.integrations.lancedb.schemas import (
    TableConfig,
    VectorData,
    SearchConfig,
    IndexConfig,
)


async def main():
    """Example demonstrating basic LanceDB operations"""

    # Create a temporary directory for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize credentials with temporary directory
        credentials = LanceDBCredentials(
            database_uri=os.path.join(temp_dir, "test.db"), create_if_not_exists=True
        )

        # Initialize service
        service = LanceDBService(credentials)

        # 1. Create a table
        table_config = TableConfig(
            name="test_vectors",
            vector_dim=128,
            schema={
                "metadata": "json",
            },
        )
        await service.create_table(table_config)

        # 2. Generate some test vectors
        num_vectors = 1000
        vectors: List[VectorData] = []
        for i in range(num_vectors):
            vector = np.random.rand(128).astype(np.float32).tolist()
            metadata = {"id": i, "category": f"category_{i % 5}"}
            vectors.append(VectorData(vector=vector, metadata=metadata))

        # 3. Add vectors to the table
        result = await service.add_vectors("test_vectors", vectors)
        print(f"Added {result.affected_rows} vectors")

        # 4. Create an index for faster search
        index_config = IndexConfig(
            table_name="test_vectors",
            index_type="hnsw",
            metric="cosine",
            params={"num_segments": 2, "num_threads": 4},
        )
        await service.create_index(index_config)

        # 5. Perform a vector search
        query_vector = np.random.rand(128).astype(np.float32).tolist()
        search_config = SearchConfig(
            table_name="test_vectors",
            query_vector=query_vector,
            metric="cosine",
            k=5,
            filter_expr="metadata.category = 'category_1'",
        )

        results = await service.search_vectors(search_config)
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Distance: {result.distance:.4f}")
            print(f"  Metadata: {result.metadata}")

        # 6. Get table information
        table_info = await service.get_table_info("test_vectors")
        print("\nTable Information:")
        print(f"  Name: {table_info['name']}")
        print(f"  Number of rows: {table_info['num_rows']}")
        print(f"  Has index: {table_info['has_index']}")

        # 7. Delete the table
        await service.delete_table("test_vectors")
        print("\nTable deleted successfully")


if __name__ == "__main__":
    asyncio.run(main())
