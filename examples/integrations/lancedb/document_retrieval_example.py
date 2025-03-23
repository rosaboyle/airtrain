import asyncio
import os
from typing import List, Dict
import tempfile
from dotenv import load_dotenv

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


def chunk_document(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Split a document into overlapping chunks

    Args:
        text: Document text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of dictionaries containing chunk text and metadata
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # Add chunk with metadata
        chunks.append(
            {
                "text": chunk_text,
                "start_idx": i,
                "end_idx": i + len(chunk_words),
                "total_chunks": (len(words) + chunk_size - 1) // (chunk_size - overlap),
            }
        )

    return chunks


async def main():
    """Example demonstrating document chunking and retrieval with OpenAI embeddings and LanceDB"""

    # Sample document (a longer text about AI and technology)
    document = """
    Artificial Intelligence (AI) has become an integral part of our daily lives, transforming how we work, communicate, and solve problems. 
    Machine learning, a subset of AI, enables computers to learn from data without explicit programming. Deep learning, inspired by the human brain's neural networks, 
    has revolutionized tasks like image recognition, natural language processing, and autonomous systems.

    The impact of AI spans across various industries. In healthcare, AI assists doctors in diagnosis and treatment planning. 
    In finance, AI algorithms detect fraudulent transactions and optimize investment strategies. Manufacturing benefits from predictive maintenance 
    and quality control powered by AI. The technology sector continues to innovate with AI-driven solutions for everything from personal assistants to autonomous vehicles.

    However, the rise of AI also brings important considerations. Ethics in AI development, data privacy, and the impact on employment are crucial discussions. 
    Responsible AI development ensures that these powerful tools benefit society while minimizing potential risks. The future of AI depends on striking the right balance 
    between innovation and ethical considerations.

    As we move forward, AI education and literacy become increasingly important. Understanding the basics of AI helps people make informed decisions about its use. 
    The collaboration between humans and AI systems creates new opportunities for solving complex problems. The next generation of AI technologies promises even more 
    exciting developments in areas like quantum computing and cognitive architectures.
    """

    # Initialize the OpenAI embeddings skill
    embeddings_skill = OpenAIEmbeddingsSkill()

    # Chunk the document
    chunks = chunk_document(document)

    # Generate embeddings for chunks
    embeddings_input = OpenAIEmbeddingsInput(
        texts=[chunk["text"] for chunk in chunks], model="text-embedding-3-large"
    )
    embeddings_output = await embeddings_skill.process_async(embeddings_input)

    # Create a temporary directory for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize LanceDB credentials and service
        credentials = LanceDBCredentials(
            database_uri=os.path.join(temp_dir, "document_retrieval.db"),
            create_if_not_exists=True,
        )
        service = LanceDBService(credentials)

        # Create a table for document chunks
        table_config = TableConfig(
            name="document_chunks",
            vector_dim=3072,  # Dimension for text-embedding-3-large
            schema={
                "metadata": "json",
            },
        )
        await service.create_table(table_config)

        # Prepare vectors with chunk metadata
        vectors: List[VectorData] = []
        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings_output.embeddings)
        ):
            vectors.append(
                VectorData(
                    vector=embedding,
                    metadata={
                        "chunk_id": i,
                        "text": chunk["text"],
                        "start_idx": chunk["start_idx"],
                        "end_idx": chunk["end_idx"],
                        "total_chunks": chunk["total_chunks"],
                    },
                )
            )

        # Add vectors to the table
        result = await service.add_vectors("document_chunks", vectors)
        print(f"Added {result.affected_rows} document chunks")

        # Create an index for faster search
        index_config = IndexConfig(
            table_name="document_chunks", index_type="hnsw", metric="cosine"
        )
        await service.create_index(index_config)

        # Example queries to demonstrate different types of retrieval
        queries = [
            "What are the main applications of AI in different industries?",
            "What are the ethical considerations in AI development?",
            "How does machine learning work?",
            "What is the future of AI technology?",
        ]

        for query in queries:
            # Generate query embedding
            query_embedding_input = OpenAIEmbeddingsInput(
                texts=query, model="text-embedding-3-large"
            )
            query_embedding_output = await embeddings_skill.process_async(
                query_embedding_input
            )

            # Search for relevant chunks
            search_config = SearchConfig(
                table_name="document_chunks",
                query_vector=query_embedding_output.embeddings[0],
                metric="cosine",
                k=2,  # Get top 2 most relevant chunks
            )

            results = await service.search_vectors(search_config)

            print(f"\nQuery: '{query}'")
            print("Relevant Document Chunks:")
            for i, result in enumerate(results, 1):
                print(f"\nChunk {i} (Similarity: {1 - result.distance:.4f}):")
                print(f"  {result.metadata['text']}")

        # Clean up
        await service.delete_table("document_chunks")
        print("\nTable deleted successfully")


if __name__ == "__main__":
    asyncio.run(main())
