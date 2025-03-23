import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

from airtrain.integrations.together.rerank_skill import TogetherAIRerankSkill
from airtrain.integrations.together.schemas import TogetherAIRerankInput


def main():
    # Initialize the rerank skill
    rerank_skill = TogetherAIRerankSkill()

    # Example query and documents
    query = "What animals can I find near Peru?"
    documents = [
        "The giant panda (Ailuropoda melanoleuca), also known as the panda bear or simply panda, is a bear species endemic to China.",
        "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.",
        "The wild Bactrian camel (Camelus ferus) is an endangered species of camel endemic to Northwest China and southwestern Mongolia.",
        "The guanaco is a camelid native to South America, closely related to the llama. Guanacos are one of two wild South American camelids; the other species is the vicuña, which lives at higher elevations.",
    ]

    # Create input for the skill
    rerank_input = TogetherAIRerankInput(
        query=query, documents=documents, top_n=2  # Optional: limit to top 2 results
    )

    try:
        # Process the reranking request
        result = rerank_skill.process(rerank_input)

        print(f"\nQuery: {query}\n")
        print("Ranked Results:")
        print("-" * 50)

        for ranked_doc in result.results:
            print(f"Score: {ranked_doc.relevance_score:.4f}")
            print(f"Document: {ranked_doc.document}")
            print(f"Original Index: {ranked_doc.index}")
            print("-" * 50)

        print(f"\nModel Used: {result.used_model}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
