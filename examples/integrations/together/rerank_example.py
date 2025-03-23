import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.together.rerank_skill import (
    TogetherAIRerankSkill,
    TogetherAIRerankInput,
)


def main():
    # Initialize the rerank skill
    skill = TogetherAIRerankSkill()

    # Example documents
    query = "What are the health benefits of exercise?"
    documents = [
        "Regular exercise improves cardiovascular health and reduces the risk of heart disease.",
        "A balanced diet is essential for maintaining good health and energy levels.",
        "Exercise helps in maintaining mental health and reducing stress levels.",
        "Getting enough sleep is crucial for overall health and well-being.",
    ]

    # Create input
    rerank_input = TogetherAIRerankInput(
        query=query,
        documents=documents,
        top_n=2,
    )

    try:
        result = skill.process(rerank_input)
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
