#!/usr/bin/env python3
"""
Example demonstrating how to use the Perplexity AI 'sonar-pro' model.

This example shows:
1. How to use sonar-pro for advanced search with grounding
2. How to access citations in the response
"""

import os
import sys
import json
from typing import Optional, List
from dotenv import load_dotenv

# Add the parent directory to the path so we can import airtrain
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from airtrain.integrations.perplexity import (
    PerplexityCredentials,
    PerplexityChatSkill,
    PerplexityInput,
    PerplexityCitation,
)


def print_citations(citations: Optional[List[PerplexityCitation]]) -> None:
    """Print the citations used in the response.

    Args:
        citations: List of citation objects or None
    """
    print("\nCitations:")
    if citations:
        for i, citation in enumerate(citations, 1):
            print(f"{i}. {citation.url}")
            if citation.title:
                print(f"   Title: {citation.title}")
            if citation.snippet:
                print(f"   Snippet: {citation.snippet[:100]}...")
            print()
    else:
        print("No citations provided in the response.")


def main() -> None:
    """Run the sonar-pro model example"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY environment variable not set")
        print("Please set it in your .env file or export it in your shell")
        sys.exit(1)

    # Set up credentials
    credentials = PerplexityCredentials(perplexity_api_key=api_key)

    # Create chat skill
    chat_skill = PerplexityChatSkill(credentials=credentials)

    print("\n=== Sonar Pro Example - Current Events Search ===")
    input_data = PerplexityInput(
        user_input="What are the latest developments in quantum computing in 2024?",
        model="sonar-pro",
        max_tokens=500,
        temperature=0.7,
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")
    print(f"Usage: {json.dumps(output.usage, indent=2)}")

    # Print citations
    print_citations(output.citations)

    # Example with different topic
    print("\n\n=== Sonar Pro Example - Technical Documentation ===")
    input_data = PerplexityInput(
        user_input="Explain the key features of PyTorch 2.0 and how it differs from earlier versions.",
        model="sonar-pro",
        max_tokens=500,
        temperature=0.1,  # Lower temperature for more factual responses
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print_citations(output.citations)


if __name__ == "__main__":
    main()
