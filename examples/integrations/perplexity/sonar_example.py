#!/usr/bin/env python3
"""
Example demonstrating how to use the Perplexity AI 'sonar' model.

This example shows:
1. How to use the lightweight sonar model
2. How to adjust parameters for different response styles
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
            print(f"{i}. {citation.url} - {citation.title or 'No title'}")
    else:
        print("No citations provided in the response.")


def main() -> None:
    """Run the sonar model example"""
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

    print("\n=== Sonar Example - Basic Search ===")
    input_data = PerplexityInput(
        user_input="What is the distance between Earth and Mars?",
        model="sonar",
        max_tokens=300,
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

    # Example with system prompt
    print("\n\n=== Sonar Example - With System Prompt ===")
    input_data = PerplexityInput(
        user_input="Explain the concept of machine learning to a 10-year-old",
        system_prompt="You are a tutor for elementary school students. Use simple language and relatable examples.",
        model="sonar",
        max_tokens=300,
        temperature=0.8,  # Higher temperature for more creative responses
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")

    # Example with different parameters
    print("\n\n=== Sonar Example - Adjusting Parameters ===")
    input_data = PerplexityInput(
        user_input="What are the major climate zones on Earth?",
        model="sonar",
        max_tokens=400,
        temperature=0.2,  # Lower temperature for more focused responses
        top_p=0.8,  # Adjust top_p for more focused token selection
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print_citations(output.citations)


if __name__ == "__main__":
    main()
