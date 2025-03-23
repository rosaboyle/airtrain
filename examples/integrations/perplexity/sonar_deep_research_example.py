#!/usr/bin/env python3
"""
Example demonstrating how to use the Perplexity AI 'sonar-deep-research' model.

This example shows:
1. How to use sonar-deep-research for comprehensive research
2. How to get detailed analysis on complex topics
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
            print()
    else:
        print("No citations provided in the response.")


def main() -> None:
    """Run the sonar-deep-research model example"""
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

    print("\n=== Sonar Deep Research Example - Comprehensive Analysis ===")
    input_data = PerplexityInput(
        user_input="Provide a comprehensive analysis of the economic impacts of climate change on global agriculture over the next 30 years.",
        model="sonar-deep-research",
        max_tokens=1000,  # Allow for longer responses
        temperature=0.2,  # Lower temperature for more focused, analytical responses
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")
    print(f"Usage: {json.dumps(output.usage, indent=2)}")

    # Print citations
    print_citations(output.citations)

    # Example with technology research
    print("\n\n=== Sonar Deep Research Example - Technology Forecast ===")
    input_data = PerplexityInput(
        user_input="Research and analyze the potential impact of quantum computing on cybersecurity over the next decade. Include specific threats, mitigation strategies, and industry preparedness.",
        model="sonar-deep-research",
        max_tokens=1000,
        temperature=0.3,
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print_citations(output.citations)


if __name__ == "__main__":
    main()
