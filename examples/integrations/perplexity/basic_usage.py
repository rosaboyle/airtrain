#!/usr/bin/env python3
"""
Basic example demonstrating how to use the Perplexity AI integration in Airtrain.

This example shows:
1. How to set up credentials
2. How to query a basic Perplexity AI model (sonar-pro)
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import airtrain
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from airtrain.integrations.perplexity import (
    PerplexityCredentials,
    PerplexityChatSkill,
    PerplexityInput,
)


def main() -> None:
    """Run the basic Perplexity AI example"""
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

    # Basic query with sonar-pro model
    print("\n=== Basic Query (sonar-pro) ===")
    input_data = PerplexityInput(
        user_input="What are the key features of Perplexity AI?",
        model="sonar-pro",
        max_tokens=300,
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response: {output.response}")
    print(f"Model: {output.used_model}")
    print(f"Usage: {json.dumps(output.usage, indent=2)}")


if __name__ == "__main__":
    main()
