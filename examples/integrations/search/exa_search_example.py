#!/usr/bin/env python3
"""
Example demonstrating how to use the Exa search skill.

This script shows how to perform web searches using the Exa search API
via the Airtrain integration.
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

# Load environment variables
load_dotenv()

from airtrain.integrations.search.exa import (
    ExaCredentials,
    ExaSearchSkill,
    ExaSearchInputSchema,
    ExaContentConfig,
)


async def run_basic_search():
    """Run a basic web search using Exa."""
    print("=== Basic Exa Search ===")

    # Get API key from environment variable or use a placeholder
    api_key = os.getenv("EXA_API_KEY", "your-api-key-here")

    # Create credentials
    credentials = ExaCredentials(api_key=api_key)

    # Initialize the search skill
    search_skill = ExaSearchSkill(credentials=credentials)

    # Create search input
    search_input = ExaSearchInputSchema(
        query="What are the latest developments in AI safety?",
        numResults=3,
        contents=ExaContentConfig(text=True),
    )

    try:
        # Execute the search
        result = await search_skill.process(search_input)

        # Print search results
        print(f"\nSearch query: {result.query}")
        print(f"Found {len(result.results)} results:\n")

        for i, item in enumerate(result.results, 1):
            print(f"Result {i}: {item.title}")
            print(f"URL: {item.url}")
            print(f"Score: {item.score}")

            # Print a snippet of the text
            if item.text:
                snippet = item.text[:200] + "..." if len(item.text) > 200 else item.text
                print(f"Snippet: {snippet}")

            print()

        # Print cost information if available
        if result.costDollars:
            print(f"Search cost: ${result.costDollars.total}")

    except Exception as e:
        print(f"Error performing search: {str(e)}")


async def run_domain_specific_search():
    """Run a domain-specific search using Exa."""
    print("\n=== Domain-Specific Exa Search ===")

    # Get API key from environment variable or use a placeholder
    api_key = os.getenv("EXA_API_KEY", "your-api-key-here")

    # Create credentials
    credentials = ExaCredentials(api_key=api_key)

    # Initialize the search skill
    search_skill = ExaSearchSkill(credentials=credentials)

    # Create search input with domain filtering
    search_input = ExaSearchInputSchema(
        query="Neural network architecture",
        numResults=3,
        includeDomains=["arxiv.org", "openai.com", "pytorch.org"],
        contents=ExaContentConfig(text=True),
    )

    try:
        # Execute the search
        result = await search_skill.process(search_input)

        # Print search results
        print(f"\nSearch query: {result.query}")
        print(f"Found {len(result.results)} results from specified domains:\n")

        for i, item in enumerate(result.results, 1):
            print(f"Result {i}: {item.title}")
            print(f"URL: {item.url}")
            print(f"Score: {item.score}")

            # Print a snippet of the text
            if item.text:
                snippet = item.text[:200] + "..." if len(item.text) > 200 else item.text
                print(f"Snippet: {snippet}")

            print()

    except Exception as e:
        print(f"Error performing search: {str(e)}")


async def main():
    """Run the example."""
    # Check for API key
    if not os.getenv("EXA_API_KEY"):
        print("Warning: EXA_API_KEY environment variable not set.")
        print("This example will not work without a valid API key.")
        print("You can get an API key from https://exa.ai/\n")

    # Run examples
    await run_basic_search()
    await run_domain_specific_search()


if __name__ == "__main__":
    asyncio.run(main())
