#!/usr/bin/env python3
"""
Example usage of AirTrain's WebSearchTool.

This script demonstrates how to use the WebSearchTool for searching the web
via the Exa search API.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List
import asyncio
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

# Load environment variables
load_dotenv()

# Import required modules
from airtrain.tools import ToolFactory, WebSearchTool
from airtrain.integrations.search.exa import ExaCredentials, ExaSearchSkill


def demo_web_search():
    """Demonstrate the WebSearchTool."""
    print("\n=== Web Search Tool ===")

    # Check for API key
    if not os.getenv("EXA_API_KEY"):
        print("Warning: EXA_API_KEY environment variable not set.")
        print("This example will not work without a valid API key.")
        print("You can get an API key from https://exa.ai/\n")
        return

    # Get the tool
    search_tool = ToolFactory.get_tool("web_search")

    # Define search parameters
    query = "What are the latest developments in AI safety research?"
    num_results = 3

    print(f"Searching for: {query}")
    print(f"Number of results: {num_results}")
    print("\nSearching the web...\n")

    # Execute the search
    results = search_tool(query=query, num_results=num_results, use_autoprompt=True)

    # Display results
    if results["success"]:
        print(f"Found {results['result_count']} results:")

        if results.get("autoprompt"):
            print(f"Autoprompt: {results['autoprompt']}\n")

        for i, result in enumerate(results["results"], 1):
            print(f"Result {i}: {result['title']}")
            print(f"URL: {result['url']}")

            # Print a snippet of the content
            content = result["content"]
            snippet = content[:200] + "..." if len(content) > 200 else content
            print(f"Snippet: {snippet}\n")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")


def demo_domain_specific_search():
    """Demonstrate the WebSearchTool with domain filtering."""
    print("\n=== Domain-Specific Web Search ===")

    # Check for API key
    if not os.getenv("EXA_API_KEY"):
        return

    # Get the tool
    search_tool = ToolFactory.get_tool("web_search")

    # Define search parameters
    query = "Neural network architecture"
    include_domains = ["arxiv.org", "openai.com", "pytorch.org"]

    print(f"Searching for: {query}")
    print(f"Include domains: {', '.join(include_domains)}")
    print("\nSearching the web...\n")

    # Execute the search
    results = search_tool(query=query, num_results=3, include_domains=include_domains)

    # Display results
    if results["success"]:
        print(f"Found {results['result_count']} results:")

        for i, result in enumerate(results["results"], 1):
            print(f"Result {i}: {result['title']}")
            print(f"URL: {result['url']}")

            # Print the domain
            domain = result["url"].split("//")[-1].split("/")[0]
            print(f"Domain: {domain}")

            # Print a snippet of the content
            content = result["content"]
            snippet = content[:200] + "..." if len(content) > 200 else content
            print(f"Snippet: {snippet}\n")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")


def main():
    """Run examples."""
    # Run the demos
    demo_web_search()
    demo_domain_specific_search()


if __name__ == "__main__":
    main()
