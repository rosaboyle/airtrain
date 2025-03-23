#!/usr/bin/env python3
"""
Example demonstrating how to combine Exa search with GPT-4o.

This script shows how to use Exa search results with OpenAI's GPT-4o model
to provide enhanced search capabilities with reasoning and synthesis.
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any
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
from airtrain.integrations.openai.credentials import OpenAICredentials
from airtrain.integrations.openai.skills import OpenAIChatSkill, OpenAIChatInput


class SearchWithReasoning:
    """Class that combines Exa search with GPT-4o reasoning."""

    def __init__(self):
        """Initialize the search with reasoning component."""
        # Initialize Exa search
        exa_api_key = os.getenv("EXA_API_KEY")
        if not exa_api_key:
            raise ValueError("EXA_API_KEY environment variable not set")

        self.exa_credentials = ExaCredentials(api_key=exa_api_key)
        self.search_skill = ExaSearchSkill(credentials=self.exa_credentials)

        # Initialize OpenAI GPT-4o
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.openai_credentials = OpenAICredentials(api_key=openai_api_key)
        self.chat_skill = OpenAIChatSkill(credentials=self.openai_credentials)

    async def search_and_reason(
        self, query: str, num_results: int = 5, model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Perform a search and then use GPT-4o to reason about the results.

        Args:
            query: The search query to run
            num_results: Number of search results to retrieve
            model: The OpenAI model to use (default: gpt-4o)

        Returns:
            Dictionary containing the search results, GPT reasoning, and sources
        """
        # Step 1: Run the Exa search
        search_input = ExaSearchInputSchema(
            query=query, numResults=num_results, contents=ExaContentConfig(text=True)
        )

        search_results = await self.search_skill.process(search_input)

        # Step 2: Format the search results for GPT-4o
        context = self._format_search_results(search_results)

        # Step 3: Generate prompt for GPT-4o
        prompt = f"""
You are a helpful AI search assistant. I'll provide you with search results for the query: "{query}"

Your task is to:
1. Analyze these search results
2. Provide a comprehensive answer to the query
3. Identify any contradictions or uncertainties in the search results
4. Include relevant facts, data, and evidence to support your answer
5. Cite your sources using [1], [2], etc. format, corresponding to the sources below

Search results:
{context}

Please provide your answer, followed by a "Sources:" section at the end that lists the URLs you referenced.
"""

        # Step 4: Get GPT-4o response
        chat_input = OpenAIChatInput(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI search assistant that analyzes search results and provides accurate, well-sourced information.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.2,
            max_tokens=1500,
        )

        gpt_response = await self.chat_skill.process(chat_input)

        # Step 5: Return the combined results
        return {
            "query": query,
            "search_results": [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": (
                        item.text[:200] + "..."
                        if item.text and len(item.text) > 200
                        else item.text
                    ),
                }
                for item in search_results.results
            ],
            "reasoning": gpt_response.content,
            "model": model,
        }

    def _format_search_results(self, search_results):
        """Format search results for inclusion in the GPT-4o prompt."""
        context = ""

        for i, result in enumerate(search_results.results, 1):
            context += f"[{i}] {result.title if result.title else 'No title'}\n"
            context += f"URL: {result.url}\n"

            # Add content if available
            if result.text:
                # Limit to first 800 chars to keep context manageable
                snippet = (
                    result.text[:800] + "..." if len(result.text) > 800 else result.text
                )
                context += f"Content: {snippet}\n"

            context += "\n"

        return context


async def run_example():
    """Run the example."""
    print("=== Exa Search with GPT-4o Reasoning ===\n")

    # Check for API keys
    if not os.getenv("EXA_API_KEY"):
        print("Warning: EXA_API_KEY environment variable not set.")
        print("This example will not work without a valid Exa API key.")
        print("You can get an API key from https://exa.ai/\n")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("This example will not work without a valid OpenAI API key.\n")
        return

    # Create the search+reasoning component
    search_with_reasoning = SearchWithReasoning()

    # Define search queries
    queries = [
        "What are the latest advancements in quantum computing?",
        "How does climate change affect ocean ecosystems?",
    ]

    # Run searches with reasoning
    for query in queries:
        print(f"\nQuery: {query}")
        print("Searching and reasoning...\n")

        try:
            result = await search_with_reasoning.search_and_reason(
                query=query, num_results=4, model="gpt-4o"  # Using GPT-4o
            )

            print("=== GPT-4o Reasoning ===")
            print(result["reasoning"])

            print("\n=== Search Sources ===")
            for i, source in enumerate(result["search_results"], 1):
                print(f"[{i}] {source['title']}")
                print(f"    URL: {source['url']}")

        except Exception as e:
            print(f"Error: {str(e)}")

        print("\n" + "-" * 50)


def main():
    """Run the main example."""
    asyncio.run(run_example())


if __name__ == "__main__":
    main()
