#!/usr/bin/env python3
"""
Example demonstrating how to use the Perplexity AI reasoning models.

This example shows:
1. How to use sonar-reasoning for problem-solving tasks
2. How to use sonar-reasoning-pro for more complex reasoning
3. How to compare results between the models
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


def main() -> None:
    """Run the reasoning models example"""
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

    # Example 1: Basic math problem with sonar-reasoning
    print("\n=== Sonar Reasoning Example - Math Problem ===")
    math_problem = "If a train travels at 60 mph for 2 hours and then at 80 mph for 1.5 hours, how far does it travel in total?"

    input_data = PerplexityInput(
        user_input=math_problem,
        model="sonar-reasoning",
        max_tokens=500,
        temperature=0.1,  # Low temperature for deterministic reasoning
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Problem: {math_problem}")
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")
    print(f"Usage: {json.dumps(output.usage, indent=2)}")

    # Example 2: Logic puzzle with sonar-reasoning-pro
    print("\n\n=== Sonar Reasoning Pro Example - Logic Puzzle ===")
    logic_puzzle = """
    There are five houses in a row, each painted a different color and inhabited by a person of a different nationality.
    These five homeowners each drink a different beverage, smoke a different brand of cigar, and keep a different pet.
    
    The Brit lives in the red house.
    The Swede keeps dogs as pets.
    The Dane drinks tea.
    The green house is on the left of the white house.
    The green house's owner drinks coffee.
    The person who smokes Pall Mall rears birds.
    The owner of the yellow house smokes Dunhill.
    The man living in the center house drinks milk.
    The Norwegian lives in the first house.
    The man who smokes Blends lives next to the one who keeps cats.
    The man who keeps horses lives next to the man who smokes Dunhill.
    The owner who smokes Bluemasters drinks beer.
    The German smokes Prince.
    The Norwegian lives next to the blue house.
    The man who smokes Blends has a neighbor who drinks water.
    
    Who owns the fish?
    """

    input_data = PerplexityInput(
        user_input=logic_puzzle,
        model="sonar-reasoning-pro",
        max_tokens=1000,
        temperature=0.1,
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Puzzle: Einstein's Puzzle")
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")
    print(f"Usage: {json.dumps(output.usage, indent=2)}")

    # Example 3: Programming problem with sonar-reasoning-pro
    print("\n\n=== Sonar Reasoning Pro Example - Programming Problem ===")
    programming_problem = """
    Write a Python function to find the longest common subsequence of two strings.
    Explain your approach and provide a time and space complexity analysis.
    Then test your function with the strings "ABCBDAB" and "BDCABA".
    """

    input_data = PerplexityInput(
        user_input=programming_problem,
        model="sonar-reasoning-pro",
        max_tokens=800,
        temperature=0.2,
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Problem: Programming Challenge")
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")


if __name__ == "__main__":
    main()
