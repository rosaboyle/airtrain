import sys
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import time

load_dotenv()

parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.fireworks.structured_completion_skills import (
    FireworksStructuredCompletionSkill,
    FireworksStructuredCompletionInput,
)


class RecipeInstructions(BaseModel):
    """Example schema for recipe instructions"""

    title: str
    ingredients: List[str]
    steps: List[str]
    cooking_time: int
    difficulty: str
    serves: int


def stream_example():
    """Example of streaming structured completion"""
    skill = FireworksStructuredCompletionSkill()

    input_data = FireworksStructuredCompletionInput(
        prompt="Create a recipe for chocolate chip cookies",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=1000,
        response_model=RecipeInstructions,
        stream=True,
    )

    print("\nStreaming structured recipe...")
    try:
        json_buffer = []
        for chunk in skill.process_stream(input_data):
            if "chunk" in chunk:
                print(chunk["chunk"], end="", flush=True)
                time.sleep(0.01)  # Optional: add slight delay for visualization
            elif "complete" in chunk:
                print("\n\nParsed Response:")
                print(chunk["complete"].model_dump_json(indent=2))
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def non_stream_example():
    """Example of non-streaming structured completion"""
    skill = FireworksStructuredCompletionSkill()

    # Example 1: Recipe
    input_data = FireworksStructuredCompletionInput(
        prompt="Create a recipe for a classic margherita pizza",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=1000,
        response_model=RecipeInstructions,
        stream=False,
    )

    try:
        print("\nGetting structured recipe...")
        result = skill.process(input_data)
        print("\nParsed Response:")
        print(result.parsed_response.model_dump_json(indent=2))
        print("\nModel Used:", result.used_model)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 2: Another recipe with different parameters
    input_data = FireworksStructuredCompletionInput(
        prompt="Create a recipe for a vegan chocolate cake",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=1000,
        response_model=RecipeInstructions,
        stream=False,
    )

    try:
        print("\n\nGetting another structured recipe...")
        result = skill.process(input_data)
        print("\nParsed Response:")
        print(result.parsed_response.model_dump_json(indent=2))
        print("\nModel Used:", result.used_model)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    # Run streaming example
    stream_example()

    # Run non-streaming examples
    non_stream_example()


if __name__ == "__main__":
    main()
