import sys
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
import json
from time import sleep

load_dotenv()

parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.fireworks.structured_requests_skills import (
    FireworksStructuredRequestSkill,
    FireworksStructuredRequestInput,
)


class MovieReview(BaseModel):
    """Schema for movie review"""

    title: str = Field(..., description="Movie title")
    year: int = Field(..., description="Release year")
    rating: float = Field(..., description="Rating out of 10")
    review: str = Field(..., description="Detailed review")
    pros: list[str] = Field(..., description="List of positive points")
    cons: list[str] = Field(..., description="List of negative points")


def stream_example():
    """Example of streaming structured output"""
    skill = FireworksStructuredRequestSkill()

    input_data = FireworksStructuredRequestInput(
        user_input="Review the movie 'Inception'",
        system_prompt="You are a movie critic providing structured reviews.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=8096,
        response_model=MovieReview,
        stream=True,
    )

    print("\nStreaming structured movie review...")
    try:
        for chunk in skill.process_stream(input_data):
            if "chunk" in chunk:
                # Print the raw streaming content
                print(chunk["chunk"], end="", flush=True)
                sleep(0.01)  # Optional: add slight delay for visualization
            elif "complete" in chunk:
                # Print the parsed response and reasoning
                print("\n\nParsed Response:")
                print(chunk["complete"].model_dump_json(indent=2))
                if "reasoning" in chunk:
                    print("\nReasoning:")
                    print(chunk["reasoning"])
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def non_stream_example():
    """Example of non-streaming structured output"""
    skill = FireworksStructuredRequestSkill()

    input_data = FireworksStructuredRequestInput(
        user_input="Review the movie 'Inception'",
        system_prompt="You are a movie critic providing structured reviews.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=8096,
        response_model=MovieReview,
        stream=False,
    )

    try:
        print("\nGetting structured movie review...")
        result = skill.process(input_data)
        print("\nParsed Response:")
        print(result.parsed_response.model_dump_json(indent=2))
        if result.reasoning:
            print("\nReasoning:")
            print(result.reasoning)
        print("\nModel Used:", result.used_model)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Run the examples"""
    print("=== Streaming Example ===")
    # stream_example()

    print("\n=== Non-Streaming Example ===")
    non_stream_example()


if __name__ == "__main__":
    main()
