import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.fireworks.structured_skills import (
    FireworksParserSkill,
    FireworksParserInput,
)


class MovieReview(BaseModel):
    """Example response model for movie review"""

    title: str
    year: int
    rating: float
    genre: List[str]
    review: str
    pros: List[str]
    cons: List[str]


def main():
    # Initialize the parser skill
    skill = FireworksParserSkill()

    # Create input for structured parsing
    input_data = FireworksParserInput(
        user_input="Review the movie 'Inception' (2010)",
        system_prompt="You are a movie critic that provides structured reviews.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        response_model=MovieReview,
    )

    try:
        result = skill.process(input_data)
        print("\nStructured Movie Review:")
        print(result.parsed_response.model_dump_json(indent=2))
        if result.reasoning:
            print("\nReasoning:")
            print(result.reasoning)
        print("\nModel Used:", result.used_model)
        print("Tokens Used:", result.tokens_used)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
