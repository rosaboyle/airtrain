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

from airtrain.integrations.openai.skills import OpenAIParserSkill, OpenAIParserInput


class MovieReview(BaseModel):
    """Example response model for movie review"""

    title: str
    year: int
    rating: float
    genre: List[str]
    review: str
    pros: List[str]
    cons: List[str]

    @validator("rating")
    def validate_rating(cls, v):
        if not 0 <= v <= 10:
            raise ValueError("Rating must be between 0 and 10")
        return v


def main():
    # Initialize the parser skill
    skill = OpenAIParserSkill()

    # Create input for structured parsing
    input_data = OpenAIParserInput(
        user_input="Review the movie 'Inception' (2010)",
        system_prompt="You are a movie critic that provides structured reviews.",
        model="gpt-4o",
        temperature=0.3,
        response_model=MovieReview,
    )

    try:
        result = skill.process(input_data)
        print("\nStructured Movie Review:")
        print(result.parsed_response.model_dump_json(indent=2))
        print("\nModel Used:", result.used_model)
        print("Tokens Used:", result.tokens_used)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
