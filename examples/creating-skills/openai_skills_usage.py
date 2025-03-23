import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.integrations.openai.skills import (
    OpenAIChatSkill,
    OpenAIInput,
    OpenAIParserSkill,
    OpenAIParserInput,
)


# Example Response Models for structured output
class PersonInfo(BaseModel):
    """Example response model for person information"""

    name: str
    age: int
    occupation: str
    skills: List[str]
    contact: Optional[Dict[str, str]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {"required": ["name", "age", "occupation", "skills"]}


class MovieReview(BaseModel):
    """Example response model for movie review"""

    title: str
    year: int
    rating: float
    genre: List[str]
    review: str
    pros: List[str]
    cons: List[str]

    class Config:
        json_schema_extra = {
            "required": ["title", "year", "rating", "genre", "review", "pros", "cons"]
        }

    @validator("rating")
    def validate_rating(cls, v):
        """Validate rating after parsing"""
        if not 0 <= v <= 10:
            raise ValueError("Rating must be between 0 and 10")
        return v


def main() -> None:
    """
    Demonstrates usage of OpenAI chat skills with different scenarios.
    """
    # Basic text completion example
    skill = OpenAIChatSkill()

    # Create proper OpenAIInput instance
    input_data = OpenAIInput(
        user_input="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful teacher who explains complex topics simply.",
        max_tokens=500,
        model="gpt-4o",  # Using latest GPT-4 model
    )

    # Process with proper input model
    result = skill.process(input_data)

    print("Basic Response:")
    print(result.response)
    print("\nUsage Statistics:", result.usage)

    # Example with image analysis using GPT-4 Vision
    image_path = Path("examples/images/quantum_circuit.jpg")  # Example image path
    if image_path.exists():
        # Create input model with image
        image_input = OpenAIInput(
            user_input="What does this quantum circuit diagram show?",
            images=[image_path],
            system_prompt="You are an expert in quantum computing. Analyze the circuit diagram.",
            model="gpt-4o",  # Using Vision model for image analysis
            temperature=0.3,  # Lower temperature for more focused analysis
            max_tokens=1000,
        )

        result_with_image = skill.process(image_input)

        print("\nImage Analysis Response:")
        print(result_with_image.response)
        print("\nUsage Statistics:", result_with_image.usage)

    # Example with function calling
    function_input = OpenAIInput(
        user_input="What's the weather like in London?",
        system_prompt="You are a weather assistant.",
        model="gpt-4o",
        functions=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    result_with_function = skill.process(function_input)

    print("\nFunction Calling Response:")
    print(result_with_function.response)
    print("\nFunction Call:", result_with_function.function_call)
    print("\nUsage Statistics:", result_with_function.usage)

    print("\n=== Testing Structured Output Capabilities ===")

    # Initialize the parser skill
    parser_skill = OpenAIParserSkill()

    # Test person info parsing
    print("\nTesting Person Info Parsing:")
    person_input = OpenAIParserInput(
        user_input="Tell me about John Doe, a 30-year-old software engineer who specializes in Python and machine learning",
        system_prompt="You are an assistant that extracts structured information about people.",
        model="gpt-4o",
        response_model=PersonInfo,
    )

    try:
        person_result = parser_skill.process(person_input)
        print("\nPerson Info Structure:")
        print(person_result.parsed_response.model_dump_json(indent=2))
        print(f"Tokens Used: {person_result.tokens_used}")
    except Exception as e:
        print(f"Error in person parsing: {e}")

    # Test movie review parsing
    print("\nTesting Movie Review Parsing:")
    movie_input = OpenAIParserInput(
        user_input="Review the movie 'Inception' (2010)",
        system_prompt="You are a movie critic that provides structured reviews.",
        model="gpt-4o",
        response_model=MovieReview,
    )

    try:
        movie_result = parser_skill.process(movie_input)
        print("\nMovie Review Structure:")
        print(movie_result.parsed_response.model_dump_json(indent=2))
        print(f"Tokens Used: {movie_result.tokens_used}")
    except Exception as e:
        print(f"Error in movie review parsing: {e}")


if __name__ == "__main__":
    main()
