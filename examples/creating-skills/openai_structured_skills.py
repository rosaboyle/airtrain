import sys
import os
from typing import Type, TypeVar, Optional, List, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.core.skills import Skill, ProcessingError
from airtrain.core.schemas import InputSchema, OutputSchema
from airtrain.integrations.openai.skills import OpenAIParserSkill, OpenAIParserInput
from airtrain.integrations.openai.credentials import OpenAICredentials

# Initialize OpenAI client
client = OpenAI()

# Generic type variable for Pydantic response models
ResponseT = TypeVar("ResponseT", bound=BaseModel)


class OpenAIParserInput(InputSchema):
    """Schema for OpenAI structured output input"""

    user_input: str
    system_prompt: str = "You are a helpful assistant that provides structured data."
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    response_model: Type[ResponseT]
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of previous conversation messages in [{'role': 'user|assistant', 'content': 'message'}] format",
    )

    class Config:
        arbitrary_types_allowed = True


class OpenAIParserOutput(OutputSchema):
    """Schema for OpenAI structured output"""

    parsed_response: BaseModel
    used_model: str
    tokens_used: int


class OpenAIParserSkill(Skill[OpenAIParserInput, OpenAIParserOutput]):
    """Skill for getting structured responses from OpenAI"""

    input_schema = OpenAIParserInput
    output_schema = OpenAIParserOutput

    def __init__(self, credentials: Optional["OpenAICredentials"] = None):
        """Initialize the skill with optional credentials"""
        super().__init__()
        self.credentials = credentials or OpenAICredentials.from_env()
        self.client = OpenAI(
            api_key=self.credentials.openai_api_key.get_secret_value(),
            organization=self.credentials.openai_organization_id,
        )

    def process(self, input_data: OpenAIParserInput) -> OpenAIParserOutput:
        try:
            # Build messages list including conversation history
            messages = [{"role": "system", "content": input_data.system_prompt}]

            # Add conversation history if present
            if input_data.conversation_history:
                messages.extend(input_data.conversation_history)

            # Add current user input
            messages.append({"role": "user", "content": input_data.user_input})

            # Use parse method with conversation history
            completion = self.client.beta.chat.completions.parse(
                model=input_data.model,
                messages=messages,
                response_format=input_data.response_model,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
            )

            if completion.choices[0].message.parsed is None:
                raise ProcessingError("Failed to parse response")

            return OpenAIParserOutput(
                parsed_response=completion.choices[0].message.parsed,
                used_model=completion.model,
                tokens_used=completion.usage.total_tokens,
            )

        except Exception as e:
            raise ProcessingError(f"OpenAI parsing failed: {str(e)}")


# Example Response Models
class PersonInfo(BaseModel):
    """Example response model for person information"""

    name: str
    age: int
    occupation: str
    skills: List[str]
    contact: Dict[str, str]


class MovieReview(BaseModel):
    """Example response model for movie review"""

    title: str
    year: int
    rating: float
    genre: List[str]
    review: str
    pros: List[str]
    cons: List[str]


class UserProfile(BaseModel):
    """Model for tracking user profile information through conversation"""

    name: str
    age: int
    last_topic_discussed: str


def main():
    # Initialize the parser skill
    skill = OpenAIParserSkill()
    conversation_history = []

    # First interaction
    input_data = OpenAIParserInput(
        user_input="Hi, I'm Alice and I love reading science fiction books.",
        system_prompt="You are an assistant that builds user profiles through conversation. Extract and update information about the user.",
        model="gpt-4o",
        response_model=UserProfile,
        conversation_history=conversation_history,
    )

    try:
        result = skill.process(input_data)
        print("\nInitial Profile:")
        print(result.parsed_response.model_dump_json(indent=2))

        # Add to conversation history
        conversation_history.extend(
            [
                {"role": "user", "content": input_data.user_input},
                {
                    "role": "assistant",
                    "content": str(result.parsed_response.model_dump()),
                },
            ]
        )

        # Second interaction
        input_data = OpenAIParserInput(
            user_input="I'm 28 years old and also enjoy hiking on weekends.",
            system_prompt="Continue building the user profile based on new information.",
            model="gpt-4o",
            response_model=UserProfile,
            conversation_history=conversation_history,
        )

        result = skill.process(input_data)
        print("\nUpdated Profile:")
        print(result.parsed_response.model_dump_json(indent=2))

        # Add to conversation history
        conversation_history.extend(
            [
                {"role": "user", "content": input_data.user_input},
                {
                    "role": "assistant",
                    "content": result.parsed_response.model_dump_json(),
                },
            ]
        )

        # Third interaction
        input_data = OpenAIParserInput(
            user_input="I prefer reading on my Kindle and usually go hiking in national parks.",
            system_prompt="Update the user profile with any new preferences or details.",
            model="gpt-4o",
            response_model=UserProfile,
            conversation_history=conversation_history,
        )

        result = skill.process(input_data)
        print("\nFinal Profile:")
        print(result.parsed_response.model_dump_json(indent=2))
        print(f"\nTokens Used: {result.tokens_used}")

    except Exception as e:
        print(f"Error: {e}")


# Usage example
if __name__ == "__main__":
    main()
