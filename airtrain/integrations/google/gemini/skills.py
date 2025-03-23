from typing import List, Optional, Dict, Any, Generator, Type, TypeVar, Generic, Union
from pydantic import Field, BaseModel
from google import genai
from loguru import logger
from PIL import Image
from pathlib import Path
import json

from airtrain.core.skills import Skill, ProcessingError
from airtrain.core.schemas import InputSchema, OutputSchema
from .credentials import Gemini2Credentials

T = TypeVar("T", bound=BaseModel)


class Gemini2GenerationConfig(InputSchema):
    """Schema for Gemini 2.0 generation config"""

    temperature: float = Field(
        default=1.0, description="Temperature for response generation", ge=0, le=1
    )
    response_mime_type: str = Field(
        default="text/plain", description="Response MIME type"
    )
    response_model: Optional[Type[BaseModel]] = Field(
        default=None, description="Optional Pydantic model for structured output"
    )

    class Config:
        arbitrary_types_allowed = True


class Gemini2Input(InputSchema, Generic[T]):
    """Schema for Gemini 2.0 chat input"""

    user_input: str = Field(..., description="User's input text")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to guide the model's behavior",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of conversation messages",
    )
    model: str = Field(default="gemini-2.0-flash", description="Gemini model to use")
    generation_config: Gemini2GenerationConfig = Field(
        default_factory=Gemini2GenerationConfig,
        description="Generation configuration",
    )
    images: List[Path] = Field(
        default_factory=list,
        description="List of image paths to include in the message",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )

    class Config:
        arbitrary_types_allowed = True


class Gemini2Output(OutputSchema, Generic[T]):
    """Schema for Gemini 2.0 chat output"""

    response: str = Field(..., description="Model's response text")
    used_model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    parsed_response: Optional[T] = Field(
        default=None, description="Parsed structured response if applicable"
    )


class Gemini2ChatSkill(Skill[Gemini2Input[T], Gemini2Output[T]], Generic[T]):
    """Skill for Gemini 2.0 chat"""

    input_schema = Gemini2Input
    output_schema = Gemini2Output

    def __init__(self, credentials: Optional[Gemini2Credentials] = None):
        super().__init__()
        self.credentials = credentials or Gemini2Credentials.from_env()
        self.client = genai.Client(
            api_key=self.credentials.gemini_api_key.get_secret_value()
        )

    def _convert_history_format(
        self, history: List[Dict[str, str]]
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        """Convert standard history format to Google's format"""
        google_history = []
        for msg in history:
            google_msg = {
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}],
            }
            google_history.append(google_msg)
        return google_history

    def _prepare_contents(self, input_data: Gemini2Input[T]) -> List[str | Image.Image]:
        contents = []
        if input_data.system_prompt:
            contents.append(input_data.system_prompt)
        for image_path in input_data.images:
            image = Image.open(image_path)
            contents.append(image)
        contents.append(input_data.user_input)
        return contents

    def process(self, input_data: Gemini2Input[T]) -> Gemini2Output[T]:
        try:
            contents = self._prepare_contents(input_data)
            config = {}

            if input_data.generation_config.response_model:
                config = {
                    "response_mime_type": "application/json",
                    "response_schema": input_data.generation_config.response_model,
                }

            response = self.client.models.generate_content(
                model=input_data.model,
                contents=contents,
                config=config,
            )

            parsed_response = None
            if input_data.generation_config.response_model:
                parsed_response = response.parsed

            return Gemini2Output(
                response=response.text,
                used_model=input_data.model,
                usage={},  # Gemini API doesn't provide usage stats
                parsed_response=parsed_response,
            )

        except Exception as e:
            logger.exception(f"Gemini 2.0 processing failed: {str(e)}")
            raise ProcessingError(f"Gemini 2.0 processing failed: {str(e)}")

    def process_stream(self, input_data: Gemini2Input[T]) -> Generator[str, None, None]:
        """Process the input and stream the response token by token."""
        try:
            contents = self._prepare_contents(input_data)
            response = self.client.models.generate_content_stream(
                model=input_data.model,
                contents=contents,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.exception(f"Gemini 2.0 streaming failed: {str(e)}")
            raise ProcessingError(f"Gemini 2.0 streaming failed: {str(e)}")
