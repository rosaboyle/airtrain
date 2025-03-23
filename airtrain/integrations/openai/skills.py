from typing import AsyncGenerator, List, Optional, Dict, TypeVar, Type, Generator, Union
from pydantic import Field, BaseModel
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionChunk
import numpy as np

from airtrain.core.skills import Skill, ProcessingError
from airtrain.core.schemas import InputSchema, OutputSchema
from .credentials import OpenAICredentials


class OpenAIInput(InputSchema):
    """Schema for OpenAI chat input"""

    user_input: str = Field(..., description="User's input text")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to guide the model's behavior",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of previous conversation messages in [{'role': 'user|assistant', 'content': 'message'}] format",
    )
    model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use",
    )
    temperature: float = Field(
        default=0.7, description="Temperature for response generation", ge=0, le=1
    )
    max_tokens: Optional[int] = Field(
        default=131072, description="Maximum tokens in response"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response token by token",
    )


class OpenAIOutput(OutputSchema):
    """Schema for OpenAI chat output"""

    response: str
    used_model: str
    usage: Dict[str, int]


class OpenAIChatSkill(Skill[OpenAIInput, OpenAIOutput]):
    """Skill for interacting with OpenAI models with async support"""

    input_schema = OpenAIInput
    output_schema = OpenAIOutput

    def __init__(self, credentials: Optional[OpenAICredentials] = None):
        """Initialize the skill with optional credentials"""
        super().__init__()
        self.credentials = credentials or OpenAICredentials.from_env()
        self.client = OpenAI(
            api_key=self.credentials.openai_api_key.get_secret_value(),
            organization=self.credentials.openai_organization_id,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.credentials.openai_api_key.get_secret_value(),
            organization=self.credentials.openai_organization_id,
        )

    def _build_messages(self, input_data: OpenAIInput) -> List[Dict[str, str]]:
        """Build messages list from input data including conversation history."""
        messages = [{"role": "system", "content": input_data.system_prompt}]

        if input_data.conversation_history:
            messages.extend(input_data.conversation_history)

        messages.append({"role": "user", "content": input_data.user_input})
        return messages

    def process_stream(self, input_data: OpenAIInput) -> Generator[str, None, None]:
        """Process the input and stream the response token by token."""
        try:
            messages = self._build_messages(input_data)

            stream = self.client.chat.completions.create(
                model=input_data.model,
                messages=messages,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ProcessingError(f"OpenAI streaming failed: {str(e)}")

    def process(self, input_data: OpenAIInput) -> OpenAIOutput:
        """Process the input and return the complete response."""
        try:
            if input_data.stream:
                # For streaming, collect the entire response
                response_chunks = []
                for chunk in self.process_stream(input_data):
                    response_chunks.append(chunk)
                response = "".join(response_chunks)
            else:
                # For non-streaming, use regular completion
                messages = self._build_messages(input_data)
                completion = self.client.chat.completions.create(
                    model=input_data.model,
                    messages=messages,
                    temperature=input_data.temperature,
                    max_tokens=input_data.max_tokens,
                    stream=False,
                )
                response = completion.choices[0].message.content

            return OpenAIOutput(
                response=response,
                used_model=input_data.model,
                usage={
                    "total_tokens": completion.usage.total_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                },
            )

        except Exception as e:
            raise ProcessingError(f"OpenAI chat failed: {str(e)}")

    async def process_async(self, input_data: OpenAIInput) -> OpenAIOutput:
        """Async version of process method"""
        try:
            messages = self._build_messages(input_data)
            completion = await self.async_client.chat.completions.create(
                model=input_data.model,
                messages=messages,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
            )
            return OpenAIOutput(
                response=completion.choices[0].message.content,
                used_model=completion.model,
                usage={
                    "total_tokens": completion.usage.total_tokens,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                },
            )
        except Exception as e:
            raise ProcessingError(f"OpenAI async chat failed: {str(e)}")

    async def process_stream_async(
        self, input_data: OpenAIInput
    ) -> AsyncGenerator[str, None]:
        """Async version of stream processor"""
        try:
            messages = self._build_messages(input_data)
            stream = await self.async_client.chat.completions.create(
                model=input_data.model,
                messages=messages,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise ProcessingError(f"OpenAI async streaming failed: {str(e)}")


ResponseT = TypeVar("ResponseT", bound=BaseModel)


class OpenAIParserInput(InputSchema):
    """Schema for OpenAI structured output input"""

    user_input: str
    system_prompt: str = "You are a helpful assistant that provides structured data."
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    response_model: Type[ResponseT]

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

    def __init__(self, credentials: Optional[OpenAICredentials] = None):
        """Initialize the skill with optional credentials"""
        super().__init__()
        self.credentials = credentials or OpenAICredentials.from_env()
        self.client = OpenAI(
            api_key=self.credentials.openai_api_key.get_secret_value(),
            organization=self.credentials.openai_organization_id,
        )

    def process(self, input_data: OpenAIParserInput) -> OpenAIParserOutput:
        try:
            # Use parse method instead of create
            completion = self.client.beta.chat.completions.parse(
                model=input_data.model,
                messages=[
                    {"role": "system", "content": input_data.system_prompt},
                    {"role": "user", "content": input_data.user_input},
                ],
                response_format=input_data.response_model,
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


class OpenAIEmbeddingsInput(InputSchema):
    """Schema for OpenAI embeddings input"""

    texts: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to generate embeddings for"
    )
    model: str = Field(
        default="text-embedding-3-large", description="OpenAI embeddings model to use"
    )
    encoding_format: str = Field(
        default="float", description="The format of the embeddings: 'float' or 'base64'"
    )
    dimensions: Optional[int] = Field(
        default=None, description="Optional number of dimensions for the embeddings"
    )


class OpenAIEmbeddingsOutput(OutputSchema):
    """Schema for OpenAI embeddings output"""

    embeddings: List[List[float]] = Field(..., description="List of embeddings vectors")
    used_model: str = Field(..., description="Model used for generating embeddings")
    tokens_used: int = Field(..., description="Number of tokens used")


class OpenAIEmbeddingsSkill(Skill[OpenAIEmbeddingsInput, OpenAIEmbeddingsOutput]):
    """Skill for generating embeddings using OpenAI models"""

    input_schema = OpenAIEmbeddingsInput
    output_schema = OpenAIEmbeddingsOutput

    def __init__(self, credentials: Optional[OpenAICredentials] = None):
        """Initialize the skill with optional credentials"""
        super().__init__()
        self.credentials = credentials or OpenAICredentials.from_env()
        self.client = OpenAI(
            api_key=self.credentials.openai_api_key.get_secret_value(),
            organization=self.credentials.openai_organization_id,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.credentials.openai_api_key.get_secret_value(),
            organization=self.credentials.openai_organization_id,
        )

    def process(self, input_data: OpenAIEmbeddingsInput) -> OpenAIEmbeddingsOutput:
        """Generate embeddings for the input text(s)"""
        try:
            # Handle single text input
            texts = (
                [input_data.texts]
                if isinstance(input_data.texts, str)
                else input_data.texts
            )

            # Create embeddings
            response = self.client.embeddings.create(
                model=input_data.model,
                input=texts,
                encoding_format=input_data.encoding_format,
                dimensions=input_data.dimensions,
            )

            # Extract embeddings
            embeddings = [data.embedding for data in response.data]

            return OpenAIEmbeddingsOutput(
                embeddings=embeddings,
                used_model=response.model,
                tokens_used=response.usage.total_tokens,
            )
        except Exception as e:
            raise ProcessingError(f"OpenAI embeddings generation failed: {str(e)}")

    async def process_async(
        self, input_data: OpenAIEmbeddingsInput
    ) -> OpenAIEmbeddingsOutput:
        """Async version of the embeddings generation"""
        try:
            # Handle single text input
            texts = (
                [input_data.texts]
                if isinstance(input_data.texts, str)
                else input_data.texts
            )

            # Create embeddings
            response = await self.async_client.embeddings.create(
                model=input_data.model,
                input=texts,
                encoding_format=input_data.encoding_format,
                dimensions=input_data.dimensions,
            )

            # Extract embeddings
            embeddings = [data.embedding for data in response.data]

            return OpenAIEmbeddingsOutput(
                embeddings=embeddings,
                used_model=response.model,
                tokens_used=response.usage.total_tokens,
            )
        except Exception as e:
            raise ProcessingError(
                f"OpenAI async embeddings generation failed: {str(e)}"
            )
