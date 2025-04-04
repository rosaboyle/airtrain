from typing import List, Optional, Dict, Any, Generator
from pydantic import Field
from cerebras.cloud.sdk import Cerebras
from loguru import logger

from airtrain.core.skills import Skill, ProcessingError
from airtrain.core.schemas import InputSchema, OutputSchema
from .credentials import CerebrasCredentials


class CerebrasInput(InputSchema):
    """Schema for Cerebras chat input"""

    user_input: str = Field(..., description="User's input text")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt to guide the model's behavior",
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of previous conversation messages in [{'role': 'user|assistant', 'content': 'message'}] format",
    )
    model: str = Field(default="llama3.1-8b", description="Cerebras model to use")
    max_tokens: Optional[int] = Field(
        default=131072, description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7, description="Temperature for response generation", ge=0, le=1
    )
    stream: bool = Field(
        default=False, description="Whether to stream the response progressively"
    )


class CerebrasOutput(OutputSchema):
    """Schema for Cerebras chat output"""

    response: str = Field(..., description="Model's response text")
    used_model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")


class CerebrasChatSkill(Skill[CerebrasInput, CerebrasOutput]):
    """Skill for Cerebras chat"""

    input_schema = CerebrasInput
    output_schema = CerebrasOutput

    def __init__(self, credentials: Optional[CerebrasCredentials] = None):
        super().__init__()
        self.credentials = credentials or CerebrasCredentials.from_env()
        self.client = Cerebras(
            api_key=self.credentials.cerebras_api_key.get_secret_value()
        )

    def _build_messages(self, input_data: CerebrasInput) -> List[Dict[str, str]]:
        """
        Build messages list from input data including conversation history.

        Args:
            input_data: The input data containing system prompt, conversation history, and user input

        Returns:
            List[Dict[str, str]]: List of messages in the format required by Cerebras
        """
        messages = [{"role": "system", "content": input_data.system_prompt}]

        # Add conversation history if present
        if input_data.conversation_history:
            messages.extend(input_data.conversation_history)

        # Add current user input
        messages.append({"role": "user", "content": input_data.user_input})

        return messages

    def process_stream(self, input_data: CerebrasInput) -> Generator[str, None, None]:
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
            logger.exception(f"Cerebras streaming failed: {str(e)}")
            raise ProcessingError(f"Cerebras streaming failed: {str(e)}")

    def process(self, input_data: CerebrasInput) -> CerebrasOutput:
        """Process the input and return the complete response."""
        try:
            if input_data.stream:
                response_chunks = []
                for chunk in self.process_stream(input_data):
                    response_chunks.append(chunk)
                response = "".join(response_chunks)
                usage = {}  # Usage stats not available in streaming
            else:
                messages = self._build_messages(input_data)
                response = self.client.chat.completions.create(
                    model=input_data.model,
                    messages=messages,
                    temperature=input_data.temperature,
                    max_tokens=input_data.max_tokens,
                )
                usage = (
                    response.usage.model_dump() if hasattr(response, "usage") else {}
                )

            return CerebrasOutput(
                response=response.choices[0].message.content,
                used_model=input_data.model,
                usage=usage,
            )

        except Exception as e:
            logger.exception(f"Cerebras processing failed: {str(e)}")
            raise ProcessingError(f"Cerebras processing failed: {str(e)}")
