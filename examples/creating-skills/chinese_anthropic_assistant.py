from typing import Optional, TypeVar
from pydantic import Field

from airtrain.integrations.anthropic.skills import (
    AnthropicChatSkill,
    AnthropicInput,
    AnthropicOutput,
)

T = TypeVar("T", bound=AnthropicInput)


class ChineseAnthropicInput(AnthropicInput):
    """Schema for Chinese Anthropic Assistant input"""

    user_input: str = Field(
        ..., description="User's input text (can be in any language)"
    )
    system_prompt: str = Field(
        default="你是一个有帮助的中文助手。请用中文回答所有问题，即使问题是用其他语言问的。回答要准确、礼貌、专业。",
        description="System prompt in Chinese",
    )
    model: str = Field(
        default="claude-3-opus-20240229", description="Anthropic model to use"
    )
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    temperature: float = Field(
        default=0.7, description="Temperature for response generation", ge=0, le=1
    )


class ChineseAnthropicSkill(AnthropicChatSkill):
    """Skill for Chinese language assistance using Anthropic"""

    input_schema = ChineseAnthropicInput
    output_schema = AnthropicOutput

    def process(self, input_data: T) -> AnthropicOutput:
        """
        Process the input and ensure Chinese language response.

        Args:
            input_data: The input data containing user's query and settings

        Returns:
            AnthropicOutput: The model's response in Chinese

        Raises:
            ProcessingError: If processing fails
        """
        if "你是" not in input_data.system_prompt:
            input_data.system_prompt = (
                "你是一个中文助手。" + input_data.system_prompt + "请用中文回答。"
            )

        return super().process(input_data)
