from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import os
from typing import Type, TypeVar, Optional, List, Dict, Any, Iterable
import dotenv
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)

dotenv.load_dotenv()

T = TypeVar("T", bound=BaseModel)


class ChatMessageUser(BaseModel):
    content: str
    """The contents of the user message."""

    role: str
    """The role of the messages author, in this case `user`."""


class OpenAIService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI service with optional API key.
        If no key provided, tries to get from environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in environment")

        self.sync_client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    def parse_input(
        self,
        system_content: str,
        user_content: str,
        response_format: Type[T],
        model: str = "gpt-4o-2024-08-06",
    ) -> T:
        """
        Generates a response from OpenAI based on the given inputs and model.
        """
        completion = self.sync_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=response_format,
        )

        if completion.choices[0].message.parsed is None:
            raise ValueError("Failed to parse response.")

        return completion.choices[0].message.parsed

    async def parse_input_async(
        self,
        system_content: str,
        user_content: str,
        response_format: Type[T],
        model: str = "gpt-4o-2024-08-06",
    ) -> T:
        """
        Generates a response from OpenAI based on the given\
 inputs and model asynchronously.
        """
        completion = await self.async_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=response_format,
        )

        if completion.choices[0].message.parsed is None:
            raise ValueError("Failed to parse response.")

        return completion.choices[0].message.parsed

    def get_text_response(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o",
    ) -> str:
        """
        Generates a text response using OpenAI's chat completions API.
        """
        response = self.sync_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response.")

        return response.choices[0].message.content

    async def get_text_response_async(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "gpt-4o",
    ) -> str:
        """
        Generates a text response using OpenAI's chat completions API asynchronously.
        """
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response.")

        return response.choices[0].message.content

    def get_sequence_response(
        self,
        messages: (
            ChatMessageUser
            | Iterable[
                ChatCompletionSystemMessageParam
                | ChatCompletionUserMessageParam
                | ChatCompletionAssistantMessageParam
            ]
        ),
        model: str = "gpt-4o",
    ) -> list[ChatMessageUser]:
        """
        Generates a sequence response using OpenAI's chat completions API.
        """
        response = self.sync_client.chat.completions.create(
            model=model, messages=messages  # type: ignore
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response.")

        return [
            *messages,
            ChatMessageUser(
                content=response.choices[0].message.content, role="assistant"
            ),
        ]

    def analyze_image(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str = "You are a helpful assistant",
        model: str = "gpt-4o",
    ) -> str:
        """
        Analyzes an image using OpenAI's vision model.
        """
        response = self.sync_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=131072,
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate image analysis response.")

        return response.choices[0].message.content

    async def analyze_image_async(
        self,
        prompt: str,
        image_url: str,
        system_prompt: str = "You are a helpful assistant",
        model: str = "gpt-4o",
    ) -> str:
        """
        Analyzes an image using OpenAI's vision model asynchronously.
        """
        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=131072,
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate image analysis response.")

        return response.choices[0].message.content

    async def analyze_multiple_images_async(
        self,
        prompt: str,
        image_urls: list[str],
        system_prompt: str = "You are a helpful assistant",
        model: str = "gpt-4o",
    ) -> str:
        """
        Analyzes multiple images using OpenAI's vision model asynchronously.
        """
        image_urls_content = [
            {"type": "image_url", "image_url": {"url": image_url}}
            for image_url in image_urls
        ]

        response = await self.async_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        *image_urls_content,
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=131072,
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate image analysis response.")

        return response.choices[0].message.content

    def get_conversation_response(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
    ) -> str:
        """
        Generates a response using the full conversation history.
        """
        response = self.sync_client.chat.completions.create(
            model=model, messages=messages
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response")

        return response.choices[0].message.content

    async def get_conversation_response_async(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
    ) -> str:
        """
        Generates a response using the full conversation history asynchronously.
        """
        response = await self.async_client.chat.completions.create(
            model=model, messages=messages
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response")

        return response.choices[0].message.content

    def get_conversation_with_images_response(
        self,
        messages: List[Dict[str, Any]],
        image_urls: List[str],
        model: str = "gpt-4o",
    ) -> str:
        """
        Generates a response using conversation history and images.
        """
        processed_messages = messages[:-1]
        last_message = messages[-1]

        image_contents = [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]

        last_message_content = [
            *image_contents,
            {"type": "text", "text": last_message["content"]},
        ]

        processed_messages.append({"role": "user", "content": last_message_content})

        response = self.sync_client.chat.completions.create(
            model=model, messages=processed_messages, max_tokens=131072
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response")

        return response.choices[0].message.content

    async def get_conversation_with_images_response_async(
        self,
        messages: List[Dict[str, Any]],
        image_urls: List[str],
        model: str = "gpt-4o",
    ) -> str:
        """
        Generates a response using conversation history and images asynchronously.
        """
        processed_messages = messages[:-1]
        last_message = messages[-1]

        image_contents = [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ]

        last_message_content = [
            *image_contents,
            {"type": "text", "text": last_message["content"]},
        ]

        processed_messages.append({"role": "user", "content": last_message_content})

        response = await self.async_client.chat.completions.create(
            model=model, messages=processed_messages, max_tokens=131072
        )

        if not response.choices[0].message.content:
            raise ValueError("Failed to generate response")

        return response.choices[0].message.content
