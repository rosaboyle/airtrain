"""Tests for Together AI integration."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import time


# Create a mock together module to prevent import errors
class MockTogether:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock()
        self.images = MagicMock()
        self.images.generate = MagicMock()
        self.rerank = MagicMock()
        self.rerank.create = MagicMock()


# Mock Models class with list method
class MockModels:
    @classmethod
    async def list(cls):
        return ["model1", "model2"]


# Create mock together module
mock_together = MagicMock()
mock_together.Together = MockTogether
mock_together.Models = MockModels
mock_together.api_key = None

# Add to sys.modules
sys.modules["together"] = mock_together

from pydantic import SecretStr
from airtrain.integrations.together.credentials import TogetherAICredentials
from airtrain.integrations.together.skills import (
    TogetherAIChatSkill,
    TogetherAIInput,
    TogetherAIOutput,
    TogetherAIImageSkill,
)
from airtrain.integrations.together.models import (
    TogetherAIImageInput,
    TogetherAIImageOutput,
    GeneratedImage,
)
from airtrain.integrations.together.rerank_skill import TogetherAIRerankSkill
from airtrain.integrations.together.schemas import (
    TogetherAIRerankInput,
    TogetherAIRerankOutput,
    RerankResult,
)
from airtrain.core.credentials import CredentialValidationError
from airtrain.core.skills import ProcessingError


@pytest.fixture
def mock_api_key():
    """Return a mock API key."""
    return "mock-together-api-key-12345"


@pytest.fixture
def mock_env_vars(monkeypatch, mock_api_key):
    """Set up mock environment variables."""
    monkeypatch.setenv("TOGETHER_API_KEY", mock_api_key)


@pytest.fixture
def mock_credentials(mock_api_key):
    """Create mock credentials."""
    return TogetherAICredentials(together_api_key=SecretStr(mock_api_key))


@pytest.fixture
def mock_documents():
    """Return a list of documents for reranking tests."""
    return [
        "Regular exercise improves cardiovascular health.",
        "A balanced diet is essential for health.",
        "Exercise helps maintain mental health and reduce stress.",
    ]


class TestTogetherAI:
    """Test Together AI integration."""

    def test_credentials_init(self, mock_api_key):
        """Test initializing credentials directly."""
        credentials = TogetherAICredentials(together_api_key=mock_api_key)
        assert credentials.together_api_key.get_secret_value() == mock_api_key

    def test_credentials_from_env(self, mock_env_vars, mock_api_key):
        """Test initializing credentials from environment."""
        credentials = TogetherAICredentials.from_env()
        assert credentials.together_api_key.get_secret_value() == mock_api_key

    @patch("airtrain.integrations.together.credentials.together.Models.list")
    async def test_credentials_validate(self, mock_list, mock_credentials):
        """Test validating credentials."""
        mock_list.return_value = ["model1", "model2"]

        assert await mock_credentials.validate_credentials() is True

    def test_chat_skill_init(self, mock_credentials):
        """Test initializing chat skill."""
        with patch(
            "airtrain.integrations.together.skills.Together"
        ) as mock_together_cls:
            mock_together = MagicMock()
            mock_together_cls.return_value = mock_together

            skill = TogetherAIChatSkill(credentials=mock_credentials)

            assert skill.credentials == mock_credentials
            assert skill.client == mock_together
            mock_together_cls.assert_called_once_with(
                api_key=mock_credentials.together_api_key.get_secret_value()
            )

    def test_chat_process(self, mock_credentials):
        """Test processing a chat request."""
        # Setup mocks
        with patch(
            "airtrain.integrations.together.skills.Together"
        ) as mock_together_cls:
            mock_client = MagicMock()
            mock_together_cls.return_value = mock_client

            # Setup mock response
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message = MagicMock()
            mock_completion.choices[0].message.content = "Test response"
            mock_completion.usage = MagicMock()
            mock_completion.usage.model_dump.return_value = {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
            mock_client.chat.completions.create.return_value = mock_completion

            # Create the skill and input
            skill = TogetherAIChatSkill(credentials=mock_credentials)
            input_data = TogetherAIInput(
                user_input="Test input",
                system_prompt="Test system prompt",
                model="deepseek-ai/DeepSeek-R1",
                temperature=0.7,
                max_tokens=1024,
                stream=False,
            )

            # Call the method
            result = skill.process(input_data)

            # Verify results
            assert isinstance(result, TogetherAIOutput)
            assert result.response == "Test response"
            assert result.used_model == input_data.model
            assert "prompt_tokens" in result.usage

            # Verify the API was called
            mock_client.chat.completions.create.assert_called_once()

    def test_chat_error_handling(self, mock_credentials):
        """Test error handling in chat."""
        with patch(
            "airtrain.integrations.together.skills.Together"
        ) as mock_together_cls:
            mock_client = MagicMock()
            mock_together_cls.return_value = mock_client

            # Setup mock to raise an exception
            mock_client.chat.completions.create.side_effect = Exception("API Error")

            # Create the skill and input
            skill = TogetherAIChatSkill(credentials=mock_credentials)
            input_data = TogetherAIInput(
                user_input="Test input",
                system_prompt="Test system prompt",
                model="deepseek-ai/DeepSeek-R1",
            )

            # Verify exception is caught and wrapped
            with pytest.raises(ProcessingError) as exc_info:
                skill.process(input_data)

            assert "Together AI processing failed" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)

    def test_chat_with_streaming(self, mock_credentials):
        """Test chat with streaming enabled."""
        with patch(
            "airtrain.integrations.together.skills.Together"
        ) as mock_together_cls:
            mock_client = MagicMock()
            mock_together_cls.return_value = mock_client

            # Setup streaming response
            mock_chunks = []
            for i in range(3):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = f"chunk {i}"
                mock_chunks.append(chunk)

            # Add final chunk with no content
            final_chunk = MagicMock()
            final_chunk.choices = [MagicMock()]
            final_chunk.choices[0].delta = MagicMock()
            final_chunk.choices[0].delta.content = None
            mock_chunks.append(final_chunk)

            mock_client.chat.completions.create.return_value = mock_chunks

            # Create skill and input with streaming
            skill = TogetherAIChatSkill(credentials=mock_credentials)
            input_data = TogetherAIInput(
                user_input="Test streaming",
                model="deepseek-ai/DeepSeek-R1",
                stream=True,
            )

            # Collect streaming results
            stream_results = list(skill.process_stream(input_data))

            # Verify results
            assert len(stream_results) == 3
            assert stream_results[0] == "chunk 0"
            assert stream_results[1] == "chunk 1"
            assert stream_results[2] == "chunk 2"

            # Verify method was called with streaming
            mock_client.chat.completions.create.assert_called_once()
            call_args, call_kwargs = mock_client.chat.completions.create.call_args
            assert call_kwargs["stream"] is True

    def test_image_generation(self, mock_credentials):
        """Test image generation."""
        with patch(
            "airtrain.integrations.together.skills.Together"
        ) as mock_together_cls:
            mock_client = MagicMock()
            mock_together_cls.return_value = mock_client

            # Setup image generation response
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(
                    b64_json="base64_image_data", seed=12345, finish_reason="success"
                )
            ]
            mock_client.images.generate.return_value = mock_response

            # Create skill and input
            skill = TogetherAIImageSkill(credentials=mock_credentials)
            input_data = TogetherAIImageInput(
                prompt="A beautiful mountain scene",
                model="black-forest-labs/FLUX.1-schnell-Free",
                steps=10,
                n=1,
                size="1024x1024",
            )

            # Mock time.time() for duration calculation
            with patch("time.time") as mock_time:
                mock_time.side_effect = [100.0, 102.5]  # Start and end times

                # Generate image
                result = skill.process(input_data)

                # Verify results
                assert isinstance(result, TogetherAIImageOutput)
                assert result.model == input_data.model
                assert result.prompt == input_data.prompt
                assert result.total_time == 2.5  # 102.5 - 100.0
                assert len(result.images) == 1
                assert result.images[0].b64_json == "base64_image_data"
                assert result.images[0].seed == 12345

                # Verify API call
                mock_client.images.generate.assert_called_once()
                call_args, call_kwargs = mock_client.images.generate.call_args
                assert call_kwargs["prompt"] == input_data.prompt
                assert call_kwargs["model"] == input_data.model
                assert call_kwargs["steps"] == input_data.steps
                assert call_kwargs["n"] == input_data.n
                assert call_kwargs["size"] == input_data.size

    def test_reranking(self, mock_credentials, mock_documents):
        """Test document reranking."""
        with patch(
            "airtrain.integrations.together.skills.Together"
        ) as mock_together_cls:
            mock_client = MagicMock()
            mock_together_cls.return_value = mock_client

            # Setup rerank response
            mock_response = MagicMock()
            mock_response.results = [
                MagicMock(index=0, relevance_score=0.95),
                MagicMock(index=2, relevance_score=0.85),
                MagicMock(index=1, relevance_score=0.75),
            ]
            mock_client.rerank.create.return_value = mock_response

            # Create skill and input
            with patch(
                "airtrain.integrations.together.rerank_skill.get_rerank_model_config"
            ) as mock_get_config:
                mock_get_config.return_value = MagicMock()

                skill = TogetherAIRerankSkill(credentials=mock_credentials)
                input_data = TogetherAIRerankInput(
                    query="What are the health benefits of exercise?",
                    documents=mock_documents,
                    model="Salesforce/Llama-Rank-v1",
                    top_n=2,
                )

                # Rerank documents
                result = skill.process(input_data)

                # Verify results
                assert isinstance(result, TogetherAIRerankOutput)
                assert result.used_model == input_data.model
                assert len(result.results) == 3

                # Check ordering by relevance score
                assert result.results[0].index == 0
                assert result.results[0].relevance_score == 0.95
                assert result.results[0].document == mock_documents[0]

                assert result.results[1].index == 2
                assert result.results[1].relevance_score == 0.85
                assert result.results[1].document == mock_documents[2]

                # Verify API call
                mock_client.rerank.create.assert_called_once()
                call_args, call_kwargs = mock_client.rerank.create.call_args
                assert call_kwargs["model"] == input_data.model
                assert call_kwargs["query"] == input_data.query
                assert call_kwargs["documents"] == mock_documents
                assert call_kwargs["top_n"] == 2
