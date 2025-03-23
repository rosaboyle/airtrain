import pytest
from unittest.mock import patch, Mock, MagicMock
import json
from typing import Dict, Any, List

from airtrain.integrations.together.skills import (
    TogetherAIChatSkill,
    TogetherAIInput,
    TogetherAIOutput,
)
from airtrain.core.skills import ProcessingError

from .debug_helpers import logger


class TestTogetherAIChatSkill:
    """Test cases for TogetherAIChatSkill."""

    @pytest.fixture
    def skill(self, mock_credentials):
        """Initialize the skill with mock credentials."""
        with patch("together.Together") as mock_together:
            mock_client = Mock()
            mock_together.return_value = mock_client
            skill = TogetherAIChatSkill(credentials=mock_credentials)
            skill.client = mock_client
            return skill

    @pytest.fixture
    def input_data(self):
        """Create a sample input for testing."""
        return TogetherAIInput(
            user_input="Test input",
            system_prompt="Test system prompt",
            model="deepseek-ai/DeepSeek-R1",
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )

    def test_init(self, mock_credentials):
        """Test initialization with provided credentials."""
        with patch("together.Together") as mock_together:
            mock_client = Mock()
            mock_together.return_value = mock_client

            skill = TogetherAIChatSkill(credentials=mock_credentials)
            assert skill.credentials == mock_credentials
            assert skill.client is not None
            mock_together.assert_called_once_with(
                api_key=mock_credentials.together_api_key.get_secret_value()
            )

    def test_init_from_env(self, mock_credentials_from_env):
        """Test initialization from environment variables."""
        with patch("together.Together") as mock_together:
            with patch(
                "airtrain.integrations.together.credentials.TogetherAICredentials.from_env"
            ) as mock_from_env:
                mock_from_env.return_value = mock_credentials_from_env
                mock_client = Mock()
                mock_together.return_value = mock_client

                skill = TogetherAIChatSkill()
                assert skill.credentials == mock_credentials_from_env
                assert skill.client is not None

    def test_build_messages(self, skill, input_data):
        """Test _build_messages method."""
        messages = skill._build_messages(input_data)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "Test system prompt"}
        assert messages[1] == {"role": "user", "content": "Test input"}

        # Test with conversation history
        input_data.conversation_history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]
        messages = skill._build_messages(input_data)
        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "Test system prompt"}
        assert messages[1] == {"role": "user", "content": "Previous message"}
        assert messages[2] == {"role": "assistant", "content": "Previous response"}
        assert messages[3] == {"role": "user", "content": "Test input"}

    def test_process_success(self, skill, input_data, mock_together_client):
        """Test process method with successful response."""
        # Set up the mock client's chat.completions.create method to return a successful response
        skill.client = mock_together_client

        # Call the method
        result = skill.process(input_data)

        # Verify results
        assert isinstance(result, TogetherAIOutput)
        assert result.used_model == input_data.model
        assert result.response == "This is a test response"
        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage
        assert "total_tokens" in result.usage

        # Verify the API was called with the right parameters
        skill.client.chat.completions.create.assert_called_once()
        call_args, call_kwargs = skill.client.chat.completions.create.call_args
        assert call_kwargs["model"] == input_data.model
        assert call_kwargs["temperature"] == input_data.temperature
        assert call_kwargs["max_tokens"] == input_data.max_tokens
        assert call_kwargs["stream"] is False

    def test_process_error(self, skill, input_data):
        """Test process method with API error."""
        # Set up the mock client to raise an exception
        skill.client.chat.completions.create.side_effect = Exception("API Error")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            skill.process(input_data)

        assert "Together AI processing failed" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)

    def test_process_with_stream(self, skill, input_data, mock_together_client):
        """Test process method with streaming enabled."""
        # Set stream flag
        input_data.stream = True
        skill.client = mock_together_client

        # Call the method with streaming
        result = skill.process(input_data)

        # Verify results
        assert isinstance(result, TogetherAIOutput)
        assert result.used_model == input_data.model
        assert "chunk0 chunk1 chunk2 chunk3 chunk4" in result.response

        # Verify the API was called with the right parameters
        skill.client.chat.completions.create.assert_called_once()
        call_args, call_kwargs = skill.client.chat.completions.create.call_args
        assert call_kwargs["model"] == input_data.model
        assert call_kwargs["stream"] is True

    def test_process_stream_method(self, skill, input_data, mock_together_client):
        """Test process_stream method directly."""
        # Set stream flag
        input_data.stream = True
        skill.client = mock_together_client

        # Call the streaming method directly and collect results
        chunks = list(skill.process_stream(input_data))

        # Verify chunks
        assert len(chunks) == 5
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert "".join(chunks) == "chunk0 chunk1 chunk2 chunk3 chunk4 "

        # Verify the API was called with the right parameters
        skill.client.chat.completions.create.assert_called_once()
        call_args, call_kwargs = skill.client.chat.completions.create.call_args
        assert call_kwargs["model"] == input_data.model
        assert call_kwargs["stream"] is True

    def test_process_stream_error(self, skill, input_data):
        """Test process_stream method with API error."""
        # Set stream flag
        input_data.stream = True

        # Set up the mock client to raise an exception
        skill.client.chat.completions.create.side_effect = Exception(
            "Streaming API Error"
        )

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            list(skill.process_stream(input_data))

        assert "Together AI streaming failed" in str(exc_info.value)
        assert "Streaming API Error" in str(exc_info.value)
