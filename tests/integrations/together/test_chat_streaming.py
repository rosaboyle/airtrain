import pytest
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any, Generator

from airtrain.integrations.together.skills import (
    TogetherAIChatSkill,
    TogetherAIInput,
    TogetherAIOutput,
)
from airtrain.core.skills import ProcessingError

from .debug_helpers import logger, debug_streaming_chunks


class TestTogetherAIChatStreaming:
    """Test cases for TogetherAIChatSkill streaming functionality."""

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
        """Create a sample input for testing with stream=True."""
        return TogetherAIInput(
            user_input="Test streaming input",
            system_prompt="Test system prompt",
            model="deepseek-ai/DeepSeek-R1",
            temperature=0.7,
            max_tokens=1024,
            stream=True,
        )

    @pytest.fixture
    def mock_stream_chunks(self):
        """Create mock streaming response chunks."""
        chunks = []
        for i in range(5):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = f"chunk{i} "
            chunks.append(chunk)

        # Add a final chunk with None content to simulate stream end
        final_chunk = MagicMock()
        final_chunk.choices = [MagicMock()]
        final_chunk.choices[0].delta = MagicMock()
        final_chunk.choices[0].delta.content = None
        final_chunk.choices[0].finish_reason = "stop"
        chunks.append(final_chunk)

        return chunks

    def test_stream_success(self, skill, input_data, mock_stream_chunks):
        """Test process_stream method with successful stream response."""
        # Setup mock client to return the stream chunks
        skill.client.chat.completions.create.return_value = mock_stream_chunks

        # Debug the mock chunks
        logger.info(f"Testing with {len(mock_stream_chunks)} chunks")
        debug_streaming_chunks(mock_stream_chunks)

        # Collect results from the generator
        logger.info("Collecting results from the generator")
        results = list(skill.process_stream(input_data))
        logger.info(f"Number of results collected: {len(results)}")

        # Verify API call
        skill.client.chat.completions.create.assert_called_once()
        call_args, call_kwargs = skill.client.chat.completions.create.call_args
        assert call_kwargs["model"] == input_data.model
        assert call_kwargs["stream"] is True

        # Verify results
        assert (
            len(results) == 5
        )  # 5 chunks with content (excluding the final None chunk)
        assert all(isinstance(r, str) for r in results)
        assert "".join(results) == "chunk0 chunk1 chunk2 chunk3 chunk4 "

    def test_stream_error_handling(self, skill, input_data):
        """Test error handling in streaming."""
        # Set up the client to raise an exception
        skill.client.chat.completions.create.side_effect = Exception("Stream API Error")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            list(skill.process_stream(input_data))

        assert "Together AI streaming failed" in str(exc_info.value)
        assert "Stream API Error" in str(exc_info.value)

    def test_stream_empty_response(self, skill, input_data):
        """Test handling empty stream response."""
        # Set up the client to return empty stream
        skill.client.chat.completions.create.return_value = []

        # Call the method and collect results
        results = list(skill.process_stream(input_data))

        # Verify empty results
        assert len(results) == 0

        # Verify API call
        skill.client.chat.completions.create.assert_called_once()

    def test_stream_null_content_handling(self, skill, input_data):
        """Test handling of null content in stream chunks."""
        # Create chunks with some having None content
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="valid content "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="more content "))]),
        ]
        skill.client.chat.completions.create.return_value = chunks

        # Call the method and collect results
        results = list(skill.process_stream(input_data))

        # Verify results only include valid content
        assert len(results) == 2
        assert results[0] == "valid content "
        assert results[1] == "more content "

    def test_process_with_stream_integration(
        self, skill, input_data, mock_stream_chunks
    ):
        """Test process method with streaming enabled."""
        # Setup mock client to return the stream chunks
        skill.client.chat.completions.create.return_value = mock_stream_chunks

        # Call the process method which should use process_stream under the hood
        result = skill.process(input_data)

        # Verify results
        assert isinstance(result, TogetherAIOutput)
        assert result.used_model == input_data.model
        assert result.response == "chunk0 chunk1 chunk2 chunk3 chunk4 "

        # Verify API call was made with streaming
        skill.client.chat.completions.create.assert_called_once()
        call_args, call_kwargs = skill.client.chat.completions.create.call_args
        assert call_kwargs["stream"] is True
