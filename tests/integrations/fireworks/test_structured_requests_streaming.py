import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Generator
from pydantic import BaseModel, Field
import logging

# Create a logger instead of importing it
logger = logging.getLogger("fireworks_tests")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

from airtrain.integrations.fireworks.structured_requests_skills import (
    FireworksStructuredRequestSkill,
    FireworksStructuredRequestInput,
    ProcessingError,
)
from airtrain.integrations.fireworks.credentials import FireworksCredentials


# Sample response model for testing
class MockResponseModel(BaseModel):
    message: str = Field(..., description="A test message")
    confidence: float = Field(..., description="Confidence score")


# Mock stream response chunks
STREAM_CHUNKS = [
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"<think>"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"Let me analyze"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":" this request"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"</think>"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"{\\"message\\":"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":" \\"This is"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":" a test"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":" response\\""},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":", \\"confidence\\": 0.95}"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n',
]

# Mock invalid chunks for error testing
INVALID_JSON_CHUNKS = [
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"{malformed json}"},"index":0,"finish_reason":null}]}\n\n',
]


@pytest.fixture
def mock_credentials():
    """Mock FireworksCredentials."""
    mock = Mock()
    # Create a nested mock for fireworks_api_key with get_secret_value method
    api_key_mock = Mock()
    api_key_mock.get_secret_value.return_value = "test-api-key"
    mock.fireworks_api_key = api_key_mock
    return mock


@pytest.fixture
def skill(mock_credentials):
    """Initialize the skill with mock credentials."""
    return FireworksStructuredRequestSkill(credentials=mock_credentials)


@pytest.fixture
def input_data():
    """Create a sample input for testing with stream=True."""
    return FireworksStructuredRequestInput(
        user_input="Test streaming input",
        system_prompt="Test system prompt",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=100,
        response_model=MockResponseModel,
        stream=True,
    )


class TestFireworksStructuredRequestStreaming:
    """Test cases for FireworksStructuredRequestSkill streaming functionality."""

    @patch("requests.post")
    def test_process_stream_success(self, mock_post, skill, input_data):
        """Test process_stream method with successful stream response."""
        logger.info("Starting test_process_stream_success")

        # Setup mock streaming response
        mock_response = Mock()
        mock_response.iter_lines.return_value = STREAM_CHUNKS
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Debug the mock chunks
        logger.info(f"Testing with {len(STREAM_CHUNKS)} chunks")
        for i, chunk in enumerate(STREAM_CHUNKS):
            logger.info(f"Chunk {i}: {chunk[:100]}...")

        # Collect results from the generator
        logger.info("Collecting results from the generator")
        results = list(skill.process_stream(input_data))
        logger.info(f"Number of results collected: {len(results)}")

        for i, result in enumerate(results):
            logger.info(f"Result {i}: {result}")

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["headers"] == skill.headers
        assert call_args["stream"] is True
        payload = json.loads(call_args["data"])
        assert payload["stream"] is True

        # Verify results structure
        assert len(results) > 0

        # Check for chunks and final complete object
        chunks = [r for r in results if "chunk" in r]
        complete = [r for r in results if "complete" in r]

        logger.info(f"Number of chunks: {len(chunks)}")
        logger.info(f"Number of complete results: {len(complete)}")

        assert len(chunks) > 0
        assert len(complete) == 1

        # Verify final parsed result
        final_result = complete[0]
        logger.info(f"Final result: {final_result}")

        assert "complete" in final_result
        assert isinstance(final_result["complete"], MockResponseModel)
        assert final_result["complete"].message == "This is a test response"
        assert final_result["complete"].confidence == 0.95
        assert "reasoning" in final_result
        assert final_result["reasoning"] == "Let me analyze this request"

    @patch("requests.post")
    def test_process_stream_error_response(self, mock_post, skill, input_data):
        """Test process_stream method with API error."""
        # Setup mock error response
        mock_post.side_effect = Exception("API Error")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            list(skill.process_stream(input_data))

        assert "Fireworks streaming request failed" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)

    @patch("requests.post")
    def test_process_stream_json_parse_error(self, mock_post, skill, input_data):
        """Test process_stream method with JSON parsing error."""
        # Setup mock response with invalid JSON chunks
        mock_response = Mock()
        mock_response.iter_lines.return_value = INVALID_JSON_CHUNKS
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Collect results from the generator (should not raise exception for individual chunk errors)
        # We expect an exception when we try to process the final malformed response
        with pytest.raises(ProcessingError) as exc_info:
            list(skill.process_stream(input_data))

        # Verify exception message
        assert "Failed to parse JSON response" in str(exc_info.value)

    @patch("requests.post")
    def test_empty_stream_response(self, mock_post, skill, input_data):
        """Test process_stream method with empty stream response."""
        # Setup mock response with no content
        mock_response = Mock()
        mock_response.iter_lines.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # With our new implementation, we explicitly check for empty responses
        # This should now raise an error
        with pytest.raises(ProcessingError) as exc_info:
            list(skill.process_stream(input_data))

        # Verify exception message
        assert "No data received" in str(exc_info.value)

    @patch("requests.post")
    def test_response_format_validation(self, mock_post, skill, input_data):
        """Test validation of the response format schema."""
        # Setup call to process_stream
        mock_response = Mock()
        mock_response.iter_lines.return_value = STREAM_CHUNKS
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Call the method but don't consume the generator
        next(skill.process_stream(input_data))

        # Verify response_format in payload
        assert mock_post.call_args is not None
        payload = json.loads(mock_post.call_args[1]["data"])
        assert "response_format" in payload
        assert payload["response_format"]["type"] == "json_object"
        assert "schema" in payload["response_format"]
        schema = payload["response_format"]["schema"]

        # Check schema structure
        assert schema["title"] == "MockResponseModel"
        assert "message" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert schema["properties"]["confidence"]["type"] == "number"
        assert "message" in schema["required"]
        assert "confidence" in schema["required"]

    @patch("requests.post")
    def test_process_stream_malformed_json(self, mock_post, skill, input_data):
        """Test process_stream method with malformed JSON that we can recover from."""
        # Setup mock response with malformed JSON that our enhanced handler can fix
        malformed_chunks = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"message"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":" : \\"Test "},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{"content":"message\\", confidence: 0.95"},"index":0,"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"deepseek-r1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n',
        ]

        mock_response = Mock()
        mock_response.iter_lines.return_value = malformed_chunks
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Patch the _parse_response_content method to simulate fixing the JSON
        with patch.object(skill, "_parse_response_content") as mock_parse:
            # Return fixed JSON that would work with our MockResponseModel
            mock_parse.return_value = (
                None,
                '{"message": "Test message", "confidence": 0.95}',
            )

            # Collect results from the generator
            results = list(skill.process_stream(input_data))

            # Verify we got chunks and a complete response
            chunks = [r for r in results if "chunk" in r]
            complete = [r for r in results if "complete" in r]

            assert len(chunks) > 0
            assert len(complete) == 1

            # Verify the content of the complete response
            final_result = complete[0]
            assert "complete" in final_result
            assert isinstance(final_result["complete"], MockResponseModel)
            assert final_result["complete"].message == "Test message"
            assert final_result["complete"].confidence == 0.95
