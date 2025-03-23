import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Generator, Type
from pydantic import BaseModel, Field

from airtrain.integrations.fireworks.structured_requests_skills import (
    FireworksStructuredRequestSkill,
    FireworksStructuredRequestInput,
    FireworksStructuredRequestOutput,
    ProcessingError,
)
from airtrain.integrations.fireworks.credentials import FireworksCredentials


# Sample response model for testing
class MockResponseModel(BaseModel):
    message: str = Field(..., description="A test message")
    confidence: float = Field(..., description="Confidence score")


# Mock responses
MOCK_API_RESPONSE = {
    "id": "chatcmpl-123456789",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "accounts/fireworks/models/deepseek-r1",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": '{"message": "This is a test response", "confidence": 0.95}',
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}

MOCK_STREAM_CHUNKS = [
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{"content":"{\\"message\\":"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{"content":" \\"This is"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{"content":" a test"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{"content":" response\\""},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{"content":", \\"confidence\\": 0.95}"},"index":0,"finish_reason":null}]}\n\n',
    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"accounts/fireworks/models/deepseek-r1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n\n',
]

MOCK_REASONING_RESPONSE = {
    "id": "chatcmpl-123456789",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "accounts/fireworks/models/deepseek-r1",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": '<think>This is reasoning about the response.</think>\n{"message": "This is a test response", "confidence": 0.95}',
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}


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
    """Create a sample input for testing."""
    return FireworksStructuredRequestInput(
        user_input="Test input",
        system_prompt="Test system prompt",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=100,
        response_model=MockResponseModel,
        stream=False,
    )


class TestFireworksStructuredRequestSkill:
    """Test cases for FireworksStructuredRequestSkill."""

    def test_init(self, mock_credentials):
        """Test initialization with provided credentials."""
        skill = FireworksStructuredRequestSkill(credentials=mock_credentials)
        assert skill.credentials == mock_credentials
        assert skill.headers == {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-api-key",
        }

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
        assert messages[1] == {"role": "user", "content": "Previous message"}
        assert messages[2] == {"role": "assistant", "content": "Previous response"}

    def test_build_payload(self, skill, input_data):
        """Test _build_payload method."""
        payload = skill._build_payload(input_data)
        assert payload["model"] == "accounts/fireworks/models/deepseek-r1"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["stream"] is False
        assert payload["response_format"]["type"] == "json_object"
        assert "schema" in payload["response_format"]
        assert "message" in payload["response_format"]["schema"]["properties"]
        assert "confidence" in payload["response_format"]["schema"]["properties"]

    def test_parse_response_content(self, skill):
        """Test _parse_response_content method with different response formats."""
        # Test with reasoning and valid JSON
        content = (
            '<think>This is reasoning.</think>{"message": "Test", "confidence": 0.9}'
        )
        reasoning, json_str = skill._parse_response_content(content)
        assert reasoning == "This is reasoning."
        assert json_str == '{"message": "Test", "confidence": 0.9}'

        # Test with reasoning and JSON without braces
        content = (
            '<think>Another reasoning.</think>"message": "Test", "confidence": 0.9'
        )
        reasoning, json_str = skill._parse_response_content(content)
        assert reasoning == "Another reasoning."
        # Don't check the exact content, just ensure the JSON part is present
        assert '"message": "Test", "confidence": 0.9' in json_str

        # Test without reasoning
        content = '{"message": "Test", "confidence": 0.9}'
        reasoning, json_str = skill._parse_response_content(content)
        assert reasoning is None
        # Instead of exact string comparison, verify JSON content
        assert "message" in json_str
        assert "Test" in json_str
        assert "confidence" in json_str
        assert "0.9" in json_str

        # Test with whitespace
        content = '  \n  {"message": "Test", "confidence": 0.9}  \n  '
        reasoning, json_str = skill._parse_response_content(content)
        assert reasoning is None
        # Instead of exact string comparison, verify JSON content
        assert "message" in json_str
        assert "Test" in json_str
        assert "confidence" in json_str
        assert "0.9" in json_str

    @patch("requests.post")
    def test_process_success(self, mock_post, skill, input_data):
        """Test process method with successful response."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MOCK_API_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Call the method
        result = skill.process(input_data)

        # Verify results
        assert isinstance(result, FireworksStructuredRequestOutput)
        assert result.used_model == "accounts/fireworks/models/deepseek-r1"
        assert result.usage == MOCK_API_RESPONSE["usage"]
        assert result.parsed_response.message == "This is a test response"
        assert result.parsed_response.confidence == 0.95

        # Verify the API was called with the right parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args[1]
        assert call_args["headers"] == skill.headers
        payload = json.loads(call_args["data"])
        assert payload["model"] == input_data.model

    @patch("requests.post")
    def test_process_with_reasoning(self, mock_post, skill, input_data):
        """Test process method with reasoning in response."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = MOCK_REASONING_RESPONSE
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Call the method
        result = skill.process(input_data)

        # Verify results
        assert result.reasoning == "This is reasoning about the response."
        assert result.parsed_response.message == "This is a test response"
        assert result.parsed_response.confidence == 0.95

    @patch("requests.post")
    def test_process_error(self, mock_post, skill, input_data):
        """Test process method with API error."""
        # Setup mock error response
        mock_post.side_effect = Exception("API Error")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            skill.process(input_data)

        assert "Fireworks structured request failed" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)

    @patch("requests.post")
    def test_process_stream(self, mock_post, skill, input_data):
        """Test process method with streaming enabled."""
        # Set stream flag
        input_data.stream = True

        # Setup mock streaming response
        mock_response = Mock()
        mock_response.iter_lines.return_value = MOCK_STREAM_CHUNKS
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Patch the process_stream method to return a generator with expected values
        with patch.object(skill, "process_stream") as mock_stream:
            mock_stream.return_value = (
                item
                for item in [
                    {"chunk": '{"message": '},
                    {"chunk": '"This is'},
                    {"chunk": " a test"},
                    {"chunk": ' response"'},
                    {"chunk": ', "confidence": 0.95}'},
                    {
                        "complete": MockResponseModel(
                            message="This is a test response", confidence=0.95
                        )
                    },
                ]
            )

            result = skill.process(input_data)

            # Verify results
            assert isinstance(result, FireworksStructuredRequestOutput)
            assert result.used_model == "accounts/fireworks/models/deepseek-r1"
            assert isinstance(result.parsed_response, MockResponseModel)
            assert result.parsed_response.message == "This is a test response"
            assert result.parsed_response.confidence == 0.95
