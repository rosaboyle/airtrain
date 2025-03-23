import pytest
from unittest.mock import Mock
from typing import Type, Dict, Any

from pydantic import BaseModel, Field, SecretStr
import sys
import os

airtain_pypi_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(airtain_pypi_folder)
from airtrain.integrations.fireworks.credentials import FireworksCredentials


@pytest.fixture
def mock_fireworks_api_key() -> str:
    """Return a mock Fireworks API key."""
    return "mock-fireworks-api-key-12345"


@pytest.fixture
def mock_credentials(mock_fireworks_api_key: str) -> FireworksCredentials:
    """Create a FireworksCredentials object with a mock API key."""
    # Create a real FireworksCredentials object with a mock key
    return FireworksCredentials(fireworks_api_key=SecretStr(mock_fireworks_api_key))


@pytest.fixture
def mock_credentials_from_env(
    monkeypatch: pytest.MonkeyPatch, mock_fireworks_api_key: str
) -> None:
    """Mock environment variables for FireworksCredentials."""
    monkeypatch.setenv("FIREWORKS_API_KEY", mock_fireworks_api_key)


@pytest.fixture
def mock_api_response() -> Dict[str, Any]:
    """Create a mock API response from Fireworks API."""
    return {
        "id": "chatcmpl-123456789",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "accounts/fireworks/models/deepseek-r1",
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"key": "value"}'},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_api_response_with_reasoning() -> Dict[str, Any]:
    """Create a mock API response that includes reasoning."""
    return {
        "id": "chatcmpl-123456789",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "accounts/fireworks/models/deepseek-r1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<think>Here is my reasoning process.</think>\n{"key": "value"}',
                },
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def simple_response_model():
    """Create a simple Pydantic model for testing responses."""

    class SimpleResponseModel(BaseModel):
        key: str = Field(..., description="A simple key")

    return SimpleResponseModel
