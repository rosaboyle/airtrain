import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any, List
from pydantic import SecretStr
import sys
import os

# Fix import path by adding the project root to the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Create a mock together module to prevent import errors
# This needs to happen before any airtrain imports
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


# Mock Models class with async list method
class MockModels:
    @classmethod
    async def list(cls):
        return ["model1", "model2"]


# Create mock together module
mock_together = MagicMock()
mock_together.Together = MockTogether
mock_together.Models = MockModels
mock_together.api_key = None

# Add to sys.modules before importing from airtrain
sys.modules["together"] = mock_together

# Now we can import from airtrain
from airtrain.integrations.together.credentials import TogetherAICredentials
from airtrain.integrations.together.schemas import RerankResult

# Create AsyncMock class if not available (for Python < 3.8)
if not hasattr(sys.modules["unittest.mock"], "AsyncMock"):

    class AsyncMock(MagicMock):
        async def __call__(self, *args, **kwargs):
            return super(AsyncMock, self).__call__(*args, **kwargs)

    sys.modules["unittest.mock"].AsyncMock = AsyncMock


@pytest.fixture
def mock_together_api_key() -> str:
    """Return a mock Together AI API key."""
    return "mock-together-api-key-12345"


@pytest.fixture
def mock_credentials(mock_together_api_key: str) -> TogetherAICredentials:
    """Create a TogetherAICredentials object with a mock API key."""
    return TogetherAICredentials(together_api_key=SecretStr(mock_together_api_key))


@pytest.fixture
def mock_credentials_from_env(
    monkeypatch: pytest.MonkeyPatch, mock_together_api_key: str
) -> None:
    """Mock environment variables for TogetherAICredentials."""
    monkeypatch.setenv("TOGETHER_API_KEY", mock_together_api_key)


@pytest.fixture
def mock_chat_response() -> Dict[str, Any]:
    """Create a mock chat response from Together AI API."""
    return {
        "id": "chatcmpl-123456789",
        "model": "deepseek-ai/DeepSeek-R1",
        "choices": [
            {
                "message": {"role": "assistant", "content": "This is a test response"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_rerank_response() -> Dict[str, Any]:
    """Create a mock rerank response from Together AI API."""
    return {
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.85},
            {"index": 1, "relevance_score": 0.75},
        ],
        "model": "Salesforce/Llama-Rank-v1",
    }


@pytest.fixture
def mock_documents() -> List[str]:
    """Return a list of documents for reranking tests."""
    return [
        "Regular exercise improves cardiovascular health.",
        "A balanced diet is essential for health.",
        "Exercise helps maintain mental health and reduce stress.",
    ]


@pytest.fixture
def mock_image_response() -> Dict[str, Any]:
    """Create a mock image generation response from Together AI API."""
    return {
        "data": [
            {
                "b64_json": "base64_encoded_image_data",
                "seed": 12345,
                "finish_reason": "success",
            }
        ],
        "created": 1678912345,
    }


@pytest.fixture
def mock_together_client() -> Mock:
    """Create a mock Together client for testing."""
    mock_client = Mock()

    # Setup chat completions
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = "This is a test response"
    mock_completion.usage = MagicMock()
    mock_completion.usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    mock_client.chat.completions.create.return_value = mock_completion

    # Setup streaming response
    mock_stream = []
    for i in range(5):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = f"chunk{i} "
        mock_stream.append(chunk)
    mock_client.chat.completions.create.side_effect = lambda **kwargs: (
        mock_stream if kwargs.get("stream", False) else mock_completion
    )

    # Setup rerank response
    mock_rerank_result = MagicMock()
    mock_rerank_result.results = [
        MagicMock(index=0, relevance_score=0.95),
        MagicMock(index=2, relevance_score=0.85),
        MagicMock(index=1, relevance_score=0.75),
    ]
    mock_client.rerank.create.return_value = mock_rerank_result

    # Setup image generation response
    mock_image_result = MagicMock()
    mock_image_result.data = [
        MagicMock(
            b64_json="base64_encoded_image_data", seed=12345, finish_reason="success"
        )
    ]
    mock_client.images.generate.return_value = mock_image_result

    return mock_client
