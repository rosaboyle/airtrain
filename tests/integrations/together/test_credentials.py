import pytest
from unittest.mock import patch, AsyncMock
import os
from pydantic import SecretStr, ValidationError

from airtrain.integrations.together.credentials import TogetherAICredentials
from airtrain.core.credentials import CredentialValidationError


class TestTogetherAICredentials:
    """Test TogetherAICredentials functionality."""

    def test_init_with_api_key(self, mock_together_api_key: str):
        """Test initialization with provided API key."""
        credentials = TogetherAICredentials(together_api_key=mock_together_api_key)
        assert credentials.together_api_key.get_secret_value() == mock_together_api_key

    def test_init_with_secret_str(self, mock_together_api_key: str):
        """Test initialization with SecretStr API key."""
        secret_key = SecretStr(mock_together_api_key)
        credentials = TogetherAICredentials(together_api_key=secret_key)
        assert credentials.together_api_key == secret_key
        assert credentials.together_api_key.get_secret_value() == mock_together_api_key

    def test_from_env(
        self, mock_credentials_from_env: None, mock_together_api_key: str
    ):
        """Test creating credentials from environment variables."""
        credentials = TogetherAICredentials.from_env()
        assert credentials.together_api_key.get_secret_value() == mock_together_api_key

    def test_missing_api_key_env(self, monkeypatch: pytest.MonkeyPatch):
        """Test handling missing environment variable."""
        # Clear the environment variable if it exists
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            TogetherAICredentials.from_env()

        assert "TOGETHER_API_KEY environment variable not set" in str(exc_info.value)

    def test_empty_api_key(self):
        """Test validation of empty API key."""
        with pytest.raises(ValidationError):
            TogetherAICredentials(together_api_key="")

    @patch("together.Models.list")
    async def test_validate_credentials_success(
        self, mock_list: AsyncMock, mock_credentials: TogetherAICredentials
    ):
        """Test successful credential validation."""
        mock_list.return_value = ["model1", "model2"]

        result = await mock_credentials.validate_credentials()
        assert result is True
        mock_list.assert_called_once()

    @patch("together.Models.list")
    async def test_validate_credentials_failure(
        self, mock_list: AsyncMock, mock_credentials: TogetherAICredentials
    ):
        """Test failed credential validation."""
        mock_list.side_effect = Exception("Invalid API key")

        with pytest.raises(CredentialValidationError) as exc_info:
            await mock_credentials.validate_credentials()

        assert "Invalid Together AI credentials" in str(exc_info.value)
