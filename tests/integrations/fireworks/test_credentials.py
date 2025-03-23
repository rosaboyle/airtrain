import pytest
import os
from pydantic import SecretStr, ValidationError

from airtrain.integrations.fireworks.credentials import FireworksCredentials


class TestFireworksCredentials:
    """Tests for the FireworksCredentials class."""

    def test_init_with_api_key(self, mock_fireworks_api_key):
        """Test initializing credentials with an API key."""
        credentials = FireworksCredentials(
            fireworks_api_key=SecretStr(mock_fireworks_api_key)
        )
        assert (
            credentials.fireworks_api_key.get_secret_value() == mock_fireworks_api_key
        )

    def test_from_env(self, mock_credentials_from_env, mock_fireworks_api_key):
        """Test loading credentials from environment variables."""
        credentials = FireworksCredentials.from_env()
        assert (
            credentials.fireworks_api_key.get_secret_value() == mock_fireworks_api_key
        )

    def test_missing_api_key_env(self, monkeypatch):
        """Test that an error is raised when the API key is not set in environment."""
        # Ensure the environment variable doesn't exist
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

        with pytest.raises(ValueError) as excinfo:
            FireworksCredentials.from_env()
        assert "FIREWORKS_API_KEY environment variable not set" in str(excinfo.value)

    def test_empty_api_key(self):
        """Test that an error is raised for an empty API key."""
        with pytest.raises(ValidationError):
            FireworksCredentials(fireworks_api_key=SecretStr(""))

    def test_model_validation(self, mock_fireworks_api_key):
        """Test the validation of the model."""
        # Valid credentials
        credentials = FireworksCredentials(
            fireworks_api_key=SecretStr(mock_fireworks_api_key)
        )
        assert credentials is not None

        # Test model validation - should pass for valid keys
        credentials_dict = credentials.model_dump()
        assert "fireworks_api_key" in credentials_dict

        # SecretStr values are not included in string representation for security
        assert (
            str(credentials)
            == "FireworksCredentials(fireworks_api_key=SecretStr('**********'))"
        )
