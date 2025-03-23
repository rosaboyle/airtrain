"""
Tests for the Exa search integration.
"""

import os
import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from airtrain.integrations.search.exa import (
    ExaCredentials,
    ExaSearchSkill,
    ExaSearchInputSchema,
    ExaSearchOutputSchema,
)
from airtrain.core.errors import ProcessingError


class TestExaCredentials:
    """Tests for the ExaCredentials class."""

    def test_init(self):
        """Test initialization of credentials."""
        creds = ExaCredentials(api_key="test-api-key")
        assert creds.api_key.get_secret_value() == "test-api-key"

    @pytest.mark.asyncio
    async def test_validate_credentials(self):
        """Test credential validation."""
        creds = ExaCredentials(api_key="test-api-key")
        assert await creds.validate_credentials() is True

        # Test with empty key
        with pytest.raises(Exception):
            empty_creds = ExaCredentials(api_key="")
            await empty_creds.validate_credentials()


class TestExaSearchSkill:
    """Tests for the ExaSearchSkill class."""

    def setup_method(self):
        """Set up test environment."""
        self.credentials = ExaCredentials(api_key="test-api-key")
        self.skill = ExaSearchSkill(credentials=self.credentials)

        # Sample search response
        self.sample_response = {
            "results": [
                {
                    "id": "123",
                    "url": "https://example.com/article1",
                    "title": "Sample Article 1",
                    "text": "This is a sample article about the topic.",
                    "score": 0.95,
                    "published": "2023-05-15",
                    "robotsAllowed": True,
                    "moderationConfig": {"llamaguardS1": False, "llamaguardS3": False},
                },
                {
                    "id": "456",
                    "url": "https://example.com/article2",
                    "title": "Sample Article 2",
                    "text": "Another sample article with relevant information.",
                    "score": 0.85,
                    "robotsAllowed": True,
                },
            ],
            "autopromptString": "related to the topic",
            "costDollars": {
                "total": 0.015,
                "search": {"neural": 0.005},
                "contents": {"text": 0.01},
            },
        }

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_process_successful(self, mock_post):
        """Test successful processing of a search request."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_response
        mock_post.return_value = mock_response

        # Create input
        input_data = ExaSearchInputSchema(
            query="test query", numResults=2, contents={"text": True}
        )

        # Call process
        result = await self.skill.process(input_data)

        # Assert response was correctly processed
        assert isinstance(result, ExaSearchOutputSchema)
        assert result.query == "test query"
        assert len(result.results) == 2
        assert result.results[0].id == "123"
        assert result.results[0].title == "Sample Article 1"
        assert result.results[1].url == "https://example.com/article2"
        assert result.costDollars.total == 0.015

        # Assert request was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.exa.ai/search"
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-api-key"

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_process_error_response(self, mock_post):
        """Test handling of error responses."""
        # Set up mock error response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        # Create input
        input_data = ExaSearchInputSchema(query="test query", numResults=2)

        # Call process and expect an error
        with pytest.raises(ProcessingError) as exc_info:
            await self.skill.process(input_data)

        # Check error message
        assert "status code 401" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post")
    async def test_process_timeout(self, mock_post):
        """Test handling of request timeouts."""
        # Set up mock to raise timeout
        mock_post.side_effect = httpx.TimeoutException("Timeout")

        # Create input
        input_data = ExaSearchInputSchema(query="test query")

        # Call process and expect a timeout error
        with pytest.raises(ProcessingError) as exc_info:
            await self.skill.process(input_data)

        # Check error message
        assert "Timeout while querying Exa API" in str(exc_info.value)
