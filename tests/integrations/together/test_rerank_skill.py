import pytest
from unittest.mock import patch, Mock, MagicMock
from typing import List, Dict, Any

from airtrain.integrations.together.rerank_skill import TogetherAIRerankSkill
from airtrain.integrations.together.schemas import (
    TogetherAIRerankInput,
    TogetherAIRerankOutput,
    RerankResult,
)
from airtrain.core.skills import ProcessingError

from .debug_helpers import logger, debug_rerank_results


class TestTogetherAIRerankSkill:
    """Test cases for TogetherAIRerankSkill."""

    @pytest.fixture
    def skill(self, mock_credentials):
        """Initialize the skill with mock credentials."""
        with patch("together.Together") as mock_together:
            mock_client = Mock()
            mock_together.return_value = mock_client
            skill = TogetherAIRerankSkill(credentials=mock_credentials)
            skill.client = mock_client
            return skill

    @pytest.fixture
    def input_data(self, mock_documents):
        """Create a sample input for testing."""
        return TogetherAIRerankInput(
            query="What are the health benefits of exercise?",
            documents=mock_documents,
            model="Salesforce/Llama-Rank-v1",
            top_n=2,
        )

    def test_init(self, mock_credentials):
        """Test initialization with provided credentials."""
        with patch("together.Together") as mock_together:
            mock_client = Mock()
            mock_together.return_value = mock_client

            skill = TogetherAIRerankSkill(credentials=mock_credentials)
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

                skill = TogetherAIRerankSkill()
                assert skill.credentials == mock_credentials_from_env
                assert skill.client is not None

    @patch("airtrain.integrations.together.rerank_skill.get_rerank_model_config")
    def test_process_success(
        self, mock_get_config, skill, input_data, mock_together_client
    ):
        """Test process method with successful response."""
        # Set up the mock
        skill.client = mock_together_client
        mock_get_config.return_value = MagicMock()

        # Call the method
        result = skill.process(input_data)

        # Verify results
        assert isinstance(result, TogetherAIRerankOutput)
        assert result.used_model == input_data.model
        assert len(result.results) == 3

        # Check first result
        assert result.results[0].index == 0
        assert result.results[0].relevance_score == 0.95
        assert result.results[0].document == input_data.documents[0]

        # Verify the API was called with the right parameters
        skill.client.rerank.create.assert_called_once()
        call_args, call_kwargs = skill.client.rerank.create.call_args
        assert call_kwargs["model"] == input_data.model
        assert call_kwargs["query"] == input_data.query
        assert call_kwargs["documents"] == input_data.documents
        assert call_kwargs["top_n"] == input_data.top_n

    @patch("airtrain.integrations.together.rerank_skill.get_rerank_model_config")
    def test_process_without_top_n(
        self, mock_get_config, skill, input_data, mock_together_client
    ):
        """Test process method without specifying top_n."""
        # Set up the mock
        skill.client = mock_together_client
        mock_get_config.return_value = MagicMock()

        # Remove top_n from input
        input_data.top_n = None

        # Call the method
        result = skill.process(input_data)

        # Verify the API was called with the right parameters
        skill.client.rerank.create.assert_called_once()
        call_args, call_kwargs = skill.client.rerank.create.call_args
        assert call_kwargs["top_n"] is None

    @patch("airtrain.integrations.together.rerank_skill.get_rerank_model_config")
    def test_process_invalid_model(self, mock_get_config, skill, input_data):
        """Test process method with invalid model."""
        # Set up the mock to raise an exception
        mock_get_config.side_effect = ValueError("Model not found")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            skill.process(input_data)

        assert "Together AI reranking failed" in str(exc_info.value)
        assert "Model not found" in str(exc_info.value)

    @patch("airtrain.integrations.together.rerank_skill.get_rerank_model_config")
    def test_process_api_error(self, mock_get_config, skill, input_data):
        """Test process method with API error."""
        # Set up the mock
        mock_get_config.return_value = MagicMock()
        skill.client.rerank.create.side_effect = Exception("API Error")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            skill.process(input_data)

        assert "Together AI reranking failed" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)

    @patch("airtrain.integrations.together.rerank_skill.get_rerank_model_config")
    def test_debug_rerank_results(
        self, mock_get_config, skill, input_data, mock_together_client
    ):
        """Test debugging of rerank results."""
        # Set up the mock
        skill.client = mock_together_client
        mock_get_config.return_value = MagicMock()

        # Call the method
        result = skill.process(input_data)

        # Use debug helper
        debug_rerank_results(result.results)

        # No assertion needed, just verifying the debug function runs without errors
