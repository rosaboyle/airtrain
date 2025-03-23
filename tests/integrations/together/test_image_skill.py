import pytest
from unittest.mock import patch, Mock, MagicMock
import time
from typing import Dict, Any

from airtrain.integrations.together.skills import TogetherAIImageSkill
from airtrain.integrations.together.models import (
    TogetherAIImageInput,
    TogetherAIImageOutput,
    GeneratedImage,
)
from airtrain.core.skills import ProcessingError

from .debug_helpers import logger


class TestTogetherAIImageSkill:
    """Test cases for TogetherAIImageSkill."""

    @pytest.fixture
    def skill(self, mock_credentials):
        """Initialize the skill with mock credentials."""
        with patch("together.Together") as mock_together:
            mock_client = Mock()
            mock_together.return_value = mock_client
            skill = TogetherAIImageSkill(credentials=mock_credentials)
            skill.client = mock_client
            return skill

    @pytest.fixture
    def input_data(self):
        """Create a sample input for image generation testing."""
        return TogetherAIImageInput(
            prompt="A beautiful mountain landscape with a lake at sunset",
            model="black-forest-labs/FLUX.1-schnell-Free",
            steps=10,
            n=1,
            size="1024x1024",
            negative_prompt="blurry, low quality",
        )

    def test_init(self, mock_credentials):
        """Test initialization with provided credentials."""
        with patch("together.Together") as mock_together:
            mock_client = Mock()
            mock_together.return_value = mock_client

            skill = TogetherAIImageSkill(credentials=mock_credentials)
            assert skill.credentials == mock_credentials
            assert skill.client is not None
            mock_together.assert_called_once_with(
                api_key=mock_credentials.together_api_key.get_secret_value()
            )

    def test_process_success(self, skill, input_data, mock_together_client):
        """Test process method with successful response."""
        # Set up the mock
        skill.client = mock_together_client

        # Mock the time.time() function
        with patch("time.time") as mock_time:
            mock_time.side_effect = [
                100.0,
                103.5,
            ]  # Start and end times for duration calculation

            # Call the method
            result = skill.process(input_data)

            # Verify results
            assert isinstance(result, TogetherAIImageOutput)
            assert result.model == input_data.model
            assert result.prompt == input_data.prompt
            assert result.total_time == 3.5  # 103.5 - 100.0
            assert len(result.images) == 1

            # Check the image
            assert isinstance(result.images[0], GeneratedImage)
            assert result.images[0].b64_json == "base64_encoded_image_data"
            assert result.images[0].seed == 12345
            assert result.images[0].finish_reason == "success"

            # Verify the API was called with the right parameters
            skill.client.images.generate.assert_called_once()
            call_args, call_kwargs = skill.client.images.generate.call_args
            assert call_kwargs["prompt"] == input_data.prompt
            assert call_kwargs["model"] == input_data.model
            assert call_kwargs["steps"] == input_data.steps
            assert call_kwargs["n"] == input_data.n
            assert call_kwargs["size"] == input_data.size
            assert call_kwargs["negative_prompt"] == input_data.negative_prompt

    def test_process_multiple_images(self, skill, input_data, mock_together_client):
        """Test generating multiple images."""
        # Set up the mock
        skill.client = mock_together_client

        # Set up multiple image response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(b64_json="image1_data", seed=12345, finish_reason="success"),
            MagicMock(b64_json="image2_data", seed=67890, finish_reason="success"),
        ]
        skill.client.images.generate.return_value = mock_response

        # Set n parameter for multiple images
        input_data.n = 2

        # Call the method
        result = skill.process(input_data)

        # Verify results
        assert len(result.images) == 2
        assert result.images[0].b64_json == "image1_data"
        assert result.images[1].b64_json == "image2_data"

        # Verify the API was called with the right parameters
        call_args, call_kwargs = skill.client.images.generate.call_args
        assert call_kwargs["n"] == 2

    def test_process_different_size(self, skill, input_data):
        """Test with different image size."""
        # Change the size
        input_data.size = "512x512"

        # Call the method
        with patch.object(skill.client, "images") as mock_images:
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(b64_json="image_data", seed=12345, finish_reason="success")
            ]
            mock_images.generate.return_value = mock_response

            result = skill.process(input_data)

            # Verify the API was called with the right parameters
            call_args, call_kwargs = mock_images.generate.call_args
            assert call_kwargs["size"] == "512x512"

    def test_process_with_seed(self, skill, input_data):
        """Test with specific seed for reproducibility."""
        # Set a seed
        input_data.seed = 42

        # Call the method
        with patch.object(skill.client, "images") as mock_images:
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(b64_json="image_data", seed=42, finish_reason="success")
            ]
            mock_images.generate.return_value = mock_response

            result = skill.process(input_data)

            # Verify the API was called with the right parameters
            call_args, call_kwargs = mock_images.generate.call_args
            assert call_kwargs["seed"] == 42

    def test_process_error(self, skill, input_data):
        """Test process method with API error."""
        # Set up the mock to raise an exception
        skill.client.images.generate.side_effect = Exception("API Error")

        # Verify exception is caught and wrapped
        with pytest.raises(ProcessingError) as exc_info:
            skill.process(input_data)

        assert "Together AI image generation failed" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)

    def test_size_validator(self):
        """Test the size validator in the input model."""
        # Valid sizes
        valid_sizes = ["1024x1024", "512x512", "768x1024", "1024x768"]
        for size in valid_sizes:
            input_data = TogetherAIImageInput(
                prompt="Test",
                size=size,
            )
            assert input_data.size == size

        # Invalid sizes
        invalid_sizes = ["invalid", "1024", "x1024", "1024x", "-1x-1", "0x0"]
        for size in invalid_sizes:
            with pytest.raises(ValueError) as exc_info:
                TogetherAIImageInput(
                    prompt="Test",
                    size=size,
                )
            assert "Size must be in format" in str(exc_info.value)
