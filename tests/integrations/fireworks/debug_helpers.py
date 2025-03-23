"""Debugging helpers for Fireworks integration tests."""

import json
import logging
from typing import List, Dict, Any, Optional

# Configure a logger specific for fireworks tests
logger = logging.getLogger("fireworks_tests")
logger.setLevel(logging.DEBUG)

# Set up a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def json_serialize_safe(data: Any) -> str:
    """Safely serialize data for logging."""
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f"<Failed to serialize: {str(e)}>"


def debug_chunks(chunks: List[bytes]) -> None:
    """Print debug information about API response chunks."""
    logger.info(f"Analyzing {len(chunks)} chunks")

    # Print first and last chunk if available
    if chunks:
        logger.info(f"First chunk: {chunks[0]}")
        if len(chunks) > 1:
            logger.info(f"Last chunk: {chunks[-1]}")

        # Parse and analyze each chunk
        all_content = []
        for i, chunk in enumerate(chunks):
            try:
                # Strip the "data: " prefix
                json_str = chunk.decode("utf-8").removeprefix("data: ")
                # Parse the JSON
                data = json.loads(json_str)
                # Extract content if present
                content = None
                if (
                    "choices" in data
                    and data["choices"]
                    and "delta" in data["choices"][0]
                ):
                    delta = data["choices"][0]["delta"]
                    if "content" in delta:
                        content = delta["content"]
                        all_content.append(content)

                logger.info(f"Chunk {i}: Content={content}")
            except Exception as e:
                logger.error(f"Error parsing chunk {i}: {str(e)}")

        # Try to assemble the complete content
        if all_content:
            complete_content = "".join(all_content)
            logger.info(
                f"Combined content: {complete_content[:200]}"
            )  # First 200 chars

            # Try to parse as JSON
            try:
                json_obj = json.loads(complete_content)
                logger.info(f"Parsed JSON object: {json_serialize_safe(json_obj)}")
            except json.JSONDecodeError:
                logger.warning("Could not parse combined content as JSON")


def debug_model_validation(model_class: Any, json_str: str) -> None:
    """Debug Pydantic model validation issues."""
    logger.info(f"Attempting to validate JSON with {model_class.__name__}")
    logger.info(f"JSON string: {json_str[:200]}")  # First 200 chars

    try:
        # Try to parse JSON first
        json_obj = json.loads(json_str)
        logger.info(f"JSON parsed successfully: {json_serialize_safe(json_obj)}")

        # Now try to validate with the model
        validated = model_class.model_validate(json_obj)
        logger.info(f"Validation successful: {validated}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
    except Exception as e:
        logger.error(f"Model validation error: {str(e)}")


def debug_api_response(response_data: Dict[str, Any]) -> None:
    """Debug an API response dict."""
    logger.info(f"API Response: {json_serialize_safe(response_data)}")

    # Check for specific fields
    if "choices" in response_data and response_data["choices"]:
        choice = response_data["choices"][0]

        # Check for message content
        if "message" in choice and "content" in choice["message"]:
            content = choice["message"]["content"]
            logger.info(f"Content: {content[:200]}")  # First 200 chars

            # Try to parse content as JSON
            try:
                json_obj = json.loads(content)
                logger.info(f"Content parsed as JSON: {json_serialize_safe(json_obj)}")
            except json.JSONDecodeError:
                logger.warning("Content could not be parsed as JSON")
