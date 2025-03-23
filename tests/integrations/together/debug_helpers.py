"""Debugging helpers for Together AI integration tests."""

import json
import logging
from typing import List, Dict, Any, Optional

# Configure a logger specific for together tests
logger = logging.getLogger("together_tests")
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


def debug_streaming_chunks(chunks: List[Any]) -> None:
    """Print debug information about streaming response chunks."""
    logger.info(f"Analyzing {len(chunks)} streaming chunks")

    # Print first and last chunk if available
    if chunks:
        logger.info(f"First chunk: {str(chunks[0])[:200]}")
        if len(chunks) > 1:
            logger.info(f"Last chunk: {str(chunks[-1])[:200]}")

        # Extract and analyze each chunk content
        all_content = []
        for i, chunk in enumerate(chunks):
            try:
                # Extract content if present
                content = None
                if hasattr(chunk, "choices") and chunk.choices:
                    if hasattr(chunk.choices[0], "delta") and hasattr(
                        chunk.choices[0].delta, "content"
                    ):
                        content = chunk.choices[0].delta.content
                        if content:
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


def debug_rerank_results(results: List[Any]) -> None:
    """Debug rerank results."""
    logger.info(f"Rerank results ({len(results)} items):")

    for i, result in enumerate(results):
        try:
            logger.info(
                f"Result {i}: index={result.index}, score={result.relevance_score:.4f}"
            )
        except Exception as e:
            logger.error(f"Error parsing result {i}: {str(e)}")


def debug_model_validation(model_class: Any, data: Any) -> None:
    """Debug Pydantic model validation issues."""
    logger.info(f"Attempting to validate data with {model_class.__name__}")

    if isinstance(data, str):
        # If it's a string, try to parse as JSON
        try:
            data_dict = json.loads(data)
            logger.info(f"JSON parsed successfully: {json_serialize_safe(data_dict)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return
    else:
        data_dict = data

    try:
        # Now try to validate with the model
        validated = model_class.model_validate(data_dict)
        logger.info(f"Validation successful: {validated}")
    except Exception as e:
        logger.error(f"Model validation error: {str(e)}")


def debug_api_response(response_data: Dict[str, Any]) -> None:
    """Debug an API response dict."""
    logger.info(f"API Response: {json_serialize_safe(response_data)}")

    # Check for specific fields based on response type
    if "choices" in response_data and response_data["choices"]:
        # Chat response
        choice = response_data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            content = choice["message"]["content"]
            logger.info(f"Content: {content[:200]}")  # First 200 chars
    elif "results" in response_data and response_data["results"]:
        # Rerank response
        logger.info(f"Rerank results: {len(response_data['results'])} items")
        for i, result in enumerate(response_data["results"][:5]):  # Log first 5 results
            logger.info(f"Result {i}: {json_serialize_safe(result)}")
    elif "data" in response_data and response_data["data"]:
        # Image generation response
        logger.info(f"Generated images: {len(response_data['data'])}")
        for i, image_data in enumerate(response_data["data"]):
            seed = image_data.get("seed", "unknown")
            finish_reason = image_data.get("finish_reason", "unknown")
            logger.info(f"Image {i}: seed={seed}, finish_reason={finish_reason}")
