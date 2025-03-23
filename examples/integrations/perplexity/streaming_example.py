#!/usr/bin/env python3
"""
Example demonstrating how to use streaming with Perplexity AI models.

This example shows:
1. How to use the PerplexityStreamingChatSkill
2. How to process streaming tokens in real-time
"""

import os
import sys
import time
from typing import List, Dict
from dotenv import load_dotenv

# Add the parent directory to the path so we can import airtrain
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from airtrain.integrations.perplexity import (
    PerplexityCredentials,
    PerplexityChatSkill,
    PerplexityStreamingChatSkill,
    PerplexityInput
)


def stream_response(streaming_skill: PerplexityStreamingChatSkill, input_data: PerplexityInput) -> None:
    """Stream and display response tokens from a model.
    
    Args:
        streaming_skill: The PerplexityStreamingChatSkill instance
        input_data: The input parameters for the model
    """
    print("\nStreaming response from model", input_data.model, "...")
    print("-" * 50)
    
    # Process the stream
    response_buffer = []
    
    # Use a simple generator to print tokens with a slight delay to simulate typing
    for token in streaming_skill.process_stream(input_data):
        print(token.token, end="", flush=True)
        response_buffer.append(token.token)
        time.sleep(0.01)  # Small delay to make the streaming visible
    
    print("\n" + "-" * 50)
    
    # Return the full response
    return "".join(response_buffer)


def main() -> None:
    """Run the streaming example"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY environment variable not set")
        print("Please set it in your .env file or export it in your shell")
        sys.exit(1)

    # Set up credentials
    credentials = PerplexityCredentials(perplexity_api_key=api_key)
    
    # Create streaming chat skill
    streaming_skill = PerplexityStreamingChatSkill(credentials=credentials)
    
    print("\n=== Streaming Example with sonar-pro ===")
    input_data = PerplexityInput(
        user_input="Explain how deep learning works in simple terms.",
        model="sonar-pro",
        max_tokens=300,
        temperature=0.7
    )
    
    # Stream the response
    response = stream_response(streaming_skill, input_data)
    
    # Regular usage for comparison
    print("\n\n=== Non-Streaming Example for Comparison ===")
    regular_skill = PerplexityChatSkill(credentials=credentials)
    regular_input = PerplexityInput(
        user_input="What is the difference between deep learning and machine learning?",
        model="sonar-pro",
        max_tokens=300,
        temperature=0.7
    )
    
    # Process the regular query
    start_time = time.time()
    output = regular_skill.process(regular_input)
    end_time = time.time()
    
    # Display results
    print(f"Response received in {end_time - start_time:.2f} seconds:")
    print(output.response)
    
    # Try streaming with different models
    print("\n\n=== Streaming Example with sonar-reasoning ===")
    reasoning_input = PerplexityInput(
        user_input="Solve this problem step by step: If a rectangle has a length of 12 cm and a width of 8 cm, what is its perimeter and area?",
        model="sonar-reasoning",
        max_tokens=300,
        temperature=0.2
    )
    
    # Stream the response
    stream_response(streaming_skill, reasoning_input)


if __name__ == "__main__":
    main() 