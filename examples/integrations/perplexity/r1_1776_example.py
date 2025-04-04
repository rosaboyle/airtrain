#!/usr/bin/env python3
"""
Example demonstrating how to use the Perplexity AI 'r1-1776' offline model.

This example shows:
1. How to use the offline r1-1776 model
2. Differences between online and offline models
3. Working with conversation history
"""

import os
import sys
import json
from typing import List, Dict
from dotenv import load_dotenv

# Add the parent directory to the path so we can import airtrain
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from airtrain.integrations.perplexity import (
    PerplexityCredentials,
    PerplexityChatSkill,
    PerplexityInput,
)


def simulate_conversation(
    chat_skill: PerplexityChatSkill,
    conversation: List[Dict[str, str]],
    model: str,
    system_prompt: str = None,
) -> None:
    """Simulate a multi-turn conversation with the model.

    Args:
        chat_skill: The PerplexityChatSkill instance
        conversation: List of message dictionaries with 'role' and 'content'
        model: The model ID to use
        system_prompt: Optional system prompt
    """
    history = []

    for i, message in enumerate(conversation):
        if message["role"] == "user":
            print(f"\nUser: {message['content']}")

            # Create input for the model
            input_data = PerplexityInput(
                user_input=message["content"],
                model=model,
                system_prompt=system_prompt,
                conversation_history=history.copy(),  # Pass the current history
                max_tokens=500,
                temperature=0.7,
            )

            # Process the query
            output = chat_skill.process(input_data)

            # Display response
            print(f"\nAssistant ({model}): {output.response}")

            # Add this exchange to history
            history.append({"role": "user", "content": message["content"]})
            history.append({"role": "assistant", "content": output.response})
        else:
            # Skip non-user messages as they'll be generated by the model
            continue


def main() -> None:
    """Run the r1-1776 model example"""
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

    # Create chat skill
    chat_skill = PerplexityChatSkill(credentials=credentials)

    print("\n=== R1-1776 Example - Basic Usage ===")
    input_data = PerplexityInput(
        user_input="What are the advantages and limitations of different political systems?",
        model="r1-1776",
        max_tokens=600,
        temperature=0.7,
    )

    # Process the query
    output = chat_skill.process(input_data)

    # Display results
    print(f"Response:\n{output.response}\n")
    print(f"Model: {output.used_model}")
    print(f"Usage: {json.dumps(output.usage, indent=2)}")

    # Notice: No citations with offline models
    if output.citations:
        print("\nCitations:")
        for i, citation in enumerate(output.citations, 1):
            print(f"{i}. {citation.url}")
    else:
        print(
            "\nNo citations available - r1-1776 is an offline model without search capability"
        )

    # Example with conversation
    print("\n\n=== R1-1776 Example - Multi-turn Conversation ===")

    # Define a series of user messages
    conversation = [
        {"role": "user", "content": "What is the concept of separation of powers?"},
        {
            "role": "user",
            "content": "How is this implemented in the United States government?",
        },
        {"role": "user", "content": "What are some criticisms of this system?"},
        {"role": "user", "content": "Can you suggest any potential improvements?"},
    ]

    # System prompt to guide the conversation
    system_prompt = "You are a political science professor who specializes in government systems. Provide detailed, nuanced, and factual information."

    # Simulate the conversation
    simulate_conversation(chat_skill, conversation, "r1-1776", system_prompt)


if __name__ == "__main__":
    main()
