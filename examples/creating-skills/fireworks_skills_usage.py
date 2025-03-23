import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.integrations.fireworks.skills import FireworksChatSkill, FireworksInput


def basic_chat_example():
    """Example of basic chat with Fireworks AI"""
    # Initialize the skill
    skill = FireworksChatSkill()

    # Create input for a basic chat
    input_data = FireworksInput(
        user_input="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful teacher who explains complex topics simply.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=500,
    )

    # Process the input
    try:
        result = skill.process(input_data)
        print("\nBasic Chat Response:")
        print(result.response)
        print("\nUsage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


def structured_chat_example():
    """Example of chat with specific context and parameters"""
    skill = FireworksChatSkill()

    # Create input with more specific parameters
    input_data = FireworksInput(
        user_input="What are the key differences between classical and quantum computers?",
        system_prompt="You are a quantum computing expert. Provide clear, technical explanations.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.2,
        max_tokens=1000,
        context_length_exceeded_behavior="truncate",
    )

    try:
        result = skill.process(input_data)
        print("\nStructured Chat Response:")
        print(result.response)
        print("\nModel Used:", result.used_model)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Running Fireworks AI Examples...")
    basic_chat_example()
    structured_chat_example()
