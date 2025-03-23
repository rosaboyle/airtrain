import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import time

load_dotenv()

parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.combined.groq_fireworks_skills import (
    GroqFireworksSkill,
    GroqFireworksInput,
)


def stream_example():
    """Example of streaming combined Groq and Fireworks processing"""
    skill = GroqFireworksSkill()

    input_data = GroqFireworksInput(
        user_input="Tell me a story about a magical forest",
        groq_model="mixtral-8x7b-32768",
        fireworks_model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=1000,
    )

    try:
        print("\nStreaming combined response...")
        for chunk in skill.process_stream(input_data):
            if "groq_chunk" in chunk:
                print(f"\nGroq: {chunk['groq_chunk']}", end="", flush=True)
            elif "fireworks_chunk" in chunk:
                print(f"\nFireworks: {chunk['fireworks_chunk']}", end="", flush=True)
            time.sleep(0.01)  # Optional: add slight delay for visualization
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def non_stream_example():
    """Example of non-streaming combined Groq and Fireworks processing"""
    skill = GroqFireworksSkill()

    # Example 1: Creative Writing
    input_data = GroqFireworksInput(
        user_input="Write a short story about a robot discovering emotions",
        groq_model="mixtral-8x7b-32768",
        fireworks_model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=1000,
    )

    try:
        print("\nProcessing creative writing task...")
        result = skill.process(input_data)

        print("\nCombined Response (with XML tags):")
        print(result.combined_response)

        print("\nGroq Response Only:")
        print(result.groq_response)

        print("\nFireworks Response Only:")
        print(result.fireworks_response)

        print("\nModels Used:", result.used_models)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")

    # Example 2: Technical Explanation
    input_data = GroqFireworksInput(
        user_input="Explain how quantum computers work",
        groq_model="mixtral-8x7b-32768",
        fireworks_model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=1000,
    )

    try:
        print("\n\nProcessing technical explanation task...")
        result = skill.process(input_data)

        print("\nCombined Response (with XML tags):")
        print(result.combined_response)

        print("\nGroq Response Only:")
        print(result.groq_response)

        print("\nFireworks Response Only:")
        print(result.fireworks_response)

        print("\nModels Used:", result.used_models)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    # Run streaming example
    # stream_example()

    # Run non-streaming examples
    non_stream_example()


if __name__ == "__main__":
    main()
