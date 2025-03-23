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

from airtrain.integrations.fireworks.requests_skills import (
    FireworksRequestSkill,
    FireworksRequestInput,
)


def stream_example():
    """Example of streaming chat with Fireworks AI using requests"""
    skill = FireworksRequestSkill()

    input_data = FireworksRequestInput(
        user_input="Write a short story about a robot learning to paint.",
        system_prompt="You are a creative storyteller.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=4096,
        top_p=1.0,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=True,
    )

    print("\nStreaming response using requests:")
    try:
        for chunk in skill.process_stream(input_data):
            print(chunk, end="", flush=True)
            time.sleep(0.01)  # Optional: add slight delay for better visualization
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def non_stream_example():
    """Example of non-streaming chat with Fireworks AI using requests"""
    skill = FireworksRequestSkill()

    input_data = FireworksRequestInput(
        user_input="Explain the concept of quantum entanglement.",
        system_prompt="You are a quantum physics expert.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=4096,
        top_p=1.0,
        top_k=40,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=False,
    )

    try:
        print("\nNon-streaming response using requests:")
        result = skill.process(input_data)
        print(result.response)
        print("\nModel Used:", result.used_model)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    # Run streaming example
    stream_example()

    # Run non-streaming example
    non_stream_example()


if __name__ == "__main__":
    main()
