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

from airtrain.integrations.fireworks.skills import FireworksChatSkill, FireworksInput


def stream_example():
    """Example of streaming chat with Fireworks AI"""
    skill = FireworksChatSkill()

    input_data = FireworksInput(
        user_input="Write a short story about a robot learning to paint.",
        system_prompt="You are a creative storyteller.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=10240,
        stream=True,
    )

    print("\nStreaming response:")
    try:
        for chunk in skill.process_stream(input_data):
            print(chunk, end="", flush=True)
            time.sleep(0.01)  # Optional: add slight delay for better visualization
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    # Run streaming example
    stream_example()

    # Compare with non-streaming response
    skill = FireworksChatSkill()
    input_data = FireworksInput(
        user_input="Write a short story about a robot learning to paint.",
        system_prompt="You are a creative storyteller.",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=10240,
        stream=False,
    )

    try:
        print("\nNon-streaming response:")
        result = skill.process(input_data)
        print(result.response)
        print("\nUsage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
