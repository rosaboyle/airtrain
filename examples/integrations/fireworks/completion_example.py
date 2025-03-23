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

from airtrain.integrations.fireworks.completion_skills import (
    FireworksCompletionSkill,
    FireworksCompletionInput,
)


def stream_example():
    """Example of streaming completion with Fireworks AI"""
    skill = FireworksCompletionSkill()

    input_data = FireworksCompletionInput(
        prompt="<USER>Write a story for me. About a magical forest.</USER><ASSISTANT><think>",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=200,
        top_p=1.0,
        top_k=50,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=True,
    )

    print("\nStreaming completion response:")
    try:
        for chunk in skill.process_stream(input_data):
            print(chunk, end="", flush=True)
            time.sleep(0.01)  # Optional: add slight delay for visualization
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def non_stream_example():
    """Example of non-streaming completion with Fireworks AI"""
    skill = FireworksCompletionSkill()

    input_data = FireworksCompletionInput(
        prompt="Complete this sentence: The future of artificial intelligence is",
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        max_tokens=100,
        top_p=1.0,
        top_k=50,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        stream=False,
    )

    try:
        print("\nNon-streaming completion response:")
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
