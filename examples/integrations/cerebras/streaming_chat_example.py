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

from airtrain.integrations.cerebras.skills import CerebrasChatSkill, CerebrasInput


def stream_example():
    """Example of streaming chat with Cerebras"""
    skill = CerebrasChatSkill()

    input_data = CerebrasInput(
        user_input="Explain neural networks in simple terms",
        system_prompt="You are a computer science professor",
        model="llama3.1-8b",
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )

    print("\nStreaming response:")
    try:
        for chunk in skill.process_stream(input_data):
            print(chunk, end="", flush=True)
            time.sleep(0.01)
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    # Run streaming example
    stream_example()

    # Compare with non-streaming response
    skill = CerebrasChatSkill()
    input_data = CerebrasInput(
        user_input="Explain neural networks in simple terms",
        system_prompt="You are a computer science professor",
        model="llama3.1-8b",
        temperature=0.7,
        max_tokens=1024,
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
