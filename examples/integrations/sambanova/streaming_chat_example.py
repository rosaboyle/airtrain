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

from airtrain.integrations.sambanova.skills import SambanovaChatSkill, SambanovaInput


def stream_example():
    """Example of streaming chat with SambaNova"""
    skill = SambanovaChatSkill()

    input_data = SambanovaInput(
        user_input="Explain distributed computing in simple terms",
        system_prompt="You are a systems engineering expert",
        model="DeepSeek-R1-Distill-Llama-70B",  # Use actual SambaNova model name
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
    skill = SambanovaChatSkill()
    input_data = SambanovaInput(
        user_input="Explain distributed computing in simple terms",
        system_prompt="You are a systems engineering expert",
        model="DeepSeek-R1-Distill-Llama-70B",
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
