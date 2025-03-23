import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.together.skills import TogetherAIInput, TogetherAIChatSkill


def main():
    # Initialize the skill
    skill = TogetherAIChatSkill()

    # Create input
    input_data = TogetherAIInput(
        user_input="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful teacher who explains complex topics simply.",
        model="togethercomputer/llama-2-70b",
        temperature=0.7,
        max_tokens=1024,
    )

    try:
        result = skill.process(input_data)
        print("\nChat Response:")
        print(result.response)
        print("\nModel Used:", result.used_model)
        print("Usage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
