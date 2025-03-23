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

from airtrain.integrations.anthropic.skills import AnthropicChatSkill, AnthropicInput


def main():
    # Initialize the skill
    skill = AnthropicChatSkill()

    # Create input
    input_data = AnthropicInput(
        user_input="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful teacher who explains complex topics simply.",
        max_tokens=500,
        model="claude-3-opus-20240229",
        temperature=0.7,
    )

    # Process the input
    try:
        result = skill.process(input_data)
        print("\nBasic Chat Response:")
        print(result.response)
        print("\nUsage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
