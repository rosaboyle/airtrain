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

from airtrain.integrations.openai.skills import OpenAITextSkill, OpenAITextInput


def main():
    # Initialize the skill
    skill = OpenAITextSkill()

    # Basic text completion example
    input_data = OpenAITextInput(
        user_input="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful teacher who explains complex topics simply.",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=500,
    )

    try:
        result = skill.process(input_data)
        print("\nChat Response:")
        print(result.response)
        print("\nModel Used:", result.used_model)
        print("Tokens Used:", result.tokens_used)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
