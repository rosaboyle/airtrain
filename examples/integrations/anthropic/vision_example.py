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

    # Example with image analysis
    image_path = Path("examples/images/quantum-circuit.png")
    if not image_path.exists():
        print(f"Warning: Image file not found at {image_path}")
        return

    # Create input model with image
    image_input = AnthropicInput(
        user_input="What does this quantum circuit diagram show?",
        images=[image_path],
        system_prompt="You are an expert in quantum computing. Analyze the circuit diagram.",
        model="claude-3-opus-20240229",
        temperature=0.3,
    )

    try:
        result = skill.process(image_input)
        print("\nImage Analysis Response:")
        print(result.response)
        print("\nUsage Statistics:", result.usage)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
