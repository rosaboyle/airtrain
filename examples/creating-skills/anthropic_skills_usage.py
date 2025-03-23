import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

# Update imports to use the correct path
from airtrain.integrations.anthropic.credentials import AnthropicCredentials
from airtrain.integrations.anthropic.skills import AnthropicChatSkill, AnthropicInput


def main():
    # Basic text completion example
    skill = AnthropicChatSkill()

    # Create proper AnthropicInput instance
    input_data = AnthropicInput(
        user_input="Explain quantum computing in simple terms.",
        system_prompt="You are a helpful teacher who explains complex topics simply.",
        max_tokens=500,
    )

    # Process with proper input model
    result = skill.process(input_data)

    print("Basic Response:")
    print(result.response)
    print("\nUsage Statistics:", result.usage)

    # Example with image analysis
    image_path = Path("examples/images/quantum_circuit.jpg")  # Example image path
    if image_path.exists():
        # Create input model with image
        image_input = AnthropicInput(
            user_input="What does this quantum circuit diagram show?",
            images=[image_path],
            system_prompt="You are an expert in quantum computing. Analyze the circuit diagram.",
            model="claude-3-opus-20240229",  # Using latest model for better image analysis
            temperature=0.3,  # Lower temperature for more focused analysis
        )

        result_with_image = skill.process(image_input)

        print("\nImage Analysis Response:")
        print(result_with_image.response)
        print("\nUsage Statistics:", result_with_image.usage)


if __name__ == "__main__":
    main()
