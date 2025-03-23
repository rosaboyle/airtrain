import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), ".."))
sys.path.append(parent_dir)
from chinese_anthropic_assistant import (
    ChineseAnthropicSkill,
    ChineseAnthropicInput,
)


def main():
    # Initialize the Chinese Anthropic Assistant skill
    skill = ChineseAnthropicSkill()

    # Test with English input
    print("\nTesting English to Chinese Translation:")
    english_input = ChineseAnthropicInput(
        user_input="Explain the concept of artificial intelligence and its impact on society.",
        system_prompt="你是一位AI专家，请用通俗易懂的中文解释复杂的概念。",
        temperature=0.7,
    )

    result = skill.process(english_input)
    print("\nQuestion in English:")
    print(english_input.user_input)
    print("\nResponse in Chinese:")
    print(result.response)
    print(
        "\nTokens Used:", result.usage["input_tokens"] + result.usage["output_tokens"]
    )

    # Test with Chinese input
    print("\nTesting Chinese to Chinese Response:")
    chinese_input = ChineseAnthropicInput(
        user_input="请解释量子计算机的基本原理和应用前景。",
        system_prompt="你是一位量子计算专家，请用简单的中文解释。",
        temperature=0.6,
    )

    result = skill.process(chinese_input)
    print("\nQuestion in Chinese:")
    print(chinese_input.user_input)
    print("\nResponse in Chinese:")
    print(result.response)
    print(
        "\nTokens Used:", result.usage["input_tokens"] + result.usage["output_tokens"]
    )


if __name__ == "__main__":
    main()
