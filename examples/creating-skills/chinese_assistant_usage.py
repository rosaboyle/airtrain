import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.integrations.openai.chinese_assistant import (
    ChineseAssistantSkill,
    ChineseAssistantInput,
)


def main():
    # Initialize the Chinese Assistant skill
    skill = ChineseAssistantSkill()

    # Test with English input
    english_input = ChineseAssistantInput(
        user_input="What are the main differences between traditional and simplified Chinese characters?",
    )

    result = skill.process(english_input)
    print("\nQuestion in English, Response in Chinese:")
    print(result.response)
    print("\nTokens Used:", result.usage["total_tokens"])

    # Test with Chinese input
    chinese_input = ChineseAssistantInput(
        user_input="请解释人工智能机器学习的基本概念。",
        system_prompt="你是一位AI专家，请用通俗易懂的中文解释复杂的概念。",
        temperature=0.6,
    )

    result = skill.process(chinese_input)
    print("\nQuestion and Response in Chinese:")
    print(result.response)
    print("\nTokens Used:", result.usage["total_tokens"])


if __name__ == "__main__":
    main()
