import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.google.skills import (
    GoogleChatSkill,
    GoogleInput,
    GoogleGenerationConfig,
)


def run_conversation(
    skill: GoogleChatSkill,
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Run a single conversation turn and return the assistant's response
    """
    generation_config = GoogleGenerationConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )

    input_data = GoogleInput(
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=skill._convert_history_format(conversation_history),
        model="gemini-1.5-flash",
        generation_config=generation_config,
    )

    result = skill.process(input_data)
    return {"role": "assistant", "content": result.response}


def main():
    # Initialize the skill
    skill = GoogleChatSkill()

    # Define the system prompt
    system_prompt = "You are a helpful AI assistant with expertise in data science and machine learning."

    # Initialize conversation history
    conversation_history = []

    # Predefined conversation turns about data science
    conversation_turns = [
        "What is data science?",
        "Can you explain the difference between data science and machine learning?",
        "What are the main steps in a data science project?",
        "How does feature engineering work in data science?",
        "Can you summarize what we've discussed about data science?",
    ]

    print("\n=== Starting Conversation Test ===\n")

    # Run through each conversation turn
    for turn_number, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {turn_number} ---")
        print(f"User: {user_input}\n")

        # Get assistant's response
        assistant_response = run_conversation(
            skill, user_input, system_prompt, conversation_history
        )

        # Add both user input and assistant response to history
        conversation_history.extend(
            [{"role": "user", "content": user_input}, assistant_response]
        )

        print(f"Assistant: {assistant_response['content']}\n")
        print(f"Current conversation history length: {len(conversation_history)}")

    print("\n=== Conversation Test Complete ===")


if __name__ == "__main__":
    main()
