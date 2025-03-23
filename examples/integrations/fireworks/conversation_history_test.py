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

from airtrain.integrations.fireworks.skills import FireworksChatSkill, FireworksInput


def run_conversation(
    skill: FireworksChatSkill,
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Run a single conversation turn and return the assistant's response
    """
    input_data = FireworksInput(
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.7,
        max_tokens=1000,
    )

    result = skill.process(input_data)
    return {"role": "assistant", "content": result.response}


def main():
    # Initialize the skill
    skill = FireworksChatSkill()

    # Define the system prompt
    system_prompt = (
        "You are a helpful AI assistant with expertise in technology and programming."
    )

    # Initialize conversation history
    conversation_history = []

    # Predefined conversation turns
    conversation_turns = [
        "Can you explain what machine learning is?",
        "What are the main types of machine learning?",
        "Can you give an example of supervised learning?",
        "How is that different from unsupervised learning?",
        "Thank you! Can you summarize what we discussed about machine learning?",
    ]

    print("\n=== Starting Conversation Test ===\n")

    # Run through each conversation turn
    for turn_number, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {turn_number} ---")
        print(f"User: {user_input}\n")

        # Get assistant's response``
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
