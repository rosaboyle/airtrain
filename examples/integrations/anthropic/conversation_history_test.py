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

from airtrain.integrations.anthropic.skills import AnthropicChatSkill, AnthropicInput


def run_conversation(
    skill: AnthropicChatSkill,
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Run a single conversation turn and return the assistant's response
    """
    input_data = AnthropicInput(
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        model="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=1024,
    )

    result = skill.process(input_data)
    return {"role": "assistant", "content": result.response}


def main():
    # Initialize the skill
    skill = AnthropicChatSkill()

    # Define the system prompt
    system_prompt = (
        "You are a helpful AI assistant with expertise in science and physics."
    )

    # Initialize conversation history
    conversation_history = []

    # Predefined conversation turns about physics
    conversation_turns = [
        "What is quantum mechanics?",
        "How does the uncertainty principle work?",
        "Can you explain wave-particle duality?",
        "What are quantum entanglement and superposition?",
        "Can you summarize what we've discussed about quantum mechanics?",
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
