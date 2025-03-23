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

from airtrain.integrations.sambanova.skills import SambanovaChatSkill, SambanovaInput


def run_conversation(
    skill: SambanovaChatSkill,
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Run a single conversation turn and return the assistant's response
    """
    input_data = SambanovaInput(
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        model="DeepSeek-R1-Distill-Llama-70B",
        temperature=0.7,
        max_tokens=1024,
        top_p=0.1,
    )

    result = skill.process(input_data)
    return {"role": "assistant", "content": result.response}


def main():
    # Initialize the skill
    skill = SambanovaChatSkill()

    # Define the system prompt
    system_prompt = "You are a helpful AI assistant with expertise in deep learning and neural networks."

    # Initialize conversation history
    conversation_history = []

    # Predefined conversation turns about deep learning
    conversation_turns = [
        "What is deep learning and how is it different from traditional machine learning?",
        "Can you explain what neural networks are and how they work?",
        "What are the different types of layers in a neural network?",
        "How does backpropagation work in deep learning?",
        "Can you summarize what we've discussed about deep learning and neural networks?",
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
