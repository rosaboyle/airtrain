import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.google.gemini.skills import (
    Gemini2ChatSkill,
    Gemini2Input,
    Gemini2GenerationConfig,
)


def run_conversation(
    skill: Gemini2ChatSkill,
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> Dict[str, str]:
    """Run a single conversation turn and return the assistant's response"""
    generation_config = Gemini2GenerationConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )

    input_data = Gemini2Input(
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=skill._convert_history_format(conversation_history),
        model="gemini-2.0-flash",
        generation_config=generation_config,
    )

    result = skill.process(input_data)
    return {"role": "assistant", "content": result.response}


def main():
    skill = Gemini2ChatSkill()
    system_prompt = (
        "You are a helpful AI assistant with expertise in cybersecurity and privacy."
    )
    conversation_history = []

    conversation_turns = [
        "What are the best practices for password security?",
        "How can I protect my personal data online?",
        "What is two-factor authentication?",
        "Can you explain what encryption is?",
        "Can you summarize the key points about cybersecurity we discussed?",
    ]

    print("\n=== Starting Conversation Test ===\n")

    for turn_number, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {turn_number} ---")
        print(f"User: {user_input}\n")

        assistant_response = run_conversation(
            skill, user_input, system_prompt, conversation_history
        )

        conversation_history.extend(
            [{"role": "user", "content": user_input}, assistant_response]
        )

        print(f"Assistant: {assistant_response['content']}\n")
        print(f"Current conversation history length: {len(conversation_history)}")

    print("\n=== Conversation Test Complete ===")


if __name__ == "__main__":
    main()
