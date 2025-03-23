import sys
import os
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.fireworks.structured_skills import (
    FireworksParserSkill,
    FireworksParserInput,
)


class BookReview(BaseModel):
    """Model for book review information"""

    title: str
    author: str
    genre: List[str]
    rating: float
    key_themes: List[str]
    summary: str
    recommendation: str


def run_conversation_turn(
    skill: FireworksParserSkill,
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> dict:
    """Run a single conversation turn and return the structured response"""
    input_data = FireworksParserInput(
        user_input=user_input,
        system_prompt=system_prompt,
        model="accounts/fireworks/models/deepseek-r1",
        temperature=0.3,
        response_model=BookReview,
        conversation_history=conversation_history,
    )

    result = skill.process(input_data)
    return {
        "parsed_response": result.parsed_response,
        "reasoning": result.reasoning,
        "tokens": result.tokens_used,
    }


def main():
    # Initialize the skill
    skill = FireworksParserSkill()
    conversation_history = []

    # Define system prompt
    system_prompt = (
        "You are a literary expert who provides structured analysis of books. "
        "Maintain context from previous responses when analyzing follow-up questions."
    )

    # Predefined conversation turns about a book
    conversation_turns = [
        "Can you analyze the book '1984' by George Orwell?",
        "How does the theme of surveillance in the book relate to modern society?",
        "What are the similarities between the book's Ministry of Truth and modern propaganda?",
        "Can you compare the love story in 1984 with modern dystopian novels?",
    ]

    print("\n=== Starting Structured Book Analysis Conversation ===\n")

    for turn_number, user_input in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {turn_number} ---")
        print(f"User: {user_input}\n")

        # Get structured response
        result = run_conversation_turn(
            skill, user_input, system_prompt, conversation_history
        )

        # Print structured response
        print("Structured Analysis:")
        print(result["parsed_response"].model_dump_json(indent=2))

        if result["reasoning"]:
            print("\nReasoning:")
            print(result["reasoning"])

        print(f"\nTokens Used: {result['tokens']}")

        # Add to conversation history
        conversation_history.extend(
            [
                {"role": "user", "content": user_input},
                {
                    "role": "assistant",
                    "content": result["parsed_response"].model_dump_json(),
                },
            ]
        )

        print(f"\nConversation history length: {len(conversation_history)}")


if __name__ == "__main__":
    main()
