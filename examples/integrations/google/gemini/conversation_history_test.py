import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel

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


class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]
    instructions: list[str]


def run_conversation(
    skill: Gemini2ChatSkill[Recipe],
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    structured_output: bool = False,
) -> Dict[str, Any]:
    """Run a single conversation turn and return the assistant's response"""
    generation_config = Gemini2GenerationConfig(
        temperature=1.0,
        response_mime_type="application/json" if structured_output else "text/plain",
        response_model=Recipe if structured_output else None,
    )

    input_data = Gemini2Input[Recipe](
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        model="gemini-2.0-flash",
        generation_config=generation_config,
        stream=False,
    )

    result = skill.process(input_data)
    return {
        "role": "assistant",
        "content": result.response,
        "parsed": result.parsed_response if structured_output else None,
    }


def run_streaming_conversation(
    skill: Gemini2ChatSkill[Recipe],
    user_input: str,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
) -> None:
    """Run a streaming conversation and print chunks as they arrive"""
    generation_config = Gemini2GenerationConfig(temperature=1.0)

    input_data = Gemini2Input[Recipe](
        user_input=user_input,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        model="gemini-2.0-flash",
        generation_config=generation_config,
        stream=True,
    )

    for chunk in skill.process_stream(input_data):
        print(chunk, end="")
    print()  # New line after streaming completes


if __name__ == "__main__":
    # Initialize the skill with the Recipe model
    skill = Gemini2ChatSkill[Recipe]()

    # Example 1: Basic conversation
    response = run_conversation(
        skill,
        "What is artificial intelligence?",
        "You are a helpful AI assistant.",
        [],
    )
    print("Basic Response:", response["content"])

    # Example 2: Conversation with history
    history = [
        {"role": "user", "content": "I want to cook something."},
        {
            "role": "assistant",
            "content": "I'd be happy to help you with cooking! What kind of dish are you interested in making?",
        },
    ]
    response = run_conversation(
        skill,
        "I'd like a cookie recipe.",
        "You are a helpful cooking assistant.",
        history,
    )
    print("\nResponse with History:", response["content"])

    # Example 3: Structured output
    response = run_conversation(
        skill,
        "Give me a recipe for chocolate chip cookies.",
        "You are a helpful cooking assistant that provides detailed recipes.",
        [],
        structured_output=True,
    )
    print("\nStructured Response:")
    if response["parsed"]:
        recipe = response["parsed"]
        print(f"Recipe Name: {recipe.recipe_name}")
        print("\nIngredients:")
        for ingredient in recipe.ingredients:
            print(f"- {ingredient}")
        print("\nInstructions:")
        for i, step in enumerate(recipe.instructions, 1):
            print(f"{i}. {step}")
    else:
        print("Failed to parse structured response")

    # Example 4: Streaming response
    print("\nStreaming Response:")
    run_streaming_conversation(
        skill,
        "Tell me a short story about a robot chef.",
        "You are a creative storyteller.",
        [],
    )
