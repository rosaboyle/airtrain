import sys
import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.contrib.travel import (
    UserVerificationAgent,
    UserTravelInfo,
    TravelCompanion,
    HealthCondition,
)
from airtrain.integrations.openai import OpenAICredentials


def simulate_conversation(agent: UserVerificationAgent) -> UserTravelInfo:
    """
    Simulate a conversation with the verification agent.
    In a real application, this would interact with a user interface.
    """
    # Start with initial user request
    conversation = ["I want to plan a trip to Japan with my family"]

    while True:
        # Process current conversation state
        result = agent.process(VerificationInput(conversation_history=conversation))

        # Check if we have all needed information
        if not result.needs_followup:
            print("\nAll necessary information collected!")
            return result.travel_info

        # Display next question
        print(f"\nAgent: {result.next_question}")

        # In real application, this would be user input from UI
        user_response = input("User: ")

        # Add to conversation history
        conversation.append(result.next_question)
        conversation.append(user_response)

        # Display missing fields
        if result.missing_fields:
            print(f"Still missing: {', '.join(result.missing_fields)}")


def display_travel_info(info: UserTravelInfo):
    """Display collected travel information in a formatted way"""
    print("\n=== Travel Information Summary ===")
    print(f"Origin: {info.origin}")
    print(f"Destination: {info.destination}")
    print(f"Dates: {info.start_date} to {info.end_date}")

    if info.companions:
        print("\nCompanions:")
        for companion in info.companions:
            details = companion.details or {}
            print(f"- {companion.count} {companion.type}(s)")
            if details:
                for key, value in details.items():
                    print(f"  * {key}: {value}")

    if info.outdoor_activities:
        print("\nPlanned Activities:")
        for activity in info.outdoor_activities:
            print(f"- {activity}")

    if info.health_conditions:
        print("\nHealth Considerations:")
        for condition in info.health_conditions:
            print(f"- {condition.condition} (Severity: {condition.severity})")
            print(f"  Requirements: {', '.join(condition.requirements)}")


def main():
    # Initialize credentials and agent
    try:
        credentials = OpenAICredentials.from_env()
        agent = UserVerificationAgent(credentials=credentials)

        print("=== Travel Information Collection ===")
        print("Please answer the following questions to plan your trip.\n")

        # Run the conversation
        travel_info = simulate_conversation(agent)

        # Display results
        display_travel_info(travel_info)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
