from typing import Optional, List, Tuple
from datetime import date
from pydantic import Field
from airtrain.core.skills import Skill, ProcessingError
from airtrain.integrations.openai.skills import OpenAIParserSkill, OpenAIParserInput
from ..modellib.verification import UserTravelInfo
from pydantic import BaseModel


class VerificationInput(OpenAIParserInput):
    conversation_history: List[str] = Field(
        default_factory=list, description="History of conversation with user"
    )
    ask_followup: bool = Field(
        default=True, description="Whether to ask follow-up questions"
    )
    followup_question: Optional[str] = Field(
        default=None, description="Specific follow-up question to ask"
    )


class VerificationOutput(BaseModel):
    travel_info: UserTravelInfo
    needs_followup: bool
    next_question: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)


class UserVerificationAgent(OpenAIParserSkill):
    """Agent for verifying and collecting user travel information"""

    input_schema = VerificationInput
    output_schema = VerificationOutput

    def __init__(self, credentials=None):
        super().__init__(credentials)
        self.model = "gpt-4o"
        self.temperature = 0.2

    def process(self, input_data: VerificationInput) -> VerificationOutput:
        system_prompt = """
        You are a travel information verification agent. Your role is to:
        1. Extract travel information from the conversation
        2. Identify missing required information
        3. Generate appropriate follow-up questions
        4. Ensure all necessary details are collected for safe travel planning
        
        Required fields:
        - Origin location
        - Destination
        - Travel dates
        - Companions (if any)
        - Preferred outdoor activities
        - Health conditions (if any)
        
        Provide structured output and indicate if follow-up questions are needed.
        """

        # Combine conversation history into a single string
        conversation = "\n".join(input_data.conversation_history)

        # Add follow-up question if present
        if input_data.followup_question:
            conversation += f"\nFollow-up question: {input_data.followup_question}"

        input_data = OpenAIParserInput(
            user_input=conversation,
            system_prompt=system_prompt,
            response_model=VerificationOutput,
            model=self.model,
            temperature=self.temperature,
        )

        try:
            result = self.process(input_data)
            return result.parsed_response
        except Exception as e:
            raise ProcessingError(f"Failed to process verification: {str(e)}")

    def get_next_question(self, missing_fields: List[str]) -> str:
        """Generate appropriate follow-up question based on missing fields"""
        questions = {
            "origin": "What is your starting location?",
            "destination": "Where would you like to travel to?",
            "start_date": "When do you plan to start your journey?",
            "end_date": "When do you plan to return?",
            "companions": "Will anyone be traveling with you (children, pets, other adults)?",
            "outdoor_activities": "What types of outdoor activities are you interested in?",
            "health_conditions": "Do you or your companions have any health conditions we should be aware of?",
        }

        for field in missing_fields:
            if field in questions:
                return questions[field]

        return "Is there anything else you'd like to share about your travel plans?"
