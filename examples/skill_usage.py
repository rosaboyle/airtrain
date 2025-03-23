import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from airtrain.core.skills import Skill, ProcessingError
from airtrain.core.schemas import InputSchema, OutputSchema
from typing import Optional, Dict


# Define schemas
class TextAnalysisInput(InputSchema):
    text: str
    language: Optional[str] = "en"
    model_preferences: Dict = {}


class TextAnalysisOutput(OutputSchema):
    sentiment: float
    confidence: float
    analysis_metadata: Dict

    def validate_output_specific(self):
        if not -1 <= self.sentiment <= 1:
            raise ValueError("Sentiment must be between -1 and 1")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


# Implement skill
class TextAnalysisSkill(Skill[TextAnalysisInput, TextAnalysisOutput]):
    input_schema = TextAnalysisInput
    output_schema = TextAnalysisOutput

    def process(self, input_data: TextAnalysisInput) -> TextAnalysisOutput:
        try:
            # Validate input
            self.validate_input(input_data)

            # Process text (dummy implementation)
            sentiment = len(input_data.text) % 2 - 0.5  # Dummy sentiment
            confidence = 0.8  # Dummy confidence

            # Create output
            output = TextAnalysisOutput(
                sentiment=sentiment,
                confidence=confidence,
                analysis_metadata={
                    "text_length": len(input_data.text),
                    "language": input_data.language,
                },
            )

            # Validate output
            self.validate_output(output)

            return output

        except Exception as e:
            raise ProcessingError(f"Processing failed: {str(e)}")


# Usage example
if __name__ == "__main__":
    # Create skill instance
    analyzer = TextAnalysisSkill()

    # Create input
    input_data = TextAnalysisInput(
        text="This is a great example!",
        language="en",
        model_preferences={"model": "fast"},
    )

    # Process
    try:
        result = analyzer.process(input_data)
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence}")
        print(f"Metadata: {result.analysis_metadata}")
    except ProcessingError as e:
        print(f"Error: {e}")
