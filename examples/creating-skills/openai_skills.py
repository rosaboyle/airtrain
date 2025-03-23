import sys
import os
from typing import List, Optional
from pathlib import Path
import base64
from openai import OpenAI
from pydantic import Field, validator
from dotenv import load_dotenv

load_dotenv()

parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

from airtrain.core.skills import Skill, ProcessingError
from airtrain.core.schemas import InputSchema, OutputSchema

# Initialize OpenAI client
client = OpenAI()


class OpenAITextInput(InputSchema):
    """Schema for basic OpenAI text completion input"""

    user_input: str
    system_prompt: str = "You are a helpful assistant."
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v


class OpenAITextOutput(OutputSchema):
    """Schema for basic OpenAI text completion output"""

    response: str
    used_model: str
    tokens_used: int


class OpenAITextSkill(Skill[OpenAITextInput, OpenAITextOutput]):
    """Basic OpenAI text completion skill"""

    input_schema = OpenAITextInput
    output_schema = OpenAITextOutput

    def process(self, input_data: OpenAITextInput) -> OpenAITextOutput:
        try:
            # Create chat completion
            response = client.chat.completions.create(
                model=input_data.model,
                messages=[
                    {"role": "system", "content": input_data.system_prompt},
                    {"role": "user", "content": input_data.user_input},
                ],
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
            )

            return OpenAITextOutput(
                response=response.choices[0].message.content,
                used_model=response.model,
                tokens_used=response.usage.total_tokens,
            )

        except Exception as e:
            raise ProcessingError(f"OpenAI processing failed: {str(e)}")


class OpenAIVisionInput(InputSchema):
    """Schema for OpenAI vision input"""

    text: str
    images: List[Path]  # List of paths to image files
    system_prompt: str = "You are a helpful assistant that can understand images."
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    @validator("images")
    def validate_images(cls, images):
        total_size = 0
        for img_path in images:
            if not img_path.exists():
                raise ValueError(f"Image not found: {img_path}")
            size_mb = img_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            if size_mb > 4:
                raise ValueError(f"Image {img_path} exceeds 4MB limit")
        if total_size > 16:  # Total size limit
            raise ValueError("Total image size exceeds 16MB limit")
        return images


class OpenAIVisionOutput(OutputSchema):
    """Schema for OpenAI vision output"""

    response: str
    used_model: str
    tokens_used: int
    image_count: int


class OpenAIVisionSkill(Skill[OpenAIVisionInput, OpenAIVisionOutput]):
    """OpenAI vision skill for image and text analysis"""

    input_schema = OpenAIVisionInput
    output_schema = OpenAIVisionOutput

    def _encode_image(self, image_path: Path) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process(self, input_data: OpenAIVisionInput) -> OpenAIVisionOutput:
        try:
            # Prepare messages with images
            messages = [
                {"role": "system", "content": input_data.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_data.text},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self._encode_image(img_path)}"
                                },
                            }
                            for img_path in input_data.images
                        ],
                    ],
                },
            ]

            # Create chat completion
            response = client.chat.completions.create(
                model=input_data.model,
                messages=messages,
                temperature=input_data.temperature,
                max_tokens=input_data.max_tokens,
            )

            return OpenAIVisionOutput(
                response=response.choices[0].message.content or "",
                used_model=response.model,
                tokens_used=response.usage.total_tokens or 0,
                image_count=len(input_data.images),
            )

        except Exception as e:
            raise ProcessingError(f"OpenAI Vision processing failed: {str(e)}")


# Usage example
if __name__ == "__main__":
    # Test text skill
    text_skill = OpenAITextSkill()
    text_input = OpenAITextInput(
        user_input="What is the capital of France?",
        system_prompt="You are a geography expert.",
    )

    text_result = text_skill.process(text_input)
    print(f"Text Response: {text_result.response}")
    print(f"Tokens Used: {text_result.tokens_used}")

    # Test vision skill
    vision_skill = OpenAIVisionSkill()
    path_parent = Path(__file__).parent
    image_path_1 = os.path.join(path_parent, "image1.jpg")
    image_path_2 = os.path.join(path_parent, "image2.jpg")
    vision_input = OpenAIVisionInput(
        text="What do you see in these images?",
        images=[image_path_1, image_path_2],
    )

    try:
        vision_result = vision_skill.process(vision_input)
        print(f"\nVision Response: {vision_result.response}")
        print(f"Images Processed: {vision_result.image_count}")
        print(f"Tokens Used: {vision_result.tokens_used}")
    except ProcessingError as e:
        print(f"Error: {e}")
