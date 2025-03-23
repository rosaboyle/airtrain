from pathlib import Path
from dotenv import load_dotenv
import os
import sys
import base64

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.integrations.together.credentials import TogetherAICredentials
from airtrain.integrations.together.skills import (
    TogetherAIImageSkill,
    TogetherAIImageInput,
)


def save_image(b64_json: str, filename: str):
    """Save base64 encoded image to file"""
    image_data = base64.b64decode(b64_json)
    with open(filename, "wb") as f:
        f.write(image_data)


def main():
    try:
        # Initialize credentials and skill
        credentials = TogetherAICredentials.from_env()
        image_skill = TogetherAIImageSkill(credentials=credentials)

        # Create input for image generation
        input_data = TogetherAIImageInput(
            prompt="Cats eating popcorn",
            model="black-forest-labs/FLUX.1-schnell-Free",
            steps=10,
            n=4,
            size="1024x1024",
        )

        # Generate images
        result = image_skill.process(input_data)

        # Save generated images
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)

        for i, image_data in enumerate(result.images):
            output_path = output_dir / f"cat_popcorn_{i}.png"
            save_image(image_data.b64_json, output_path)
            print(f"Saved image to: {output_path}")

        print(f"\nGeneration Stats:")
        print(f"Model used: {result.model}")
        print(f"Total images: {len(result.images)}")

    except Exception as e:
        print(f"Error generating images: {str(e)}")


if __name__ == "__main__":
    main()
