import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

from airtrain.integrations.together.image_skill import (
    TogetherAIImageSkill,
    TogetherAIImageInput,
)


def download_image(url: str, output_path: Path) -> None:
    """Download image from URL and save to file"""
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)


def main():
    # Initialize the skill
    skill = TogetherAIImageSkill()

    # Create input for image generation
    input_data = TogetherAIImageInput(
        prompt="A quantum computer in a futuristic laboratory, digital art style",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        n=1,
        size="1024x1024",
    )

    try:
        result = skill.process(input_data)
        print("\nImage Generation Results:")
        for i, image in enumerate(result.images, 1):
            if image.url:
                output_path = Path(f"generated_image_{i}.png")
                download_image(image.url, output_path)
                print(f"Image {i} downloaded and saved to: {output_path}")
            else:
                print(f"No URL available for image {i}")
        print("\nModel Used:", result.model)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
