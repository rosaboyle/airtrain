from pathlib import Path
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(parent_dir)

from airtrain.integrations.together.credentials import TogetherAICredentials
from airtrain.integrations.together.image_skill import (
    TogetherAIImageSkill,
    TogetherAIImageInput,
)


def main():
    try:
        # Initialize credentials and skill
        credentials = TogetherAICredentials.from_env()
        image_skill = TogetherAIImageSkill(credentials=credentials)

        # Basic image generation example
        basic_input = TogetherAIImageInput(
            prompt="A serene landscape with mountains and a lake at sunset",
            model="stabilityai/stable-diffusion-xl-base-1.0",
            steps=20,
            n=1,
            size="1024x1024",
        )

        print("Generating basic landscape image...")
        basic_result = image_skill.process(basic_input)

        # Save the basic image
        output_dir = Path("generated_images/landscape")
        saved_paths = image_skill.save_images(basic_result, output_dir)
        print(f"Saved landscape image to: {saved_paths[0]}")
        print(f"Generation time: {basic_result.total_time:.2f} seconds")

        # Advanced example with multiple images and negative prompt
        advanced_input = TogetherAIImageInput(
            prompt="A futuristic cityscape with flying cars and neon lights",
            model="black-forest-labs/FLUX.1-schnell-Free",
            steps=30,
            n=4,  # Generate 4 variations
            size="1024x1024",
            negative_prompt="blurry, low quality, distorted",
            seed=42,  # Set seed for reproducibility
        )

        print("\nGenerating multiple futuristic cityscapes...")
        advanced_result = image_skill.process(advanced_input)

        # Save the advanced images
        output_dir = Path("generated_images/cityscape")
        saved_paths = image_skill.save_images(advanced_result, output_dir)

        print("\nGeneration Results:")
        print(f"Model used: {advanced_result.model}")
        print(f"Total images: {len(advanced_result.images)}")
        print(f"Generation time: {advanced_result.total_time:.2f} seconds")
        print("\nSaved images to:")
        for path in saved_paths:
            print(f"- {path}")

        # Print any usage statistics if available
        if advanced_result.usage:
            print("\nUsage Statistics:")
            for key, value in advanced_result.usage.items():
                print(f"{key}: {value}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
