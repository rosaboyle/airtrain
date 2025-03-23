# import sys
# import os
# from pathlib import Path
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Add parent directory to path
# parent_dir = os.path.abspath(
#     os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
# )
# sys.path.append(parent_dir)

# from airtrain.integrations.openai.skills import OpenAIVisionSkill, OpenAIVisionInput


# def main():
#     # Initialize the vision skill
#     skill = OpenAIVisionSkill()

#     # Example with single image
#     image_path = Path("examples/images/quantum_circuit.jpg")

#     # Create input for image analysis
#     input_data = OpenAIVisionInput(
#         text="What does this quantum circuit diagram show?",
#         images=[image_path],
#         system_prompt="You are an expert in quantum computing. Analyze the circuit diagram.",
#         model="gpt-4o",
#         temperature=0.3,
#     )

#     try:
#         result = skill.process(input_data)
#         print("\nVision Analysis Results:")
#         print("Response:", result.response)
#         print("\nModel Used:", result.used_model)
#         print("Tokens Used:", result.tokens_used)
#         print("Images Analyzed:", result.image_count)
#     except Exception as e:
#         print(f"Error: {str(e)}")


# if __name__ == "__main__":
#     main()
