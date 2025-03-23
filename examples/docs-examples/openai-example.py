from airtrain.integrations.openai.skills import OpenAIChatSkill, OpenAIInput
from airtrain.integrations.openai.credentials import OpenAICredentials
import os

# Initialize the skill with credentials
credentials = OpenAICredentials(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat_skill = OpenAIChatSkill(credentials=credentials)

# Create input for the model
input_data = OpenAIInput(
    user_input="How do I reset my password?",
    system_prompt="You are a helpful customer support agent.",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)

# Get response
result = chat_skill.process(input_data)
print(result.response)
