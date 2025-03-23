from airtrain.integrations.openai.credentials import OpenAICredentials
import os

# Create credentials
openai_creds = OpenAICredentials(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    organization_id=os.getenv("OPENAI_ORG_ID"),  # Optional
)

# Save to file
openai_creds.save_to_file("credentials.env")

# Load from file
loaded_creds = OpenAICredentials.from_file("credentials.env")

# Load to environment
loaded_creds.load_to_env()
