# More things will be added here

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airtrain.core.credentials import (
    OpenAICredentials,
    AWSCredentials,
    CredentialValidationError,
)
from pathlib import Path

# Create and save OpenAI credentials
openai_creds = OpenAICredentials(api_key="sk-your-api-key", organization_id="org-123")

# Save to different formats
openai_creds.save_to_file(Path("credentials.env"))
openai_creds.save_to_file(Path("credentials.json"))
openai_creds.save_to_file(Path("credentials.yaml"))

# Load from file
loaded_creds = OpenAICredentials.from_file(Path("credentials.env"))

# Load to environment
loaded_creds.load_to_env()

# Validate credentials
try:
    loaded_creds.validate_credentials()
    print("Credentials are valid")
except CredentialValidationError as e:
    print(f"Invalid credentials: {e}")

# AWS example
aws_creds = AWSCredentials(
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    aws_region="us-west-2",
)

# Load AWS credentials to environment
aws_creds.load_to_env()

# Clear credentials from environment when done
aws_creds.clear_from_env()
