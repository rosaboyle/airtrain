"""
Configuration management for the chat agent.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Default configuration
DEFAULT_STORAGE_DIR = Path.home() / ".localagents" / "chat_agent" / "messages"
DEFAULT_CREDENTIALS_DIR = Path.home() / ".localagents" / "credentials"
DEFAULT_TOGETHER_CREDENTIALS_FILE = DEFAULT_CREDENTIALS_DIR / "togetherai.json"


class Config:
    """Configuration for the chat agent."""

    def __init__(self):
        """Initialize configuration with values from environment variables."""
        # Get storage directory from environment or use default
        storage_base = os.getenv("LOCALAGENTS_DIR")
        if storage_base:
            self.storage_dir = Path(storage_base) / "chat_agent" / "messages"
            self.credentials_dir = Path(storage_base) / "credentials"
        else:
            self.storage_dir = DEFAULT_STORAGE_DIR
            self.credentials_dir = DEFAULT_CREDENTIALS_DIR

        # Together AI API key - check env var first, then credentials file
        self.together_api_key = os.getenv("TOGETHER_API_KEY")

        # If not in env var, try to load from credentials file
        if not self.together_api_key:
            self.together_api_key = self._load_together_credentials()

        # Ensure storage directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_dir.mkdir(parents=True, exist_ok=True)

    def _load_together_credentials(self) -> str:
        """Load Together AI credentials from file."""
        creds_file = self.credentials_dir / "togetherai.json"
        if creds_file.exists():
            try:
                with open(creds_file, "r") as f:
                    creds = json.load(f)
                return creds.get("api_key", "")
            except (json.JSONDecodeError, IOError):
                return ""
        return ""

    def save_together_credentials(self, api_key: str) -> bool:
        """Save Together AI credentials to file."""
        try:
            # Ensure credentials directory exists
            self.credentials_dir.mkdir(parents=True, exist_ok=True)

            # Save credentials
            creds_file = self.credentials_dir / "togetherai.json"
            with open(creds_file, "w") as f:
                json.dump({"api_key": api_key}, f, indent=2)

            # Update current key
            self.together_api_key = api_key
            return True
        except IOError:
            return False

    @property
    def is_api_key_set(self) -> bool:
        """Check if the Together AI API key is set."""
        return self.together_api_key is not None and len(self.together_api_key) > 0

    def get_model_name(self) -> str:
        """Get the model name to use with Together AI."""
        return os.getenv("TOGETHER_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")

    def get_sessions_list(self) -> list[Path]:
        """Get a list of available chat sessions."""
        if not self.storage_dir.exists():
            return []

        return sorted(
            [f for f in self.storage_dir.glob("*.json")],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )


# Create a singleton config instance
config = Config()
