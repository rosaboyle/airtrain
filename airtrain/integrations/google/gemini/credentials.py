from pydantic import Field, SecretStr
from airtrain.core.credentials import BaseCredentials, CredentialValidationError
from google import genai
import os


class Gemini2Credentials(BaseCredentials):
    """Gemini 2.0 API credentials"""

    gemini_api_key: SecretStr = Field(..., description="Gemini API Key")

    _required_credentials = {"gemini_api_key"}

    @classmethod
    def from_env(cls) -> "Gemini2Credentials":
        """Create credentials from environment variables"""
        return cls(gemini_api_key=SecretStr(os.environ.get("GEMINI_API_KEY", "")))

    async def validate_credentials(self) -> bool:
        """Validate Gemini API credentials"""
        try:
            client = genai.Client(api_key=self.gemini_api_key.get_secret_value())
            response = client.models.generate_content(
                model="gemini-2.0-flash", contents="test"
            )
            return True
        except Exception as e:
            raise CredentialValidationError(f"Invalid Gemini 2.0 credentials: {str(e)}")
