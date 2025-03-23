from pydantic import Field
from airtrain.core.credentials import BaseCredentials, CredentialValidationError
import lancedb


class LanceDBCredentials(BaseCredentials):
    """LanceDB credentials for embedded database"""

    database_uri: str = Field(
        default=".lancedb", description="URI/path for the embedded LanceDB database"
    )

    _required_credentials = {"database_uri"}

    async def validate_credentials(self) -> bool:
        """Validate LanceDB credentials by attempting to connect"""
        try:
            # Test connection to the database
            db = lancedb.connect(uri=self.database_uri)
            # Try to list tables to verify connection
            _ = db.table_names()
            return True
        except Exception as e:
            raise CredentialValidationError(f"Invalid LanceDB configuration: {str(e)}")
