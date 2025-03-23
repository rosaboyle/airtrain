import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from airtrain.core.schemas import InputSchema, OutputSchema, ValidationError
from typing import List, Optional, Union
from pydantic import BaseModel
from datetime import datetime


# Using Pydantic model with complex types
class UserInputModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = []
    last_login: Optional[datetime] = None
    status: Union[str, int] = "active"


# Create schema from Pydantic model
UserInputSchema = InputSchema.from_pydantic_schema(UserInputModel)

# Using JSON schema with complex types
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        "status": {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
            "default": "active",
        },
    },
    "required": ["name", "age"],
}

# Create schema from JSON
JsonUserSchema = InputSchema.from_json_schema(json_schema)

# Test both approaches
pydantic_user = UserInputSchema(
    name="John", age=30, email=None, tags=["user", "active"], status=1
)

json_user = JsonUserSchema(
    name="Jane", age=25, email="jane@example.com", status="inactive"
)

# Validate
pydantic_user.validate_all()
json_user.validate_all()


# Add custom validation
class ValidatedUserSchema(UserInputSchema):
    def validate_input_specific(self):
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        if len(self.name) < 2:
            raise ValueError("Name too short")


# Test custom validation
validated_user = ValidatedUserSchema(name="Al", age=30)
try:
    validated_user.validate_all()
except ValidationError as e:
    print(f"Validation failed: {e}")

# Publish schema
schema_id = validated_user.publish()
print(f"Published schema ID: {schema_id}")
