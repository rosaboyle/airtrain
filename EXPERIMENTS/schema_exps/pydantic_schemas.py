from pydantic import BaseModel, create_model
import json
from typing import Optional, List

# 1. Define original model
class User(BaseModel):
    id: int
    name: str
    email: str

# 2. Export schema
schema = User.model_json_schema()
with open('schema.json', 'w') as f:
    json.dump(schema, f)

# 3. Load schema and create model
with open('schema.json', 'r') as f:
    loaded_schema = json.load(f)

# 4. Create model using create_model
fields = {}
for field_name, field_info in loaded_schema['properties'].items():
    field_type = {
        'integer': int,
        'string': str,
        'number': float,
        'boolean': bool
    }.get(field_info['type'])
    
    is_required = field_name in loaded_schema.get('required', [])
    fields[field_name] = (field_type, ... if is_required else None)

DynamicModel = create_model('AirtrainSchema', **fields)

# Test
user = DynamicModel(id=1, name="John", email="john@example.com")
print(user.model_dump())