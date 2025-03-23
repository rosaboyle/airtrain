# Airtrain

A powerful platform for building and deploying AI agents with structured skills and capabilities.

## Features

- **Structured Skills**: Build modular AI skills with defined input/output schemas
- **Multiple LLM Integrations**: Built-in support for OpenAI and Anthropic models
- **Structured Outputs**: Parse LLM responses into structured Pydantic models
- **Credential Management**: Secure handling of API keys and credentials
- **Type Safety**: Full type hints and Pydantic model support
- **Image Support**: Handle image inputs for multimodal models
- **Error Handling**: Robust error handling and logging

## Installation

```bash
pip install airtrain
```

## Quick Start

### 1. Basic OpenAI Chat

```python
from airtrain.integrations.openai.skills import OpenAIChatSkill, OpenAIInput

# Initialize the skill
skill = OpenAIChatSkill()

# Create input
input_data = OpenAIInput(
    user_input="Explain quantum computing in simple terms.",
    system_prompt="You are a helpful teacher.",
    max_tokens=500,
    temperature=0.7
)

# Get response
result = skill.process(input_data)
print(result.response)
print(f"Tokens Used: {result.usage['total_tokens']}")
```

### 2. Anthropic Claude Integration

```python
from airtrain.integrations.anthropic.skills import AnthropicChatSkill, AnthropicInput

# Initialize the skill
skill = AnthropicChatSkill()

# Create input
input_data = AnthropicInput(
    user_input="Explain the theory of relativity.",
    system_prompt="You are a physics expert.",
    model="claude-3-opus-20240229",
    temperature=0.3
)

# Get response
result = skill.process(input_data)
print(result.response)
print(f"Usage: {result.usage}")
```

### 3. Structured Output with OpenAI

```python
from pydantic import BaseModel
from typing import List
from airtrain.integrations.openai.skills import OpenAIParserSkill, OpenAIParserInput

# Define your response model
class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str
    skills: List[str]

# Initialize the parser skill
parser_skill = OpenAIParserSkill()

# Create input with response model
input_data = OpenAIParserInput(
    user_input="Tell me about John Doe, a 30-year-old software engineer who specializes in Python and AI",
    system_prompt="Extract structured information about the person.",
    response_model=PersonInfo
)

# Get structured response
result = parser_skill.process(input_data)
person_info = result.parsed_response
print(f"Name: {person_info.name}")
print(f"Skills: {', '.join(person_info.skills)}")
```

## Error Handling

All skills include built-in error handling:

```python
from airtrain.core.skills import ProcessingError

try:
    result = skill.process(input_data)
except ProcessingError as e:
    print(f"Processing failed: {e}")
```

## Advanced Features

- Image Analysis Support
- Function Calling
- Custom Validators
- Async Processing
- Token Usage Tracking

For more examples and detailed documentation, visit our [documentation](https://airtrain.readthedocs.io/).

## Documentation

For detailed documentation, visit [our documentation site](https://docs.airtrain.dev/).

## Telemetry

Airtrain collects telemetry data to help improve the library. The data collected includes:

- Agent run information (model used, task description, environment settings)
- Agent steps and actions (full action details and reasoning)
- Performance metrics (token usage, execution time, CPU and memory usage)
- System information (OS, Python version, machine details)
- Error information (complete stack traces and context)
- Model usage details (prompts, responses, parameters)

The telemetry helps us identify usage patterns, troubleshoot issues, and improve the library based on real-world usage. The user ID is stored at `~/.cache/airtrain/telemetry_user_id`.

### Disabling Telemetry

Telemetry is enabled by default, but you can disable it if needed:

1. Set an environment variable:
```bash
export AIRTRAIN_TELEMETRY_ENABLED=false
```

2. In your Python code:
```python
import os
os.environ["AIRTRAIN_TELEMETRY_ENABLED"] = "false"
```

### Viewing Telemetry Debug Information

To see what telemetry data is being sent:
```python
os.environ["AIRTRAIN_LOGGING_LEVEL"] = "debug"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 