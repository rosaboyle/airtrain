# Together AI Integration with Airtrain

This directory contains examples demonstrating how to use Together AI models with Airtrain. Together AI provides access to various high-quality open-source models like Llama-3, Mixtral, and many others, with support for advanced features like tool calls.

## Features

The Together AI integration in Airtrain supports:

1. **Tool Usage** (Function Calling): Call external functions from LLM responses
2. **JSON Mode**: Generate valid JSON outputs
3. **Various Models**: Support for multiple Together AI hosted models

## Prerequisites

To use the Together AI integration, you'll need:

1. An [Together AI account](https://together.ai)
2. An API key from Together AI
3. Python 3.8+ and Airtrain installed

## Getting Started

### Setting up your API key

Set your Together AI API key as an environment variable:

```bash
export TOGETHER_API_KEY="your-api-key-here"
```

Alternatively, you can create a `.env` file in your project directory:

```
TOGETHER_API_KEY=your-api-key-here
```

### Example Scripts

This directory contains example scripts demonstrating different capabilities:

1. `tool_calls_example.py` - Demonstrates using Together AI models with tool calls (function calling)

## Tool Calls Example

The `tool_calls_example.py` script showcases how to:

1. Define tools/functions that the model can call
2. Process tool calls from the model's response
3. Handle the results of tool calls
4. Continue the conversation with tool results

The example creates a travel and weather assistant that uses:
- A `get_weather` function to retrieve weather information for cities
- A `get_travel_info` function to provide travel recommendations

### Running the Example

To run the tool calls example:

```bash
python examples/integrations/together/tool_calls_example.py
```

## Supported Models

The integration supports various Together AI models, including:

- `Qwen/Qwen2.5-72B-Instruct-Turbo` (recommended)
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.1-70B-Instruct`
- `meta-llama/Llama-3.1-405B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `mistralai/Mixtral-8x22B-Instruct-v0.1`
- More models are being added as they become available

Check the `airtrain/integrations/together/models_config.py` file for the complete list of supported models and their capabilities.

## API Reference

For more details on the API usage, refer to the main Airtrain documentation.

```python
from airtrain.integrations.together import TogetherAIChatSkill, TogetherAIInput

# Create a chat skill instance
skill = TogetherAIChatSkill()

# Create input with tools
input_data = TogetherAIInput(
    user_input="Your question here",
    model="Qwen/Qwen2.5-72B-Instruct-Turbo",
    tools=[...],  # Optional tools definitions
    tool_choice="auto",  # Can be "auto", "none", or a specific tool
    temperature=0.2,
    max_tokens=1024
)

# Process the input
result = skill.process(input_data)

# Check for tool calls
if result.tool_calls:
    # Handle tool calls
    pass
```

## Related Documentation

- [Together AI API Documentation](https://docs.together.ai/reference/chat-completions)
- [Airtrain Documentation](https://airtrain.dev) 