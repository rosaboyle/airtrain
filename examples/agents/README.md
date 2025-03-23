# AirTrain Agent Examples

This directory contains example scripts demonstrating how to use the AirTrain agent system.

## Setup

1. Copy `.env.example` to `.env` in this directory
2. Add your API keys to the `.env` file (Groq, Fireworks, etc.)

## Available Examples

### 1. Basic Agent Example (`agent_example.py`)

A simple example demonstrating the agent system with memory. This example uses an echo agent that doesn't require any API keys.

```bash
python agent_example.py
```

### 2. Groq Memory Example (`groq_memory_example.py`)

Demonstrates a Groq-powered agent with persistent memory that saves to `~/.trmx/agents/<agent_name>/<memory_name>/uuid.json`.

Requirements:
- Groq API key in `.env` file

```bash
python groq_memory_example.py
```

### 3. Advanced Tools Example (`groq_advanced_tools_example.py`)

Demonstrates using the Groq agent with advanced tools:
- Command execution
- Directory listing
- Directory tree
- File finding
- API calls

Requirements:
- Groq API key in `.env` file

```bash
python groq_advanced_tools_example.py
```

### Example Commands for Advanced Tools

- "List the files in the current directory"
- "Show me the directory structure"
- "Execute 'ls -la' command"
- "Find all Python files in the current project"
- "Make an API call to https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true"

## Creating Your Own Agents

To create your own agent:

1. Subclass `BaseAgent` from `airtrain.agents`
2. Use the `@register_agent` decorator
3. Implement the `process` method
4. (Optional) Override other methods as needed 