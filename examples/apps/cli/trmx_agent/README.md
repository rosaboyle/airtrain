# TRMX - Terminal Chat Interface

A powerful terminal-based chat interface that lets you interact with various AI models directly from your command line. TRMX stores your conversations locally and makes it easy to manage multiple chat sessions.

## Features

- **Multiple LLM providers**: OpenAI, Anthropic, TogetherAI, Groq, Fireworks, Cerebras, Google
- **Chat history**: Store and retrieve conversations locally
- **Dynamic model listing**: Uses airtrain's ListModelsSkillFactory to fetch available models for each provider
- **Environment variable support**: Load API keys from .env files
- **Multiple prompt formats**: System prompts and multi-turn chat
- **Intuitive command-line interface**: Easy to use and navigate
- **Advanced session management**: Titles and session IDs for easy reference
- **Display options**: Custom time format display options
- **Auto-update capability**: Stay up-to-date with the latest features

## Installation

```bash
pip install trmx
```

## Quick Start

```bash
# Start a new chat using your default provider and model
trmx

# List your saved chat sessions
trmx --list

# Update TRMX to the latest version
trmx --update
```

## Detailed Usage Guide

### Managing Chat Sessions

```bash
# Start a new chat session
trmx

# List all previous chat sessions
# Shows title, ID, creation date, message count, provider, and model
trmx --list

# Continue a previous session (multiple ways)
trmx 92f31c          # Using a partial session ID directly
trmx 92              # Even just a few characters will work
trmx --continue 92   # Using the --continue or -c flag
trmx -c 92           # Short form

# Delete a chat session (by its number in the list)
trmx --delete 3      # Deletes the 3rd session in the list

# Show information about chat storage location
trmx --info
```

### Configuring Models and Providers

TRMX supports multiple AI providers including OpenAI, Anthropic, Together, Groq, Fireworks, Cerebras, and Google.

```bash
# List all available providers and their status
trmx --list-providers

# List available models for the current provider
trmx --list-models

# List models for a specific provider
trmx --list-models --provider openai   # GPT models
trmx --list-models --provider anthropic # Claude models
trmx --list-models --provider groq     # Llama and other models
```

TRMX dynamically fetches the latest available models from each provider using AirTrain's ListModelsSkillFactory, ensuring you always have access to the most up-to-date model options without requiring credentials for some providers like OpenAI.

```bash
# Use a specific provider and model for a single chat session
trmx --provider openai --model gpt-4
trmx --provider anthropic --model claude-3-opus-20240229
trmx --provider groq --model llama-3-70b-8192

# Set a new default provider/model configuration
trmx --add --provider openai --model gpt-4-turbo
trmx --add --provider anthropic --model claude-3-haiku-20240307
```

### Display Settings

```bash
# Set the time display style for chat sessions
trmx --set-timestyle iso      # Display times in ISO format (2025-03-17T22:55:28)
trmx --set-timestyle human    # Display times in human-readable format (2025-03-17 22:55:28)
trmx --set-timestyle relative # Display times in relative format (2 hours ago)

# Show model's thinking process (for supported models like DeepSeek)
trmx --provider fireworks --model fireworks/deepseek-r1 --show-thinking
```

### Maintenance

```bash
# Check the current version
trmx --version

# Update to the latest version
trmx --update

# Show help information
trmx --help
```

## Chat Interface Features

During a chat session:
- The provider and model information are displayed prominently
- Chat history is shown when continuing a session
- Type `exit`, `quit`, or `q` to end the session
- Multi-line input is supported:
  - Use `/m`, `/multiline`, `/multi`, `/p`, or `/paste` (end with `/end`)
  - Use triple quotes `"""` or `'''` (end with corresponding triple quotes)

## Configuration

TRMX can be configured using environment variables:

- `TRMX_DIR`: Path to store chat history, credentials, and configuration (default: `~/.trmx`)
- API key variables for each provider (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)

You can set these in your shell or create a `.env` file in your working directory.

### API Keys

TRMX will search for API keys in this order:
1. Environment variables 
2. Credential files in `~/.trmx/credentials/`
3. Interactive prompt (if not found, TRMX will ask if you want to enter and save the key)

### Example Configuration

For OpenAI:
```bash
export OPENAI_API_KEY=your-key-here
```

For Anthropic:
```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Session Information

When you list your sessions with `trmx --list`, you'll see:
- Session title (auto-generated from the conversation)
- Session ID (unique identifier)
- Creation time
- Message count
- Provider (which AI service was used)
- Model (which specific model was used)
- Preview of the conversation

## Requirements

- Python 3.8 or higher
- Internet connection for AI model access 