# Chat Agent CLI

A personal chat agent that interfaces with Together AI through the AirTrain module. This CLI application allows you to have conversations with AI models and stores your chat history locally.

## Features

- Interactive chat with AI powered by Together AI
- Local storage of chat history
- Easy-to-use command-line interface
- Session management (start new chats, continue previous ones)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-repo/airtrain.git
cd airtrain/examples/apps/cli/chat_agent

# Install the package
pip install -e .
```

### From PyPI

```bash
pip install zchat
```

## Usage

After installation, you can use the chat agent with the following commands:

```bash
# Start a new chat session
z

# List previous chat sessions
z list

# Continue a previous session (multiple ways):
z 92f31c          # Using a partial session ID directly
z 92              # Even just a few characters will work
z --continue 92   # Using the --continue or -c flag
z -c 92           # Short form

# Show storage info
z info

# Show help
z --help
```

## Configuration

The chat agent can be configured using environment variables:

- `LOCALAGENTS_DIR`: Path to store chat history (default: `~/.localagents`)
- `TOGETHER_API_KEY`: Your Together AI API key (required)

You can set these in your shell or create a `.env` file in your working directory.

Credentials are loaded in the following order:
1. From environment variables
2. From the credentials file at `~/.localagents/credentials/togetherai.json`
3. If not found, the CLI will prompt you to enter your API key and offer to save it

## Requirements

- Python 3.8 or higher
- airtrain package
- Internet connection for AI model access 