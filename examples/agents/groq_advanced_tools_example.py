#!/usr/bin/env python3
"""
Example of using GroqAgent with advanced tools.

This script demonstrates how to use the GroqAgent with advanced tools like
command execution, directory listing, and API calls.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..")
)
sys.path.append(parent_dir)

# Load environment variables
load_dotenv()

# Import AirTrain components
from airtrain.agents import GroqAgent
from airtrain.tools import (
    ToolFactory,
    ListDirectoryTool,
    DirectoryTreeTool,
    ApiCallTool,
    ExecuteCommandTool,
    FindFilesTool
)


def setup_tools():
    """Set up and return the advanced tools for the agent."""
    try:
        # Get directory listing tool
        list_dir_tool = ToolFactory.get_tool("list_directory")
        print("Found list_directory tool")
    except ValueError:
        list_dir_tool = ListDirectoryTool()
        print("Created list_directory tool")
    
    try:
        # Get directory tree tool
        dir_tree_tool = ToolFactory.get_tool("directory_tree")
        print("Found directory_tree tool")
    except ValueError:
        dir_tree_tool = DirectoryTreeTool()
        print("Created directory_tree tool")
    
    try:
        # Get API call tool
        api_call_tool = ToolFactory.get_tool("api_call")
        print("Found api_call tool")
    except ValueError:
        api_call_tool = ApiCallTool()
        print("Created api_call tool")
    
    try:
        # Get command execution tool
        execute_cmd_tool = ToolFactory.get_tool("execute_command")
        print("Found execute_command tool")
    except ValueError:
        execute_cmd_tool = ExecuteCommandTool()
        print("Created execute_command tool")
    
    try:
        # Get find files tool
        find_files_tool = ToolFactory.get_tool("find_files")
        print("Found find_files tool")
    except ValueError:
        find_files_tool = FindFilesTool()
        print("Created find_files tool")
    
    return [
        list_dir_tool,
        dir_tree_tool,
        api_call_tool,
        execute_cmd_tool,
        find_files_tool
    ]


def create_advanced_agent():
    """Create a GroqAgent with advanced tools."""
    # Set up tools
    tools = setup_tools()
    
    # Create the agent
    agent = GroqAgent(
        name="GroqAdvancedAgent",
        memory_size=10,
        temperature=0.7,
        max_tokens=1024,
        tools=tools,
        system_prompt=(
            "You are GroqAdvancedAgent, an AI assistant with access to advanced tools. "
            "You can list directories, execute commands, make API calls, and more. "
            "Use these tools wisely to help the user. Always explain what you're doing. "
            "For security reasons, avoid executing potentially dangerous commands."
        )
    )
    
    return agent


def safe_error_handling(func):
    """Decorator for safe error handling in the interactive session."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
            return None
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Continuing session...")
            return None
    return wrapper


@safe_error_handling
def interactive_session():
    """Start an interactive session with the advanced agent."""
    print("\n=== GroqAgent with Advanced Tools ===")
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        print("Get a key from https://console.groq.com/keys")
        print("Then add it to your .env file: GROQ_API_KEY=your_key_here")
        return
    
    # Create the agent
    agent = create_advanced_agent()
    
    # Print available tools
    print("\nAvailable tools:")
    for i, tool in enumerate(agent.tools):
        print(f"{i+1}. {tool.name} - {tool.description}")
    
    # Interactive session
    print("\nChat with the agent (type 'exit' to quit):")
    print("Example commands to try:")
    print("- List the files in the current directory")
    print("- Show me the directory structure")
    print("- Execute 'ls -la' command")
    print("- Find all Python files in the current project")
    print("- Make an API call to https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&current_weather=true")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
                
            # Process with agent
            print(f"{agent.name} is thinking...")
            response = agent.process(user_input)
            print(f"\n{agent.name}: {response}")
        except KeyboardInterrupt:
            print("\nSession interrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"\nError processing request: {str(e)}")
            print("You can continue or type 'exit' to quit.")


def main():
    """Main function to demonstrate GroqAgent with advanced tools."""
    interactive_session()


if __name__ == "__main__":
    main() 