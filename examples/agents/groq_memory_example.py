#!/usr/bin/env python3
"""
Example of using GroqAgent with persistent memory.

This script demonstrates how to use the GroqAgent with persistent
memory that saves to ~/.trmx/agents/<agent_name>/<memory_name>/uuid.json
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
from airtrain.tools import ToolFactory, register_tool, StatelessTool
from airtrain.agents import (
    GroqAgent,
    AgentFactory,
    register_agent
)


def register_example_tools():
    """Register example tools if not already registered."""
    # Check if calculator tool is already registered
    try:
        ToolFactory.get_tool("calculator")
        print("Calculator tool already registered")
    except ValueError:
        # Define a simple calculator tool
        @register_tool("calculator")
        class CalculatorTool(StatelessTool):
            """Tool for basic calculator operations."""
            
            def __init__(self):
                self.name = "calculator"
                self.description = "Perform basic arithmetic calculations"
                self.parameters = {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            
            def __call__(self, expression):
                """Execute the calculator tool."""
                try:
                    # Security: Limit evaluated expressions to basic arithmetic
                    allowed_chars = set("0123456789+-*/()%. ")
                    if not all(c in allowed_chars for c in expression):
                        return {
                            "status": "error",
                            "message": "Expression contains disallowed characters"
                        }
                    
                    # Evaluate the expression
                    result = eval(expression)
                    
                    return {
                        "status": "success",
                        "expression": expression,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "expression": expression,
                        "message": f"Error evaluating expression: {str(e)}"
                    }
            
            def to_dict(self):
                """Convert tool to dictionary format for LLM function calling."""
                return {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": self.description,
                        "parameters": self.parameters
                    }
                }
        
        print("Registered calculator tool")


def create_groq_agent():
    """Create a GroqAgent with tools."""
    # First register tools
    register_example_tools()
    
    # Get calculator tool
    calculator = ToolFactory.get_tool("calculator")
    
    # Create the agent
    agent = GroqAgent(
        name="GroqLongMemory",
        memory_size=10,
        temperature=0.7,
        tools=[calculator]
    )
    
    return agent


def test_persistent_memory():
    """Test if memory persists between agent sessions."""
    print("\n=== Testing Persistent Memory ===")
    
    # Create the agent
    agent = create_groq_agent()
    
    # Check if memory already exists
    home_dir = os.path.expanduser("~")
    memory_path = os.path.join(home_dir, ".trmx", "agents", agent.name)
    
    if os.path.exists(memory_path):
        print(f"Memory found at {memory_path}")
        print("Continuing previous conversation...\n")
    else:
        print(f"No previous memory found. Creating new at {memory_path}\n")
    
    # Interactive session
    print("Chat with the agent (type 'exit' to quit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        # Process with agent
        response = agent.process(user_input)
        print(f"{agent.name}: {response}")
    
    # Show where memory was saved
    print(f"\nMemory saved to {memory_path}")
    print("You can continue this conversation in future sessions.")


def main():
    """Main function to demonstrate GroqAgent with persistent memory."""
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        print("Get a key from https://console.groq.com/keys")
        print("Then add it to your .env file: GROQ_API_KEY=your_key_here")
        return
    
    # Run the example
    test_persistent_memory()


if __name__ == "__main__":
    main() 