#!/usr/bin/env python3
"""
Example of using AirTrain's agent system.

This script demonstrates how to use the agent system with memory
and tools to create intelligent agents.
"""

import os
import sys
from dotenv import load_dotenv
from airtrain.tools import ToolFactory, register_tool, StatelessTool
from airtrain.agents import (
    BaseAgent,
    register_agent,
    AgentFactory,
    SharedMemory
)

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..")
)
sys.path.append(parent_dir)

# Load environment variables
load_dotenv()


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


@register_agent("echo_agent")
class EchoAgent(BaseAgent):
    """Simple agent that echoes user input and demonstrates memory."""

    def __init__(self, name, models=None, tools=None):
        super().__init__(name, models, tools)
        # Add a specialized memory for reasoning
        self.create_memory("thoughts", 5)

    def process(self, user_input, memory_name="default"):
        """Process user input and return a response."""
        # Add to memory
        self.memory.add_to_all({"role": "user", "content": user_input})

        # Add some "thinking" to the thoughts memory
        self.memory.add_to_memory(
            "thoughts",
            {"role": "system", "content": f"Thinking about: {user_input}"}
        )

        # Generate response - here just echoing with history info
        context = self.memory.get_context(memory_name)
        message_count = len(context)

        response = (
            f"Echo: {user_input}\n"
            f"Message count in memory: {message_count}"
        )

        # Add response to memory
        self.memory.add_to_all({"role": "assistant", "content": response})

        return response


def create_agent_with_tools():
    """Create an agent with tools."""
    try:
        # First register example tools
        register_example_tools()

        # Get calculator tool
        calculator = ToolFactory.get_tool("calculator")

        # Create the agent
        agent = AgentFactory.create_agent(
            agent_type="echo_agent",
            name="ToolEchoAgent",
            tools=[calculator]
        )

        return agent
    except Exception as e:
        print(f"Error creating agent with tools: {str(e)}")
        return None


def create_agent_team():
    """Create a team of agents with shared memory."""
    # Create shared memory
    shared_memory = SharedMemory("team_knowledge")

    # Create agents
    agent1 = AgentFactory.create_agent("echo_agent", name="Agent1")
    agent2 = AgentFactory.create_agent("echo_agent", name="Agent2")

    # Add shared memory to both agents
    agent1.memory.add_shared_memory(shared_memory)
    agent2.memory.add_shared_memory(shared_memory)

    return agent1, agent2


def main():
    """Main function to demonstrate the agent system."""
    print("\n=== AirTrain Agent System Example ===\n")

    # Example 1: Basic agent with tools
    print("Creating agent with tools...")
    agent = create_agent_with_tools()

    if agent:
        print(f"\nCreated agent: {agent.name}")
        print(f"Available tools: {len(agent.tools)}")

        # Test the agent
        print("\n--- Testing Basic Agent ---")
        inputs = [
            "Hello, what can you do?",
            "This is a second message to demonstrate memory",
            "What's in your memory now?"
        ]

        for i, user_input in enumerate(inputs):
            print(f"\nUser: {user_input}")
            response = agent.process(user_input)
            print(f"{agent.name}: {response}")

    # Example 2: Agent team with shared memory
    print("\n\n--- Testing Agent Team with Shared Memory ---")
    agent1, agent2 = create_agent_team()

    # Test shared memory
    print("\nAgent1 adds information to shared memory:")
    response1 = agent1.process("The secret code is 12345")
    print(f"{agent1.name}: {response1}")

    print("\nAgent2 tries to access information from shared memory:")
    response2 = agent2.process("What was the secret code?")
    print(f"{agent2.name}: {response2}")


if __name__ == "__main__":
    main()
