#!/usr/bin/env python3
"""
Example of using AirTrain's tool registry system with Groq.
This script demonstrates how to use both stateful and stateless tools
with Groq's API.
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import required modules first
from airtrain.tools import (
    StatefulTool,
    StatelessTool,
    register_tool,
    ToolFactory,
    execute_tool_call
)

# Import Groq integration
from airtrain.integrations.groq.skills import GroqChatSkill, GroqInput as GroqChatInput
# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..")
)
sys.path.append(parent_dir)

# Load environment variables
load_dotenv()


# Example of a stateful tool - maintains conversation history
@register_tool("conversation_memory", tool_type="stateful")
class ConversationMemoryTool(StatefulTool):
    """Tool for storing and retrieving conversation history with memory."""
    
    def __init__(self):
        self.name = "conversation_memory"
        self.description = "Store and retrieve messages from conversation history"
        self.parameters = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "get", "clear", "search"],
                    "description": "Action to perform on the conversation memory"
                },
                "message": {
                    "type": "string",
                    "description": "Message to add when action is 'add'"
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system"],
                    "description": "Role of the message when action is 'add'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return when action is 'get'"
                },
                "keyword": {
                    "type": "string",
                    "description": "Keyword to search for when action is 'search'"
                }
            },
            "required": ["action"]
        }
        self.reset()
    
    @classmethod
    def create_instance(cls):
        """Create a new instance with fresh state."""
        return cls()
    
    def reset(self):
        """Reset the conversation memory."""
        self.messages = []
        self.message_count = 0
    
    def __call__(
        self, 
        action: str,
        message: Optional[str] = None,
        role: str = "user",
        limit: int = 10,
        keyword: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the conversation memory tool."""
        if action == "add" and message:
            self.message_count += 1
            self.messages.append({
                "id": self.message_count,
                "role": role,
                "content": message,
                "timestamp": None  # Could add timestamp if needed
            })
            return {
                "status": "success",
                "action": "add",
                "message_id": self.message_count,
                "message_count": len(self.messages)
            }
        
        elif action == "get":
            if limit <= 0 or limit > 100:
                limit = 10
            return {
                "status": "success",
                "action": "get", 
                "messages": self.messages[-limit:],
                "message_count": len(self.messages)
            }
        
        elif action == "clear":
            self.reset()
            return {
                "status": "success",
                "action": "clear",
                "message_count": 0
            }
        
        elif action == "search" and keyword:
            found_messages = [
                msg for msg in self.messages 
                if keyword.lower() in msg.get("content", "").lower()
            ]
            return {
                "status": "success",
                "action": "search",
                "keyword": keyword,
                "messages": found_messages,
                "match_count": len(found_messages)
            }
        
        return {
            "status": "error",
            "message": "Invalid action or missing required parameters"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# Example of a stateless tool - calculator that doesn't maintain state
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
    
    def __call__(self, expression: str) -> Dict[str, Any]:
        """Execute the calculator tool."""
        try:
            # Security: Limit evaluated expressions to basic arithmetic
            # This is a simplified example and not secure for production
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


def run_with_groq():
    """Run an example using Groq with tool calls."""
    print("\n=== Demonstrating Tools with Groq ===")
    
    # Check if API key is available
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        return
    
    # Get instances of tools
    memory_tool = ToolFactory.get_tool("conversation_memory", "stateful")
    calculator_tool = ToolFactory.get_tool("calculator")
    
    # Initialize Groq skill
    groq_skill = GroqChatSkill()
    
    # Prepare input with tool definitions
    input_data = GroqChatInput(
        model="llama-3.3-70b-versatile",  # Model with function calling support
        user_input="Calculate 23.5 * 17 and store the result in memory.",
        conversation_history=[
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant with access to tools. "
                    "Use the calculator tool to solve math problems and the "
                    "conversation_memory tool to store important information."
                )
            },
            
        ],
        tools=[memory_tool.to_dict(), calculator_tool.to_dict()],
        temperature=0.2,
        max_tokens=4096
    )
    
    try:
        # Process the request
        result = groq_skill.process(input_data)
        
        print("\nGroq Response:")
        print("Content:", result.response)
        
        # Check for tool calls
        tool_calls = result.tool_calls
        if tool_calls:
            print("\nTool calls detected:")
            
            # Process each tool call and get results
            for i, tool_call in enumerate(tool_calls):
                print(f"\nExecuting tool call {i+1}:")
                
                # Execute the tool call
                tool_result = execute_tool_call(tool_call)
                print(f"Tool result: {json.dumps(tool_result, indent=2)}")
                
                # Create a followup message with the tool result
                followup_messages = input_data.conversation_history.copy()

                followup_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result)
                })
                
                # Get a completion incorporating the tool results
                followup_input = GroqChatInput(
                    user_input="Answer:",
                    model=input_data.model,
                    conversation_history=followup_messages,
                    temperature=input_data.temperature,
                    max_tokens=input_data.max_tokens
                )
                
                followup_result = groq_skill.process(followup_input)
                print("\nFinal response:")
                print(followup_result.response)
    
    except Exception as e:
        print(f"Error running Groq example: {str(e)}")


def main():
    """Main function to demonstrate Groq tool usage."""
    # Print registered tools
    print("Registered tools:")
    for tool_name in ToolFactory.list_tools():
        print(f"- {tool_name}")
    
    # Run the Groq example
    run_with_groq()


if __name__ == "__main__":
    main() 