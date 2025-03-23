#!/usr/bin/env python3
"""
Example usage of AirTrain's stateful and stateless tools registry system.
This script demonstrates how to create, register and use custom tools.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..")
)
sys.path.append(parent_dir)

# Load environment variables
load_dotenv()

# Import required modules
from airtrain.tools import (
    StatefulTool,
    StatelessTool,
    register_tool,
    ToolFactory,
    execute_tool_call
)

# Import Groq integration for testing
from airtrain.integrations.groq.skills import GroqChatSkill, GroqInput as GroqChatInput


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


def demonstrate_stateful_tool():
    """Demonstrate how stateful tools maintain state per instance."""
    print("\n=== Demonstrating Stateful Tool State ===")
    
    # Create two separate instances of the stateful tool
    memory1 = ToolFactory.get_tool("conversation_memory", "stateful")
    memory2 = ToolFactory.get_tool("conversation_memory", "stateful")
    
    # Add a message to the first memory instance
    result1 = memory1(action="add", message="Hello from memory 1", role="user")
    print(f"Memory 1 after adding message: {result1}")
    
    # Add a different message to the second memory instance
    result2 = memory2(action="add", message="Hello from memory 2", role="user")
    print(f"Memory 2 after adding message: {result2}")
    
    # Show that each instance maintains its own separate state
    print("\nMemory 1 content:")
    print(json.dumps(memory1(action="get"), indent=2))
    
    print("\nMemory 2 content:")
    print(json.dumps(memory2(action="get"), indent=2))
    
    # Add more messages to memory 1
    memory1(action="add", message="This is another message for memory 1", role="user")
    memory1(action="add", message="And a third message", role="assistant")
    
    # Show final state of memory 1
    print("\nMemory 1 final content:")
    print(json.dumps(memory1(action="get"), indent=2))
    
    # Show that memory 2 remains unchanged
    print("\nMemory 2 is still separate:")
    print(json.dumps(memory2(action="get"), indent=2))


def demonstrate_stateless_tool():
    """Demonstrate how stateless tools don't maintain state between instances."""
    print("\n=== Demonstrating Stateless Tool Behavior ===")
    
    # Get two references to the calculator tool
    calc1 = ToolFactory.get_tool("calculator")
    calc2 = ToolFactory.get_tool("calculator")
    
    # Show that they are the same instance (singleton pattern)
    print(f"Are calc1 and calc2 the same instance? {calc1 is calc2}")
    
    # Use the calculator tools
    result1 = calc1(expression="2 + 2")
    print(f"Calculator result 1: {result1}")
    
    result2 = calc2(expression="10 * 5")
    print(f"Calculator result 2: {result2}")


def run_with_llm():
    """Demonstrate using the tools with an LLM."""
    print("\n=== Demonstrating Tools with LLM ===")
    
    # Initialize the skill
    skill = GroqChatSkill()
    
    # Get the tool definitions from the factory
    tools = ToolFactory.get_tool_definitions()
    
    # Create input for this turn
    input_data = GroqChatInput(
        user_input="Calculate 15 * 7 and store the result in the conversation memory.",
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use the calculator tool to perform calculations and the "
            "conversation memory tool to store important information."
        ),
        model="llama-3.3-70b-versatile",
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
        max_tokens=4096,
    )
    
    # Get response
    result = skill.process(input_data)
    
    # Check for tool calls
    if result.tool_calls:
        print(f"\nLLM made {len(result.tool_calls)} tool calls:")
        
        # Process each tool call
        for i, tool_call in enumerate(result.tool_calls):
            print(f"\n- Tool call {i+1}: {tool_call['function']['name']}")
            print(f"  Arguments: {tool_call['function']['arguments']}")
            
            # Execute the tool call
            tool_result = execute_tool_call(tool_call)
            print(f"\n  Result: {json.dumps(tool_result, indent=2)}")
            
            # Add tool result to the conversation
            if isinstance(tool_result, dict):
                result.add_tool_result(tool_call["id"], json.dumps(tool_result))
            else:
                result.add_tool_result(tool_call["id"], tool_result)
        
        # Get final response after tool calls
        final_response = skill.process_tool_results(result)
        print("\nFinal response:")
        print(final_response.content)
    else:
        print("\nNo tool calls. Direct response:")
        print(result.content)


def main():
    """Main function to run demonstrations."""
    
    # List all registered tools using the factory
    print("=== Registered Tools ===")
    all_tools = ToolFactory.list_tools()
    print(json.dumps(all_tools, indent=2))
    
    # Demonstrate stateful tool behavior
    demonstrate_stateful_tool()
    
    # Demonstrate stateless tool behavior
    demonstrate_stateless_tool()
    
    # Demonstrate tools with LLM (if API key is available)
    if os.getenv("GROQ_API_KEY"):
        try:
            run_with_llm()
        except Exception as e:
            print(f"Error running LLM example: {str(e)}")
    else:
        print("\nSkipping LLM example - GROQ_API_KEY not found in environment")


if __name__ == "__main__":
    main() 