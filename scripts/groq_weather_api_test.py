#!/usr/bin/env python3
"""
Simple test for Groq function calling API
This script allows direct testing of the API without using AirTrain
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

# API endpoint
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Weather tool definition
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

def test_tool_call(user_query="What's the weather like in Boston today?"):
    """Test the tool call API with a simple query"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": user_query}
        ],
        "tools": [weather_tool],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 4096
    }
    
    print(f"Sending request to Groq API with query: {user_query}")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Print full response for debugging
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
        # Extract tool calls
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            
            if tool_calls:
                print("\nTool calls detected:")
                for tool_call in tool_calls:
                    print(f"  Tool ID: {tool_call['id']}")
                    print(f"  Function: {tool_call['function']['name']}")
                    print(f"  Arguments: {tool_call['function']['arguments']}")
                    
                    # Parse arguments
                    try:
                        args = json.loads(tool_call['function']['arguments'])
                        location = args.get("location", "unknown")
                        unit = args.get("unit", "celsius")
                        print(f"  Parsed: location={location}, unit={unit}")
                    except json.JSONDecodeError:
                        print("  Error: Could not parse arguments JSON")
            else:
                print("No tool calls in the response. The model chose to respond directly.")
                
            # Print model's content response
            print("\nModel's content response:")
            print(message.get("content", "No content in response"))
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")


def test_with_function_response():
    """Test a conversation with a function response"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Initial user query
    query = "What's the weather like in Boston?"
    
    # First, get the function call
    first_payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": query}
        ],
        "tools": [weather_tool],
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 4096
    }
    
    try:
        # First request to get function call
        print(f"Sending initial request to get function call for: {query}")
        response = requests.post(API_URL, headers=headers, json=first_payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract tool call
        message = result["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls:
            print("No tool calls received. Ending test.")
            return
            
        tool_call = tool_calls[0]
        tool_call_id = tool_call["id"]
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])
        
        print(f"Received function call: {function_name}({function_args})")
        
        # Prepare mock function response
        mock_result = f"The weather in Boston is sunny with a temperature of 19°C"
        
        # Second request with function result
        second_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a helpful weather assistant."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": tool_call["function"]["arguments"]
                        }
                    }
                ]},
                {"role": "tool", "tool_call_id": tool_call_id, "content": mock_result}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }
        
        print("\nSending follow-up request with function result")
        response = requests.post(API_URL, headers=headers, json=second_payload)
        response.raise_for_status()
        result = response.json()
        
        print("\nFinal response:")
        print(json.dumps(result, indent=2))
        
        final_message = result["choices"][0]["message"]["content"]
        print("\nModel's final response:")
        print(final_message)
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")


if __name__ == "__main__":
    print("=== Testing simple tool call ===")
    test_tool_call()
    
    print("\n\n=== Testing conversation with function response ===")
    test_with_function_response() 