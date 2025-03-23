#!/usr/bin/env python3
"""
Test script for Fireworks AI Function Calling
Simple direct test using the API without the AirTrain framework
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY environment variable not set")

# API endpoint for Fireworks AI
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

# Tool definition for getting weather
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state/country, e.g. 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

# Sample request with tool
def test_tool_call():
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }

    # Test payload based on Fireworks API documentation
    payload = {
        "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful weather assistant."},
            {"role": "user", "content": "What's the weather like in San Francisco today?"}
        ],
        "tools": [weather_tool],
        "temperature": 0.7,
        "max_tokens": 131072
    }

    print(f"Sending request to Fireworks API with tool definition...")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print("\nAPI Response:")
        print(json.dumps(result, indent=2))
        
        # Check for tool calls
        if "choices" in result and result["choices"]:
            message = result["choices"][0]["message"]
            if "tool_calls" in message and message["tool_calls"]:
                print("\nTool Call Detected:")
                for tool_call in message["tool_calls"]:
                    print(f"Tool ID: {tool_call['id']}")
                    print(f"Function: {tool_call['function']['name']}")
                    print(f"Arguments: {tool_call['function']['arguments']}")
                    
                    # Parse the arguments
                    args = json.loads(tool_call['function']['arguments'])
                    print(f"\nParsed Location: {args.get('location')}")
                    print(f"Parsed Unit: {args.get('unit', 'celsius')}")
            else:
                print("\nNo tool calls in the response.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")

if __name__ == "__main__":
    test_tool_call() 