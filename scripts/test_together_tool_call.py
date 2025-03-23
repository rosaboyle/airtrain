#!/usr/bin/env python3
"""
Test script for Together AI Function Calling
Simple direct test using the API without the AirTrain framework
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError(
        "TOGETHER_API_KEY environment variable not set. "
        "Please set it to your Together AI API key."
    )

# Together API endpoint
API_URL = "https://api.together.xyz/v1/chat/completions"

# Define a tool for getting current weather information
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use.",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

def test_tool_call():
    """Test Together AI function calling capability."""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to functions. "
                    "Always use functions when appropriate."
                ),
            },
            {
                "role": "user",
                "content": "What's the weather like in San Francisco? I'm planning a trip there.",
            },
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.5,
        "max_tokens": 1024,
    }

    try:
        print("Sending request to Together AI...")
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()

        # Print the response (for debugging)
        print(f"Response status code: {response.status_code}")
        print("Response JSON:")
        print(json.dumps(response_data, indent=2))

        # Check if the model generated a function call
        assistant_message = response_data.get("choices", [{}])[0].get("message", {})
        tool_calls = assistant_message.get("tool_calls", [])

        if tool_calls:
            print("\nFunction call detected!")
            for i, tool_call in enumerate(tool_calls):
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                print(f"Function {i+1}: {function_name}")
                print(f"Arguments: {arguments}")
        else:
            print("\nNo function call detected in the response.")
            print("Content:", assistant_message.get("content"))

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
    except Exception as e:
        print(f"Error: {e}")


def test_with_function_response():
    """
    Test a multi-turn conversation with function calling and response.
    This demonstrates how to handle the 'tool' message type.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    # First, send an initial message that should trigger a function call
    first_data = {
        "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful weather assistant with access to functions. "
                    "Use the get_current_weather function to answer weather-related questions."
                ),
            },
            {
                "role": "user",
                "content": "What's the weather like in New York and Tokyo right now?",
            },
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    try:
        # Step 1: Get the function call
        print("\n--- STEP 1: Getting function call ---")
        response = requests.post(API_URL, headers=headers, json=first_data)
        response.raise_for_status()
        response_data = response.json()

        # Extract the assistant's message with the function call
        assistant_message = response_data["choices"][0]["message"]
        messages = [
            {"role": "system", "content": first_data["messages"][0]["content"]},
            {"role": "user", "content": first_data["messages"][1]["content"]},
            assistant_message,
        ]

        # Check if we got function calls
        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls:
            print("No function calls detected. Exiting.")
            return

        # Print the function calls
        print(f"Got {len(tool_calls)} function calls:")
        for i, tool_call in enumerate(tool_calls):
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            print(f"  Function {i+1}: {function_name}")
            print(f"  Arguments: {arguments}")

        # Step 2: Respond to each function call with mock results
        print("\n--- STEP 2: Responding to function calls ---")
        for tool_call in tool_calls:
            tool_call_id = tool_call["id"]
            function_args = json.loads(tool_call["function"]["arguments"])
            location = function_args.get("location", "")

            # Mock weather data based on the location
            if "New York" in location:
                function_response = "Temperature: 22°C, Condition: Sunny"
            elif "Tokyo" in location:
                function_response = "Temperature: 28°C, Condition: Partly Cloudy"
            else:
                function_response = f"Weather data for {location} is not available"

            # Add function response to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": function_response,
            })
            print(f"Added mock response for {location}: {function_response}")

        # Step 3: Get final answer from the assistant
        print("\n--- STEP 3: Getting final answer ---")
        second_data = {
            "model": first_data["model"],
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        response = requests.post(API_URL, headers=headers, json=second_data)
        response.raise_for_status()
        final_response = response.json()

        # Display the final answer
        final_content = final_response["choices"][0]["message"]["content"]
        print("\nFinal answer from assistant:")
        print(final_content)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        if response:
            print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Testing Together AI Tool Calling ===")
    
    print("\n1. Testing basic tool call functionality")
    test_tool_call()
    
    print("\n2. Testing multi-turn conversation with function responses")
    test_with_function_response() 