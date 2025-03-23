import sys
import os
import json
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

# Import airtrain components
from airtrain.integrations.fireworks.structured_requests_skills import (  # noqa
    FireworksStructuredRequestSkill,
    FireworksStructuredRequestInput,
)


# Define structured output models
class WeatherInfo(BaseModel):
    """Structured weather information"""
    location: str
    temperature: float
    unit: str
    condition: str
    additional_notes: Optional[str] = None


class TripPlanSummary(BaseModel):
    """Structured trip plan summary with weather"""
    destination: str
    weather: List[WeatherInfo]
    best_days_to_visit: List[str]
    activities_recommended: List[str]
    packing_recommendations: List[str]


def get_weather(location, unit="celsius"):
    """
    Mock function to simulate getting weather data.
    In a real application, this would call a weather API.
    """
    # Simulate weather data - in a real app, you'd call a weather API here
    mock_weather_data = {
        "San Francisco": {"celsius": 18, "fahrenheit": 64, "condition": "Foggy"},
        "New York": {"celsius": 22, "fahrenheit": 72, "condition": "Sunny"},
        "London": {"celsius": 15, "fahrenheit": 59, "condition": "Rainy"},
        "Tokyo": {"celsius": 26, "fahrenheit": 79, "condition": "Cloudy"},
        "Paris": {"celsius": 21, "fahrenheit": 70, "condition": "Partly Cloudy"},
    }
    
    # Default weather for locations not in our mock data
    default_weather = {"celsius": 20, "fahrenheit": 68, "condition": "Clear"}
    
    # Get the weather for the location (case insensitive)
    for known_location in mock_weather_data:
        if known_location.lower() in location.lower():
            weather = mock_weather_data[known_location]
            temp = weather[unit]
            condition = weather["condition"]
            return (
                f"The weather in {known_location} is {condition} "
                f"with a temperature of {temp}°{unit[0].upper()}"
            )
    
    # Use default weather if location not found
    temp = default_weather[unit]
    condition = default_weather["condition"]
    return (
        f"I don't have specific data for {location}, but I estimate it's {condition} "
        f"with a temperature of {temp}°{unit[0].upper()}"
    )


def run_structured_with_tools():
    """Demonstrate structured output with tool calls"""
    
    # Initialize the skill
    skill = FireworksStructuredRequestSkill()
    
    # Define our tools
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
                        "description": "The city name"
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
    
    # Create input for structured output with tools
    input_data = FireworksStructuredRequestInput(
        user_input=(
            "Create a trip plan for visiting Tokyo. Include information "
            "about weather, activities, and what to pack."
        ),
        system_prompt=(
            "You are a helpful travel assistant that provides structured trip plans. "
            "Always use the provided weather function to get current conditions. "
            "Return your response as a structured JSON object."
        ),
        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        tools=[weather_tool],
        tool_choice="auto",
        temperature=0.2,
        max_tokens=131072,
        response_model=TripPlanSummary,
    )
    
    print("\nSending structured request with tool calls...\n")
    
    try:
        # Get structured response
        result = skill.process(input_data)
        
        print("Model Used:", result.used_model)
        print(
            "Tool Calls:", 
            json.dumps(result.tool_calls, indent=2) if result.tool_calls else "None"
        )
        
        # Process tool calls if present
        if result.tool_calls:
            print("\nProcessing tool calls:")
            
            # Process each tool call
            for i, tool_call in enumerate(result.tool_calls):
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                print(f"- Tool call {i+1}: {function_name}({arguments})")
                
                # Execute the weather function
                if function_name == "get_weather":
                    location = arguments.get("location")
                    unit = arguments.get("unit", "celsius")
                    tool_result = get_weather(location, unit)
                    print(f"  Result: {tool_result}")
                else:
                    print(f"  Unknown function: {function_name}")
        
        # Display structured output
        print("\nStructured Response:")
        print(json.dumps(result.parsed_response.model_dump(), indent=2))
        
        # If reasoning is present, show it
        if result.reasoning:
            print("\nReasoning:")
            print(result.reasoning)
        
        print("\nUsage Statistics:", result.usage)
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function to demonstrate structured output with tool calls"""
    run_structured_with_tools()


if __name__ == "__main__":
    main() 