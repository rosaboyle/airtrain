import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables
# NOTE: This example requires a valid GROQ_API_KEY environment variable
# You can set this in a .env file or directly in your environment
# Example:
#   export GROQ_API_KEY="your-api-key-here"
# or create a .env file with:
#   GROQ_API_KEY=your-api-key-here
load_dotenv()

# Add parent directory to path
parent_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..")
)
sys.path.append(parent_dir)

# Import airtrain components
from airtrain.integrations.groq.skills import GroqChatSkill, GroqInput


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
        "Sydney": {"celsius": 24, "fahrenheit": 75, "condition": "Sunny"},
        "Boston": {"celsius": 19, "fahrenheit": 66, "condition": "Windy"},
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


def get_travel_info(origin, destination):
    """
    Mock function to simulate getting travel information.
    In a real application, this would call a travel API.
    """
    # Simulate travel data
    travel_data = {
        "New York-London": {"duration": "7 hours", "price": "$650", 
                           "transport": "flight"},
        "London-Paris": {"duration": "2.5 hours", "price": "$180", 
                        "transport": "train"},
        "Tokyo-Sydney": {"duration": "9.5 hours", "price": "$720", 
                        "transport": "flight"},
        "San Francisco-New York": {"duration": "5.5 hours", "price": "$320", 
                                  "transport": "flight"},
        "Boston-New York": {"duration": "4 hours", "price": "$75", 
                           "transport": "train"},
        "Paris-Rome": {"duration": "2 hours", "price": "$210", 
                      "transport": "flight"},
    }
    
    # Try to find route in both directions
    route_key = f"{origin}-{destination}"
    reverse_route_key = f"{destination}-{origin}"
    
    if route_key in travel_data:
        data = travel_data[route_key]
        return (
            f"Travel from {origin} to {destination} by {data['transport']} "
            f"takes approximately {data['duration']} and costs around {data['price']}."
        )
    elif reverse_route_key in travel_data:
        data = travel_data[reverse_route_key]
        return (
            f"Travel from {origin} to {destination} by {data['transport']} "
            f"takes approximately {data['duration']} and costs around {data['price']}."
        )
    
    # Default response if route not found
    return (
        f"I don't have specific travel information from {origin} to {destination}, "
        f"but I recommend checking flight comparison websites or train services."
    )


def run_conversation():
    """Run an interactive conversation with the model using tool calls"""
    
    # Initialize the skill
    skill = GroqChatSkill()
    
    # Define our tools
    tools = [
        {
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
        },
        {
            "type": "function",
            "function": {
                "name": "get_travel_info",
                "description": "Get travel information between two cities",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "The city of origin"
                        },
                        "destination": {
                            "type": "string",
                            "description": "The destination city"
                        }
                    },
                    "required": ["origin", "destination"]
                }
            }
        }
    ]
    
    # Start a new conversation
    system_prompt = (
        "You are a helpful travel and weather assistant. "
        "IMPORTANT: You must use the provided functions for ALL information. "
        "You MUST use get_weather function for EACH city mentioned in weather queries. "
        "You MUST use get_travel_info function for ALL travel-related inquiries. "
        "You have NO built-in knowledge about weather or travel data. "
        "You MUST CALL BOTH FUNCTIONS when asked about weather AND travel. "
        "You MUST call each function separately for each piece of information needed. "
        "For example, if asked about weather in London and Paris, call get_weather twice, "
        "once for each city. Always use the tools to get information."
    )
    conversation_history = []
    
    # First query
    user_input = (
        "I'm planning a trip from Boston to New York. What's the weather like in both "
        "cities, and how can I travel between them?"
    )
    
    # Let's continue the conversation for multiple turns
    turn_count = 0
    max_turns = 3  # Limit to avoid infinite loops
    
    while turn_count < max_turns:
        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")
        print(f"User: {user_input}")
        
        # Create input for this turn
        input_data = GroqInput(
            user_input=user_input,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            model="llama-3.3-70b-versatile",
            tools=tools,
            # Use auto for tool_choice to let the model decide which function to call
            tool_choice="auto",
            temperature=0.2,
            max_tokens=4096,  # Safe value for most Groq models
        )
        
        # Get response
        result = skill.process(input_data)
        
        # Update conversation history with user message
        conversation_history.append({"role": "user", "content": user_input})
        
        # Check for tool calls
        if result.tool_calls:
            print("\nAssistant is calling tools...")
            print(f"Number of tool calls: {len(result.tool_calls)}")
            
            # For storing tool responses 
            tool_responses = []
            tool_call_messages = []
            
            # Process each tool call
            for i, tool_call in enumerate(result.tool_calls):
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                print(f"- Tool call {i+1}: {function_name}({arguments})")
                
                # Execute the appropriate function
                if function_name == "get_weather":
                    location = arguments.get("location")
                    unit = arguments.get("unit", "celsius")
                    tool_result = get_weather(location, unit)
                elif function_name == "get_travel_info":
                    origin = arguments.get("origin")
                    destination = arguments.get("destination")
                    tool_result = get_travel_info(origin, destination)
                else:
                    tool_result = f"Unknown function: {function_name}"
                
                print(f"  Result: {tool_result}")
                
                # Save the tool response
                tool_responses.append({
                    "tool_call_id": tool_call["id"],
                    "function_name": function_name,
                    "result": tool_result
                })
                
                # Add to messages list for the API
                tool_call_messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"]
                })
            
            # Add the assistant message with empty content (required format)
            conversation_history.append({"role": "assistant", "content": ""})
            
            # Add all tool responses
            conversation_history.extend(tool_call_messages)
            
            # Get final response with tool results
            final_input = GroqInput(
                user_input="Please summarize the information for me.",
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                model=input_data.model,
                temperature=input_data.temperature,
                max_tokens=4096,  # Safe value for most Groq models
            )
            
            final_result = skill.process(final_input)
            print("\nAssistant's response after tool calls:")
            print(final_result.response)
            
            # Add final user and assistant messages to history
            conversation_history.append(
                {"role": "user", "content": "Please summarize the information for me."}
            )
            conversation_history.append(
                {"role": "assistant", "content": final_result.response}
            )
            
        else:
            # No tool calls, just a regular response
            print("\nAssistant's response:")
            print(result.response)
            
            # Add assistant message to history
            conversation_history.append(
                {"role": "assistant", "content": result.response}
            )
        
        # For demo purposes, set the next user input
        if turn_count == 1:
            user_input = (
                "What about the weather in Paris and Tokyo? "
                "How can I travel between them?"
            )
        elif turn_count == 2:
            user_input = "Thank you for all the information!"
        else:
            break


def main():
    """Main function to demonstrate tool calling capabilities"""
    try:
        run_conversation()
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 