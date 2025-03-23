#!/usr/bin/env python
"""
Example demonstrating how telemetry works in Airtrain
"""
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional: Uncomment to disable telemetry (for testing)
# os.environ["AIRTRAIN_TELEMETRY_ENABLED"] = "false"

# Optional: Enable debug logging to see telemetry events
os.environ["AIRTRAIN_LOGGING_LEVEL"] = "debug"

import logging
logging.basicConfig(level=logging.DEBUG)

from airtrain.telemetry import (
    telemetry,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
    AgentEndTelemetryEvent,
    ModelInvocationTelemetryEvent,
    ErrorTelemetryEvent
)

# Example telemetry data
agent_id = f"example-agent-{uuid.uuid4()}"

def send_example_telemetry():
    """
    Send example telemetry events to demonstrate the telemetry system
    """
    print(f"Telemetry user ID: {telemetry.user_id}")
    print(f"Telemetry ID file location: {telemetry.USER_ID_PATH}")
    print(f"Telemetry enabled: {telemetry._posthog_client is not None}")
    print(f"Debug logging: {telemetry.debug_logging}")

    print("\nSending telemetry events...")

    # Example 1: Agent Run Event (sent when the agent starts)
    run_event = AgentRunTelemetryEvent(
        agent_id=agent_id,
        task="Search for Python documentation",
        model_name="gpt-4",
        model_provider="openai",
        version="0.1.53",
        source="example_script"
    )
    telemetry.capture(run_event)
    print(f"✓ Sent run event: {run_event.name}")

    # Example 2: Agent Step Event (sent for each step the agent takes)
    step_event = AgentStepTelemetryEvent(
        agent_id=agent_id,
        step=1,
        step_error=[],
        consecutive_failures=0,
        actions=[
            {"type": "search", "query": "Python documentation"},
            {"type": "read_content", "url": "https://docs.python.org"}
        ]
    )
    telemetry.capture(step_event)
    print(f"✓ Sent step event: {step_event.name}")

    # Example 3: Model Invocation Event
    model_event = ModelInvocationTelemetryEvent(
        agent_id=agent_id,
        model_name="gpt-4",
        model_provider="openai",
        tokens=1500,
        prompt_tokens=1200,
        completion_tokens=300,
        duration_seconds=2.5
    )
    telemetry.capture(model_event)
    print(f"✓ Sent model invocation event: {model_event.name}")

    # Example 4: Agent End Event (sent when the agent completes)
    end_event = AgentEndTelemetryEvent(
        agent_id=agent_id,
        steps=5,
        is_done=True,
        success=True,
        total_tokens=3500,
        prompt_tokens=2700,
        completion_tokens=800,
        total_duration_seconds=15.7,
        errors=[]
    )
    telemetry.capture(end_event)
    print(f"✓ Sent end event: {end_event.name}")

    # Example 5: Error Event
    error_event = ErrorTelemetryEvent(
        error_type="ConnectionError",
        error_message="Failed to connect to the API",
        component="OpenAIChatSkill",
        agent_id=agent_id
    )
    telemetry.capture(error_event)
    print(f"✓ Sent error event: {error_event.name}")

    print("\nAll example telemetry events sent!")
    print("Note: These events appear in the logs when debug logging is enabled")
    print("Telemetry data is sent to PostHog and is anonymized")
    print("To disable telemetry, set AIRTRAIN_TELEMETRY_ENABLED=false in your environment")

if __name__ == "__main__":
    send_example_telemetry() 