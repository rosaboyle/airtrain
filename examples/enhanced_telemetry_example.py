#!/usr/bin/env python
"""
Enhanced example demonstrating the extensive telemetry capabilities in Airtrain
"""
import os
import uuid
import json
import time
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable debug logging to see telemetry events
os.environ["AIRTRAIN_LOGGING_LEVEL"] = "debug"

import logging
logging.basicConfig(level=logging.DEBUG)

from airtrain.telemetry import (
    telemetry,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
    AgentEndTelemetryEvent,
    ModelInvocationTelemetryEvent,
    ErrorTelemetryEvent,
    UserFeedbackTelemetryEvent
)

# Generate example agent ID
agent_id = f"example-agent-{uuid.uuid4()}"

def send_comprehensive_telemetry():
    """
    Send comprehensive telemetry events with detailed information
    """
    print(f"Telemetry user ID: {telemetry.user_id}")
    print(f"Telemetry ID file location: {telemetry.USER_ID_PATH}")
    print(f"Telemetry enabled: {telemetry._posthog_client is not None}")
    print(f"System information: {json.dumps(telemetry.system_info, indent=2)}")

    print("\nSending detailed telemetry events...")

    # Example 1: Agent Run Event with detailed information
    sample_prompt = """
    You are an AI assistant tasked with helping the user find information about machine learning.
    Be concise, accurate, and helpful. Provide practical examples when appropriate.
    """
    
    run_event = AgentRunTelemetryEvent(
        agent_id=agent_id,
        task="Research machine learning frameworks for computer vision",
        model_name="gpt-4-turbo",
        model_provider="openai",
        version="0.1.53",
        source="enhanced_example_script",
        user_prompt=sample_prompt
    )
    telemetry.capture(run_event)
    print(f"✓ Sent detailed run event: {run_event.name}")
    print(f"  - Contains environment variables: {len(run_event.environment_variables or {})}")
    print(f"  - API key hash available: {run_event.api_key_hash is not None}")

    # Example 2: Multiple Agent Step Events with reasoning
    actions = [
        {"type": "search", "query": "best computer vision frameworks python"},
        {"type": "read_content", "url": "https://docs.opencv.org"}
    ]
    
    thinking = """
    I should start by searching for the most popular computer vision frameworks in Python.
    I'll need to find information about OpenCV, TensorFlow, PyTorch, and scikit-image
    to provide a comprehensive comparison. Let me start with a general search.
    """
    
    memory_state = {
        "topics_covered": ["computer vision", "python frameworks"],
        "frameworks_found": ["OpenCV"],
        "pending_topics": ["TensorFlow", "PyTorch", "performance comparison"]
    }
    
    for step in range(1, 4):
        step_event = AgentStepTelemetryEvent(
            agent_id=agent_id,
            step=step,
            step_error=[],
            consecutive_failures=0,
            actions=actions,
            action_details=json.dumps(actions),
            thinking=thinking,
            memory_state=memory_state
        )
        telemetry.capture(step_event)
        print(f"✓ Sent detailed step event {step}: {step_event.name}")
        time.sleep(0.5)  # Simulate time between steps

    # Example 3: Model Invocation Events with full prompt/response
    prompt = """
    Compare OpenCV and TensorFlow for computer vision tasks.
    Focus on ease of use, performance, and community support.
    """
    
    response = """
    # OpenCV vs TensorFlow for Computer Vision
    
    ## Ease of Use
    - OpenCV: More straightforward API for basic CV operations, lower learning curve
    - TensorFlow: Steeper learning curve, but more powerful for deep learning
    
    ## Performance
    - OpenCV: Optimized for classical CV algorithms, excellent CPU performance
    - TensorFlow: Better for GPU acceleration, ideal for deep learning models
    
    ## Community Support
    - OpenCV: Mature library with extensive documentation and examples
    - TensorFlow: Larger community for deep learning, more research papers and models
    
    For basic image processing tasks, OpenCV is preferable. For deep learning-based
    computer vision, TensorFlow offers more advanced capabilities.
    """
    
    model_event = ModelInvocationTelemetryEvent(
        agent_id=agent_id,
        model_name="gpt-4-turbo",
        model_provider="openai",
        tokens=1500,
        prompt_tokens=1200,
        completion_tokens=300,
        duration_seconds=2.5,
        request_id="req_abc123xyz456",
        full_prompt=prompt,
        full_response=response,
        parameters={
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 800,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    )
    telemetry.capture(model_event)
    print(f"✓ Sent detailed model invocation event: {model_event.name}")
    print(f"  - Full prompt/response captured: {len(model_event.full_prompt)} chars / {len(model_event.full_response)} chars")

    # Example 4: Error Event with stack trace
    try:
        # Generate a sample error
        x = 1 / 0
    except Exception as e:
        error_event = ErrorTelemetryEvent(
            error_type="ZeroDivisionError",
            error_message=str(e),
            component="ExampleScript",
            agent_id=agent_id,
            context={
                "operation": "division",
                "operands": {"numerator": 1, "denominator": 0},
                "current_step": 3
            }
        )
        telemetry.capture(error_event)
        print(f"✓ Sent detailed error event: {error_event.name}")
        print(f"  - Stack trace captured: {error_event.stack_trace is not None}")

    # Example 5: User Feedback Event
    feedback_event = UserFeedbackTelemetryEvent(
        agent_id=agent_id,
        rating=4,
        feedback_text="The information was mostly accurate, but could have included more examples.",
        interaction_id=f"interaction-{uuid.uuid4()}"
    )
    telemetry.capture(feedback_event)
    print(f"✓ Sent user feedback event: {feedback_event.name}")

    # Example 6: Agent End Event with full conversation
    conversation_history = [
        {"role": "system", "content": sample_prompt},
        {"role": "user", "content": "What are the best computer vision frameworks in Python?"},
        {"role": "assistant", "content": "The top computer vision frameworks in Python are:\n1. OpenCV\n2. TensorFlow\n3. PyTorch\n4. scikit-image"},
        {"role": "user", "content": "Tell me more about OpenCV and TensorFlow specifically."},
        {"role": "assistant", "content": response}
    ]
    
    end_event = AgentEndTelemetryEvent(
        agent_id=agent_id,
        steps=3,
        is_done=True,
        success=True,
        total_tokens=3500,
        prompt_tokens=2700,
        completion_tokens=800,
        total_duration_seconds=15.7,
        errors=[],
        full_conversation=conversation_history
    )
    telemetry.capture(end_event)
    print(f"✓ Sent detailed end event: {end_event.name}")
    print(f"  - CPU usage: {end_event.cpu_usage}")
    print(f"  - Memory usage: {end_event.memory_usage} MB")
    print(f"  - Full conversation included: {len(end_event.full_conversation or [])} messages")

    print("\nAll enhanced telemetry events sent!")
    print("Note: These events include much more detailed information than standard telemetry")
    print("The debug logs show the full content of each event")

if __name__ == "__main__":
    send_comprehensive_telemetry() 