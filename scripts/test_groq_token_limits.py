#!/usr/bin/env python3
"""
Test script to verify token limit handling for Groq
"""

import os
import sys

# Add the parent directory to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import the necessary modules
from airtrain.integrations.groq.models_config import (
    get_model_config,
    get_max_completion_tokens,
    GROQ_MODELS_CONFIG,
)
from airtrain.integrations.groq.skills import GroqInput


def test_token_limits():
    """Test that token limits are correctly enforced"""
    print("Testing token limit handling for Groq models:\n")
    
    # Test all models in the config
    for model_id in GROQ_MODELS_CONFIG:
        config = get_model_config(model_id)
        max_tokens = get_max_completion_tokens(model_id)
        
        print(f"Model: {model_id}")
        print(f"  Display name: {config['name']}")
        print(f"  Context window: {config['context_window']}")
        print(f"  Max completion tokens: {max_tokens}")
        
        # Test token limit validation
        test_values = [
            1024, 
            4096, 
            8192, 
            16384, 
            32768,
            131072,
        ]
        
        print("  Testing token limit validation:")
        for value in test_values:
            input_data = GroqInput(
                user_input="Test input",
                model=model_id,
                max_tokens=value,
            )
            actual = input_data.max_tokens
            expected = min(value, max_tokens)
            result = "✓" if actual == expected else "✗"
            print(f"    {value} -> {actual} {'(limited)' if actual < value else '(unchanged)'} {result}")
        
        print()


if __name__ == "__main__":
    test_token_limits() 