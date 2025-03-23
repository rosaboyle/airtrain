#!/usr/bin/env python3
"""
Standalone script to list Fireworks AI models.

This script uses the airtrain library to fetch and display all available models
from Fireworks AI in a tabular format.
"""

import os
import json

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Rich library not installed. Output will be in JSON format.")
    print("Install with: pip install rich")

# Initialize console
console = Console() if HAS_RICH else None


def list_fireworks_models():
    """List all available models from Fireworks AI."""
    try:
        # Import the necessary modules
        from airtrain.integrations.fireworks.list_models import (
            FireworksListModelsSkill, 
            FireworksListModelsInput
        )
        from airtrain.integrations.fireworks.credentials import FireworksCredentials
        
        # Check if FIREWORKS_API_KEY is set
        if not os.environ.get("FIREWORKS_API_KEY"):
            print("Error: FIREWORKS_API_KEY environment variable not set.")
            print("Please set it with: export FIREWORKS_API_KEY=your_api_key")
            return
            
        # Check if account_id is set
        account_id = os.environ.get("FIREWORKS_ACCOUNT_ID")
        if not account_id:
            print("Error: FIREWORKS_ACCOUNT_ID environment variable not set.")
            print("Please set it with: export FIREWORKS_ACCOUNT_ID=your_account_id")
            return
            
        # Create credentials from environment
        credentials = FireworksCredentials.from_env()
        
        # Create the skill and input
        skill = FireworksListModelsSkill(credentials=credentials)
        input_data = FireworksListModelsInput(account_id=account_id)
        
        # Process and get models
        result = skill.process(input_data)
        
        if HAS_RICH:
            # Display in a table using rich
            table = Table(title="Fireworks AI Models")
            
            # Add columns
            table.add_column("Model Name", style="bold")
            table.add_column("Display Name")
            table.add_column("Context Length")
            table.add_column("Status")
            
            # Add rows
            for model in result.models:
                context_len = ""
                if model.context_length:
                    context_len = str(model.context_length)
                    
                table.add_row(
                    model.name,
                    model.display_name or "",
                    context_len,
                    model.state or ""
                )
            
            console.print(table)
            
            # Print pagination info if available
            if result.next_page_token:
                token_msg = f"More models available. Use page token: {result.next_page_token}"
                console.print(token_msg)
            if result.total_size:
                console.print(f"Total models available: {result.total_size}")
                
        else:
            # Display as JSON
            models_data = []
            for model in result.models:
                model_dict = {
                    "name": model.name,
                    "display_name": model.display_name,
                    "context_length": model.context_length,
                    "state": model.state
                }
                models_data.append(model_dict)
            
            output = {
                "models": models_data,
                "next_page_token": result.next_page_token,
                "total_size": result.total_size
            }
            
            print(json.dumps(output, indent=2))
            
    except ImportError:
        print("Error: airtrain library not installed or missing components.")
        print("Please install it with: pip install airtrain")
    except Exception as e:
        print(f"Error listing Fireworks AI models: {str(e)}")


if __name__ == "__main__":
    list_fireworks_models() 