#!/usr/bin/env python3
"""
Standalone script to list Fireworks AI models using the ListModelsSkillFactory.

This script uses the airtrain library's ListModelsSkillFactory to fetch and display 
all available models from Fireworks AI in a tabular format.
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


def list_fireworks_models_factory():
    """List all available models from Fireworks AI using the ListModelsSkillFactory."""
    try:
        # Import the necessary modules for factory approach
        from airtrain.integrations import (
            ListModelsSkillFactory,
            GenericListModelsInput
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
        
        # Print supported providers
        providers = ListModelsSkillFactory.get_supported_providers()
        if HAS_RICH:
            console.print("Supported providers in factory:", ", ".join(providers))
        else:
            print("Supported providers in factory:", ", ".join(providers))
        
        # Get the Fireworks skill from the factory
        provider = "fireworks"
        skill = ListModelsSkillFactory.get_skill(provider, credentials=credentials)
        
        # Create input data for the skill with account_id
        # Note: We need to set account_id in the environment for it to work
        # via the factory as this doesn't directly translate to GenericListModelsInput
        input_data = GenericListModelsInput(api_models_only=False, account_id=account_id, page_size=100, page_token=None, order_by=None, filter=None)
        result = skill.process(input_data)
        
        if HAS_RICH:
            # Display in a table using rich
            table = Table(title="Fireworks AI Models (via Factory)")
            
            # Add columns
            table.add_column("Model Name", style="bold")
            table.add_column("Display Name")
            table.add_column("Context Length")
            table.add_column("Status")
            
            # Add rows - handle different output format from factory
            if hasattr(result, 'models'):
                # Standard format with 'models' attribute (direct skill output)
                models_list = result.models
                next_page = result.next_page_token
                total_size = result.total_size
            else:
                # GenericListModelsOutput format (factory output)
                models_list = []
                for model_dict in result.models:
                    # Convert dict to object with attribute access if needed
                    class ModelObj:
                        pass
                    model = ModelObj()
                    model.name = model_dict.get("name", "")
                    model.display_name = model_dict.get("display_name", "")
                    model.context_length = model_dict.get("context_length")
                    model.state = model_dict.get("state", "")
                    models_list.append(model)
                next_page = None
                total_size = None
                
            for model in models_list:
                context_len = ""
                if hasattr(model, "context_length") and model.context_length:
                    context_len = str(model.context_length)
                    
                table.add_row(
                    model.name,
                    getattr(model, "display_name", "") or "",
                    context_len,
                    getattr(model, "state", "") or ""
                )
            
            console.print(table)
            
            # Print pagination info if available
            if next_page:
                console.print(f"More models available. Use page token: {next_page}")
            if total_size:
                console.print(f"Total models available: {total_size}")
                
        else:
            # Display as JSON
            if hasattr(result, 'models'):
                # Direct skill output
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
            else:
                # Factory output
                output = {
                    "models": result.models,
                    "provider": result.provider
                }
            
            print(json.dumps(output, indent=2))
            
    except ImportError as e:
        print(f"Error: airtrain library not installed or missing components: {e}")
        print("Please install it with: pip install airtrain")
    except Exception as e:
        print(f"Error listing Fireworks AI models via factory: {str(e)}")


if __name__ == "__main__":
    list_fireworks_models_factory() 