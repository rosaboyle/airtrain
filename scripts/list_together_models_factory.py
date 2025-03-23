#!/usr/bin/env python3
"""
Standalone script to list Together AI models using the ListModelsSkillFactory.

This script uses the airtrain library's ListModelsSkillFactory to fetch and display 
all available models from Together AI in a tabular format.
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


def list_together_models_factory():
    """List all available models from Together AI using the ListModelsSkillFactory."""
    try:
        # Import the necessary modules for factory approach
        from airtrain.integrations import (
            ListModelsSkillFactory,
            GenericListModelsInput
        )
        from airtrain.integrations.together.credentials import TogetherAICredentials
        
        # Check if TOGETHER_API_KEY is set
        if not os.environ.get("TOGETHER_API_KEY"):
            print("Error: TOGETHER_API_KEY environment variable not set.")
            print("Please set it with: export TOGETHER_API_KEY=your_api_key")
            return
            
        # Create credentials from environment
        credentials = TogetherAICredentials.from_env()
        
        # Print supported providers
        providers = ListModelsSkillFactory.get_supported_providers()
        if HAS_RICH:
            console.print("Supported providers in factory:", ", ".join(providers))
        else:
            print("Supported providers in factory:", ", ".join(providers))
        
        # Get the Together skill from the factory
        provider = "together"
        skill = ListModelsSkillFactory.get_skill(provider, credentials=credentials)
        
        # Create input data for the skill
        input_data = GenericListModelsInput(api_models_only=False)
        
        # Process and get models
        result = skill.process(input_data)
        
        if HAS_RICH:
            # Display in a table using rich
            table = Table(title="Together AI Models (via Factory)")
            
            # Add columns
            table.add_column("Model ID", style="bold")
            table.add_column("Name")
            table.add_column("Owner")
            table.add_column("Context Length")
            
            # Add rows
            for model in result.data:
                # Factory returns models in the same way as direct call
                # because it uses the same underlying skills
                table.add_row(
                    model.id,
                    model.name or "",
                    model.owned_by or "",
                    str(model.context_length) if model.context_length else ""
                )
            
            console.print(table)
        else:
            # Display as JSON
            models_data = []
            for model in result.data:
                model_dict = {
                    "id": model.id,
                    "name": model.name,
                    "owned_by": model.owned_by,
                    "context_length": model.context_length
                }
                models_data.append(model_dict)
            
            print(json.dumps(models_data, indent=2))
            
    except ImportError as e:
        print(f"Error: airtrain library not installed or missing components: {e}")
        print("Please install it with: pip install airtrain")
    except Exception as e:
        print(f"Error listing Together AI models via factory: {str(e)}")


if __name__ == "__main__":
    list_together_models_factory() 