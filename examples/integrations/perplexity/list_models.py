#!/usr/bin/env python3
"""
Example demonstrating how to list all available Perplexity AI models.

This example shows:
1. How to use the PerplexityListModelsSkill
2. How to get model information including capabilities
"""

import os
import sys
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add the parent directory to the path so we can import airtrain
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from airtrain.integrations.perplexity import (
    PerplexityCredentials,
    PerplexityListModelsSkill,
    StandalonePerplexityListModelsSkill,
    PerplexityListModelsInput,
    get_models_by_category,
)
from airtrain.integrations.combined.list_models_factory import (
    GenericListModelsInput,
    ListModelsSkillFactory,
)


def print_model_details(model: Dict[str, Any]) -> None:
    """Print details of a single model.

    Args:
        model: Dictionary containing model information
    """
    print(f"- ID: {model['id']}")
    print(f"  Display Name: {model['display_name']}")
    print(f"  Description: {model.get('description', 'N/A')}")
    print(f"  Category: {model.get('category', 'N/A')}")

    # Print capabilities if available
    if "capabilities" in model:
        print("  Capabilities:")
        for key, value in model["capabilities"].items():
            print(f"    - {key}: {value}")

    print()


def main() -> None:
    """Run the Perplexity AI model listing example"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY environment variable not set")
        print("Please set it in your .env file or export it in your shell")
        sys.exit(1)

    # Set up credentials
    credentials = PerplexityCredentials(perplexity_api_key=api_key)

    print("\n=== Method 1: Using PerplexityListModelsSkill directly ===")
    list_models_skill = PerplexityListModelsSkill(credentials=credentials)
    models_output = list_models_skill.process(GenericListModelsInput())

    # Print model information
    for model in models_output.models:
        print_model_details(model)

    print("\n=== Method 2: Using Standalone skill ===")
    standalone_skill = StandalonePerplexityListModelsSkill(credentials=credentials)
    standalone_output = standalone_skill.process(PerplexityListModelsInput())

    # Print count of models
    print(f"Total models available: {len(standalone_output.models)}")
    print(f"Provider: {standalone_output.provider}")

    print("\n=== Method 3: Using ListModelsSkillFactory ===")
    factory_skill = ListModelsSkillFactory.get_skill(
        "perplexity", credentials=credentials
    )
    factory_output = factory_skill.process(GenericListModelsInput())

    print(f"Models from factory: {len(factory_output.models)}")

    print("\n=== Models by Category ===")
    # Print models grouped by category
    categories = ["search", "research", "reasoning", "offline"]

    for category in categories:
        models_in_category = get_models_by_category(category)
        print(f"\n{category.upper()} Models ({len(models_in_category)})")
        for model_id, config in models_in_category.items():
            print(f"- {model_id}: {config['name']}")
            print(f"  Description: {config.get('description', 'N/A')}")


if __name__ == "__main__":
    main()
