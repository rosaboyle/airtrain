#!/usr/bin/env python
"""Script to list OpenAI models."""

import os
import argparse
import json
from tabulate import tabulate
from decimal import Decimal

from airtrain.integrations.openai import (
    OpenAIListModelsSkill,
    OpenAIListModelsInput,
    OpenAICredentials,
)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "List OpenAI models\n\n"
            "You can optionally provide an API key either via the --api-key argument "
            "or by setting the OPENAI_API_KEY environment variable.\n"
            "Example: OPENAI_API_KEY=your_key python list_openai_models.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help=(
            "OpenAI API key (can also be set via OPENAI_API_KEY "
            "environment variable)"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="File to save the output to (optional)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "price"],
        default="name",
        help="Sort by name or input price (default: name)",
    )
    parser.add_argument(
        "--api-models-only",
        action="store_true",
        help="Fetch models from OpenAI API (requires API key)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Get API key from command line or environment variable
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    # If --api-models-only is set, an API key is required
    if args.api_models_only and not api_key:
        print("Error: API key is required when using --api-models-only.")
        print("Please either:")
        print("  1. Set the OPENAI_API_KEY environment variable:")
        print("     export OPENAI_API_KEY=your_api_key")
        print("  2. Pass the API key as a command-line argument:")
        print("     python list_openai_models.py --api-key your_api_key")
        return 1
    
    try:
        # Create skill based on whether API key is provided
        if args.api_models_only:
            # For API models, we need credentials
            try:
                credentials = OpenAICredentials(openai_api_key=api_key)
                skill = OpenAIListModelsSkill(credentials=credentials)
            except Exception as e:
                print(f"Error creating OpenAI credentials: {str(e)}")
                return 1
        else:
            # For local models, no credentials needed
            skill = OpenAIListModelsSkill()
        
        # Create input and process
        input_data = OpenAIListModelsInput(api_models_only=args.api_models_only)
        result = skill.process(input_data)
        
        # Extract model data
        models_data = result.models
        
        # Sort the data
        if args.sort_by == "name":
            models_data.sort(key=lambda x: x["id"])
        else:
            # Only sort by price if price is available
            # When using API models only, price might not be available
            if not args.api_models_only:
                models_data.sort(key=lambda x: float(x["input_price"]))
        
        # Print the result
        if args.format == "json":
            output = {"models": models_data}
            json_output = json.dumps(output, indent=2, cls=DecimalEncoder)
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    f.write(json_output)
            else:
                print(json_output)
        else:
            # Create a table
            headers = [
                "Model ID", 
                "Display Name", 
                "Base Model"
            ]
            
            # Add price columns if not using API models only
            if not args.api_models_only:
                headers.extend(["Input Price", "Output Price"])
            
            rows = []
            
            for model in models_data:
                row = [
                    model["id"],
                    model.get("display_name", ""),
                    model.get("base_model", ""),
                ]
                
                # Add price columns if available
                if not args.api_models_only:
                    row.append(f"${float(model['input_price'])}/1K tokens")
                    row.append(f"${float(model['output_price'])}/1K tokens")
                
                rows.append(row)
            
            table = tabulate(rows, headers=headers, tablefmt="simple")
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    f.write(table)
            else:
                print(table)
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 