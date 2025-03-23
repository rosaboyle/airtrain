#!/usr/bin/env python
"""Script to list Anthropic models."""

import argparse
import json
from tabulate import tabulate
from decimal import Decimal

from airtrain.integrations.anthropic import (
    AnthropicListModelsSkill,
    AnthropicListModelsInput,
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
            "List Anthropic models\n\n"
            "Note: Anthropic does not provide a public API for listing models, "
            "so this script uses the locally defined models in airtrain."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        # For local models, no credentials needed
        skill = AnthropicListModelsSkill()
        
        # Create input and process
        input_data = AnthropicListModelsInput(api_models_only=False)
        result = skill.process(input_data)
        
        # Extract model data
        models_data = result.models
        
        # Sort the data
        if args.sort_by == "name":
            models_data.sort(key=lambda x: x["id"])
        else:
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
                "Base Model", 
                "Input Price", 
                "Output Price"
            ]
            rows = []
            
            for model in models_data:
                rows.append([
                    model["id"],
                    model["display_name"],
                    model["base_model"],
                    f"${float(model['input_price'])}/1K tokens",
                    f"${float(model['output_price'])}/1K tokens",
                ])
            
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