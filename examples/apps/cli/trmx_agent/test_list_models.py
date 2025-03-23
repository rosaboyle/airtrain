#!/usr/bin/env python3
"""
Test script for the list_models functionality in trmx_agent.

This script verifies that the ListModelsSkillFactory integration is working properly.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import trmx_agent modules
sys.path.append(str(Path(__file__).parent))

from trmx_agent.config import Config
from rich.console import Console

console = Console()

def test_list_models():
    """Test the list_models functionality with different providers."""
    config = Config()
    
    # Test if we have the ListModelsSkillFactory
    try:
        from airtrain.integrations import ListModelsSkillFactory
        console.print("[green]✓ ListModelsSkillFactory is available[/green]")
        
        # Print the supported providers from the factory
        providers = ListModelsSkillFactory.get_supported_providers()
        console.print(f"Supported providers in factory: {', '.join(providers)}")
    except ImportError:
        console.print("[red]✗ ListModelsSkillFactory is not available - make sure airtrain>=0.1.45 is installed[/red]")
        return
    
    # Test listing models for each provider
    providers_to_test = ["openai", "anthropic", "together", "groq", "fireworks", "cerebras"]
    
    for provider in providers_to_test:
        console.print(f"\n[bold]Testing list_models for {provider}...[/bold]")
        try:
            config.list_models(provider)
            console.print(f"[green]✓ Successfully listed models for {provider}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to list models for {provider}: {e}[/red]")
    
    console.print("\n[bold green]Testing complete![/bold green]")

if __name__ == "__main__":
    test_list_models() 