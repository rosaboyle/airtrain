#!/usr/bin/env python
"""
Test script for testing OpenAI model listing with ListModelsSkillFactory
"""

import sys
from loguru import logger

try:
    from airtrain.integrations import (
        ListModelsSkillFactory,
        GenericListModelsInput
    )
except ImportError:
    print("Could not import ListModelsSkillFactory. Make sure airtrain >= 0.1.45 is installed.")
    sys.exit(1)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_openai_models():
    logger.info("Testing OpenAI model listing with ListModelsSkillFactory")
    
    try:
        # Get the factory
        logger.info("Creating ListModelsSkillFactory")
        factory = ListModelsSkillFactory()
        
        # Get supported providers
        providers = factory.get_supported_providers()
        logger.info(f"Supported providers: {providers}")
        
        # Get the skill for OpenAI
        logger.info("Getting skill for OpenAI provider")
        skill = factory.get_skill("openai")
        
        # For OpenAI, we can get models without credentials
        logger.info("Configuring for OpenAI (without credentials)")
        input_data = GenericListModelsInput(
            api_models_only=False
        )
        
        logger.info("Processing with OpenAI skill")
        result = skill.process(input_data)
        
        if hasattr(result, "is_success"):
            success = result.is_success()
            error = result.error if hasattr(result, "error") else None
            logger.info(f"Process result: success={success}, error={error}")
        else:
            logger.info(f"Result type: {type(result)}")
        
        # Check if we got models
        if hasattr(result, 'models'):
            models = result.models
            logger.info(f"Found {len(models)} models")
            for model in models[:10]:  # Show first 10 models
                logger.info(f"Model: {model}")
            logger.info(f"Total models found: {len(models)}")
            return True
        elif hasattr(result, 'data'):
            data = result.data
            logger.info(f"Found data with {len(data)} items")
            for item in data[:10]:  # Show first 10 items
                logger.info(f"Item: {item}")
            return True
        else:
            logger.warning(f"Result has unexpected structure: {dir(result)}")
            return False
    except Exception as e:
        logger.exception(f"Error testing OpenAI models: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_models()
    sys.exit(0 if success else 1) 