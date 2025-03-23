# Perplexity AI Integration Examples

This directory contains examples demonstrating how to use the Perplexity AI integration with Airtrain.

## Setup

Before running these examples, make sure to set up your Perplexity API key:

1. Get your API key from [Perplexity AI](https://www.perplexity.ai)
2. Create an `.env` file in the root directory of your project or export in your shell:
   ```
   PERPLEXITY_API_KEY=your_api_key_here
   ```

## Available Examples

### Basic Usage
- `basic_usage.py`: Demonstrates the basic functionality of the Perplexity AI integration

### Model Listing
- `list_models.py`: Shows how to list all available Perplexity AI models and their capabilities

### Model-specific Examples
- `sonar_pro_example.py`: Examples using the sonar-pro model for advanced search with citations
- `sonar_example.py`: Examples using the lightweight sonar model
- `sonar_deep_research_example.py`: Examples using sonar-deep-research for comprehensive research
- `sonar_reasoning_example.py`: Examples using both sonar-reasoning and sonar-reasoning-pro for problem-solving
- `r1_1776_example.py`: Examples using the offline r1-1776 model

### Advanced Features
- `streaming_example.py`: Demonstrates how to use streaming capabilities with Perplexity AI models

## Model Categories

Perplexity AI provides different types of models:

1. **Search Models**
   - `sonar-pro`: Advanced search with grounding, ideal for complex queries
   - `sonar`: Lightweight model with grounding for basic search tasks

2. **Research Models**
   - `sonar-deep-research`: Expert-level research model for comprehensive reports

3. **Reasoning Models**
   - `sonar-reasoning-pro`: Premier reasoning offering with Chain of Thought
   - `sonar-reasoning`: Fast, real-time reasoning model for problem-solving

4. **Offline Models**
   - `r1-1776`: DeepSeek R1 model without search capabilities

## Running the Examples

You can run any of the examples using Python:

```bash
cd airtrain-pypi
python examples/integrations/perplexity/list_models.py
python examples/integrations/perplexity/sonar_pro_example.py
# etc.
```

Each example is self-contained and demonstrates specific capabilities of the Perplexity AI integration.

## Features Demonstrated

- Setting up credentials
- Using different models
- Adjusting parameters like temperature and max_tokens
- Working with conversation history
- Accessing citations in responses
- Streaming tokens in real-time
- Comparing results between models 