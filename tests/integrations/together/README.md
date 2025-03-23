# TogetherAI Integration Tests

This directory contains tests for the TogetherAI integration in the Airtrain package.

## Test Structure

The test suite consists of the following files:

- `conftest.py` - Common fixtures for the tests
- `debug_helpers.py` - Debugging utilities
- `test_credentials.py` - Tests for the TogetherAICredentials class
- `test_chat_skill.py` - Tests for the basic TogetherAIChatSkill functionality
- `test_chat_streaming.py` - Tests specifically for the streaming functionality
- `test_rerank_skill.py` - Tests for the TogetherAIRerankSkill
- `test_image_skill.py` - Tests for the TogetherAIImageSkill

## Running the Tests

To run all TogetherAI integration tests:

```bash
python -m pytest tests/integrations/together/ -v
```

To run specific test files:

```bash
python -m pytest tests/integrations/together/test_credentials.py -v
```

## Test Coverage

The tests cover the following aspects of the TogetherAI integration:

### Credentials

- Initialization from direct input
- Initialization from environment variables
- Validation of credentials
- Error handling for invalid/missing credentials

### Chat Skill

- Basic chat functionality
- Building messages with conversation history
- Processing requests
- Error handling

### Streaming

- Streaming responses
- Handling of stream chunks
- Error handling for stream responses
- Handling null content in stream chunks

### Reranking

- Document reranking
- Different top_n values
- Model validation
- Error handling

### Image Generation

- Basic image generation
- Multiple image generation
- Different image sizes
- Using seeds for reproducibility
- Input validation for image size
- Error handling

## Mock Responses

The tests use mock responses to avoid making real API calls. The `conftest.py` file contains fixtures for these mock responses. 