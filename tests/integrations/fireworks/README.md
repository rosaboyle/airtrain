# Fireworks Integration Tests

This directory contains unit tests for the Fireworks AI integration in the AirTrain library.

## Test Structure

The tests are organized as follows:

- `conftest.py` - Common fixtures and utilities for all tests
- `test_credentials.py` - Tests for the FireworksCredentials class
- `test_structured_requests_skills.py` - Tests for the main FireworksStructuredRequestSkill class
- `test_structured_requests_streaming.py` - Additional tests focusing on streaming functionality

## Running Tests

To run the tests for the Fireworks integration, use:

```bash
# From the project root directory
pytest -xvs airtrain-pypi/tests/integrations/fireworks/

# Run with coverage
pytest -xvs airtrain-pypi/tests/integrations/fireworks/ --cov=airtrain.integrations.fireworks
```

## Test Coverage

These tests aim to provide comprehensive coverage of the Fireworks integration, including:

1. Credentials management and validation
2. Message and payload construction
3. Response parsing with and without reasoning
4. Error handling
5. Streaming functionality
6. End-to-end processing

## Mocking Strategy

The tests use pytest's monkeypatch and unittest.mock to:

1. Mock API calls to Fireworks AI to avoid actual network requests
2. Mock environment variables for credential testing
3. Simulate different response scenarios including errors

## Adding New Tests

When adding new features to the Fireworks integration, please follow these guidelines:

1. Create unit tests for all new functionality
2. Mock any external dependencies
3. Test both success and failure cases
4. Aim for >90% code coverage
5. Use existing fixtures from conftest.py where appropriate 