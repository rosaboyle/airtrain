#!/bin/bash
set -e

# Navigate to the trmx_agent directory
cd "$(dirname "$0")/.."

# Build the package first
./scripts/build.sh

echo "🚀 Publishing to PyPI..."
python -m twine upload dist/*

echo "✨ Published to PyPI successfully!" 