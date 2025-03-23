#!/bin/bash
set -e

# Build the package
./scripts/build.sh

# Upload to PyPI
python -m twine upload dist/*

echo "Published to PyPI successfully!" 