#!/bin/bash
# Publish trmx package to PyPI

# Clean old builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m pip install --upgrade build
python -m build

# Publish to PyPI
python -m pip install --upgrade twine
python -m twine upload dist/*

echo "Package published successfully!" 