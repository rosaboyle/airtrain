#!/bin/bash
set -e

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build package
python -m build

echo "Build completed successfully!" 