#!/bin/bash
set -e

# Navigate to the trmx_agent directory
cd "$(dirname "$0")/.."

echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

echo "📦 Building package..."
python -m build

echo "✅ Build completed successfully!" 