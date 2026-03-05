#!/bin/bash
# Quick start script for local development
# Uses uv for fast dependency management

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or with Homebrew:"
    echo "  brew install uv"
    exit 1
fi

# Configuration
export OUTPUT_DIR="${OUTPUT_DIR:-./output}"
export API_KEY="${API_KEY:-}"

# On macOS, add xformers stub to Python path (xformers not available without CUDA)
if [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTHONPATH="$(pwd)/stubs:${PYTHONPATH:-}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install dependencies
echo "Installing dependencies with uv..."
uv sync --all-extras

# Detect device
echo ""
echo "Checking available devices..."
uv run python -c "
import torch
if torch.cuda.is_available():
    print('CUDA available - using NVIDIA GPU')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('MPS available - using Apple Metal GPU')
else:
    print('No GPU available - using CPU (will be slow)')
"

# Start the server
echo ""
echo "Starting MusicGen + Demucs API server..."
echo "  Output directory: $OUTPUT_DIR"
echo "  API Key: ${API_KEY:+configured}"
echo "  URL: http://0.0.0.0:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
