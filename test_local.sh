#!/bin/bash
# Local testing script - uses small model on CPU for quick validation
# Run this before deploying to verify everything works

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Configuration for local CPU testing
export OUTPUT_DIR="${OUTPUT_DIR:-./test_outputs}"
export API_KEY="${API_KEY:-test-api-key-12345}"

# Use small model for CPU testing (medium requires too much RAM/time on CPU)
export MUSICGEN_MODEL="${MUSICGEN_MODEL:-facebook/musicgen-small}"

# Disable slow features for faster local testing
export USE_STEM_AWARE_CROSSFADE="${USE_STEM_AWARE_CROSSFADE:-false}"
export USE_QUALITY_LOOP="${USE_QUALITY_LOOP:-false}"

# On macOS, add xformers stub to Python path
if [[ "$OSTYPE" == "darwin"* ]]; then
    export PYTHONPATH="$(pwd)/stubs:${PYTHONPATH:-}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MusicGen API Local Testing"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model:               $MUSICGEN_MODEL"
echo "  Stem-aware crossfade: $USE_STEM_AWARE_CROSSFADE"
echo "  Quality loop:         $USE_QUALITY_LOOP"
echo "  Output directory:     $OUTPUT_DIR"
echo "  API Key:              ${API_KEY:0:10}..."
echo ""

# Install dependencies
echo "Installing dependencies..."
uv sync --all-extras
echo ""

# Check device
echo "Checking device..."
uv run python -c "
import torch
if torch.cuda.is_available():
    print('✓ CUDA available - using NVIDIA GPU')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('⚠ MPS available but not supported by audiocraft - using CPU')
else:
    print('⚠ No GPU available - using CPU (will be slow)')
print()
"

# Parse command line arguments
RUN_SERVER=true
RUN_TESTS=false

for arg in "$@"; do
    case $arg in
        --test-only)
            RUN_SERVER=false
            RUN_TESTS=true
            ;;
        --server-only)
            RUN_SERVER=true
            RUN_TESTS=false
            ;;
        --full)
            RUN_SERVER=true
            RUN_TESTS=true
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --server-only   Start the server only (default)"
            echo "  --test-only     Run tests against existing server"
            echo "  --full          Start server, run tests, then keep server running"
            echo ""
            echo "Environment variables:"
            echo "  MUSICGEN_MODEL             Model to use (default: facebook/musicgen-small)"
            echo "  USE_STEM_AWARE_CROSSFADE   Enable stem crossfade (default: false for local)"
            echo "  USE_QUALITY_LOOP           Enable quality loop (default: false for local)"
            echo "  MUSICGEN_API_URL           API URL for tests (default: http://localhost:8000)"
            exit 0
            ;;
    esac
done

if [ "$RUN_TESTS" = true ] && [ "$RUN_SERVER" = false ]; then
    # Run tests only against existing server
    echo "Running E2E tests against ${MUSICGEN_API_URL:-http://localhost:8000}..."
    export MUSICGEN_API_URL="${MUSICGEN_API_URL:-http://localhost:8000}"
    export MUSICGEN_API_KEY="$API_KEY"
    uv run pytest tests/e2e/smoke_test.py -v
    exit $?
fi

if [ "$RUN_SERVER" = true ]; then
    echo "Starting server..."
    echo "  URL: http://localhost:8000"
    echo "  Health: http://localhost:8000/health"
    echo ""
    echo "To run tests in another terminal:"
    echo "  MUSICGEN_API_URL=http://localhost:8000 MUSICGEN_API_KEY=$API_KEY ./test_local.sh --test-only"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""

    exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi
