# Makefile for musicgen-api
# Uses uv for fast Python package management

.PHONY: help install dev run test lint format typecheck check clean docker test-soundtrack

# Default target
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install      Install production dependencies"
	@echo "  dev          Install all dependencies (including dev)"
	@echo "  run          Run the FastAPI server"
	@echo "  run-reload   Run with auto-reload for development"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  test-soundtrack  Test multi-mood soundtrack generation (server must be running)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run ruff linter"
	@echo "  format       Format code with ruff"
	@echo "  typecheck    Run mypy type checker"
	@echo "  check        Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Building:"
	@echo "  docker       Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean        Remove build artifacts and cache"
	@echo "  clean-output Remove generated audio files"

# =============================================================================
# Development
# =============================================================================

install:
	uv sync --no-dev --all-extras

dev:
	uv sync --all-extras

run:
	@mkdir -p output
	PYTHONPATH="$(shell pwd)/stubs:${PYTHONPATH}" OUTPUT_DIR=./output uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

run-reload:
	@mkdir -p output
	PYTHONPATH="$(shell pwd)/stubs:${PYTHONPATH}" OUTPUT_DIR=./output uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Testing
# =============================================================================

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=app --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-soundtrack:
	@echo "Testing multi-mood soundtrack generation..."
	@echo "Make sure the server is running: make run-reload"
	@echo ""
	uv run python scripts/test_soundtrack.py

# =============================================================================
# Code Quality
# =============================================================================

lint:
	uv run ruff check app

format:
	uv run ruff format app
	uv run ruff check --fix app

typecheck:
	uv run mypy app

check: lint typecheck test

# =============================================================================
# Building
# =============================================================================

docker:
	docker build -t ghcr.io/sam-dumont/musicgen-api:latest .

docker-run:
	docker run --gpus all -p 8000:8000 \
		-e API_KEY=$${API_KEY:-} \
		-v $$(pwd)/output:/data/output \
		ghcr.io/sam-dumont/musicgen-api:latest

docker-push:
	docker push ghcr.io/sam-dumont/musicgen-api:latest

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf .venv .mypy_cache .pytest_cache .ruff_cache __pycache__ .coverage htmlcov dist build *.egg-info

clean-output:
	rm -rf output/*.wav output/*.mp3

clean-all: clean clean-output
