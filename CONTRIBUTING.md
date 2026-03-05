# Contributing

## Development Setup

```bash
git clone https://github.com/sam-dumont/musicgen-api.git
cd musicgen-api
make dev        # Install all dependencies (requires uv)
make run-reload # Start dev server with auto-reload
```

## Running Tests

```bash
make test       # Unit tests
make lint       # Ruff linter
make typecheck  # mypy
make check      # All of the above
```

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes
3. Make sure `make check` passes
4. Open a PR with a clear description of what changed and why

## Code Style

- Python 3.11+
- Formatted with [ruff](https://docs.astral.sh/ruff/)
- Type hints where practical

## Reporting Issues

Open an issue with:
- What you expected vs what happened
- Steps to reproduce
- Your environment (OS, GPU, Python version, model used)
