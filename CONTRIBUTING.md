# Contributing to Subagents for Pydantic AI

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Getting Started

```bash
# Clone the repository
git clone https://github.com/vstorm-co/subagents-pydantic-ai.git
cd subagents-pydantic-ai

# Install dependencies
make install

# Run tests to verify setup
make test
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_toolset.py -v

# Run specific test
uv run pytest tests/test_toolset.py::test_task_sync -v

# Run with debug output
uv run pytest tests/test_toolset.py -v -s
```

### Code Quality

```bash
# Run all checks (lint + typecheck + test)
make all

# Run linting only
make lint

# Run type checking
make typecheck

# Format code
make format
```

### Pre-commit Hooks

Pre-commit hooks run automatically on commit. To run manually:

```bash
pre-commit run --all-files
```

## Code Standards

### Test Coverage

**100% test coverage is required.** Every PR must maintain full coverage.

```bash
# Check coverage
make test

# View coverage report
uv run pytest --cov=subagents_pydantic_ai --cov-report=html
open htmlcov/index.html
```

Use `# pragma: no cover` only for legitimately untestable code (e.g., platform-specific branches).

### Type Annotations

All code must pass both Pyright and MyPy strict checking:

```bash
make typecheck       # Pyright
make typecheck-mypy  # MyPy
```

### Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Keep functions focused and single-purpose

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/my-feature`
3. **Write tests** for your changes
4. **Ensure all checks pass**: `make all`
5. **Commit** with clear messages
6. **Push** to your fork
7. **Open a PR** against `main`

### PR Requirements

- [ ] All tests pass
- [ ] 100% coverage maintained
- [ ] Type checking passes
- [ ] Code is formatted
- [ ] Documentation updated (if applicable)

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom message bus
fix: handle empty subagent list gracefully
docs: update README with auto-mode examples
test: add tests for task cancellation
```

## Project Structure

```
subagents-pydantic-ai/
├── src/subagents_pydantic_ai/
│   ├── __init__.py      # Public API exports
│   ├── types.py         # Type definitions
│   ├── protocols.py     # Protocol definitions
│   ├── toolset.py       # Main toolset implementation
│   ├── factory.py       # Dynamic agent factory
│   ├── message_bus.py   # Message bus implementation
│   ├── registry.py      # Agent registry
│   └── prompts.py       # System prompts
├── tests/
│   ├── test_toolset.py  # Toolset tests
│   ├── test_types.py    # Type tests
│   └── ...
├── docs/                # Documentation
├── pyproject.toml       # Project configuration
└── Makefile             # Development commands
```

## Questions?

- Open an [issue](https://github.com/vstorm-co/subagents-pydantic-ai/issues) for bugs or feature requests
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
