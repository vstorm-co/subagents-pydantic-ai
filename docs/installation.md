# Installation

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install with uv (recommended)

```bash
uv add subagents-pydantic-ai
```

## Install with pip

```bash
pip install subagents-pydantic-ai
```

## Dependencies

The library has minimal dependencies:

- `pydantic>=2.0` - Data validation
- `pydantic-ai-slim>=0.1.0` - Core agent framework

## Environment Setup

### API Key

Subagents for Pydantic AI uses Pydantic AI which supports multiple model providers. Set your API key:

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=your-api-key
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY=your-api-key
    ```

=== "Google"

    ```bash
    export GOOGLE_API_KEY=your-api-key
    ```

## Verify Installation

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())

# Create a simple subagent
subagents = [
    SubAgentConfig(
        name="greeter",
        description="Says hello",
        instructions="You are a friendly greeter. Say hello!",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=Deps,
    toolsets=[toolset],
)

result = agent.run_sync("Say hello using the greeter subagent", deps=Deps())
print(result.output)
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the correct Python version:

```bash
python --version  # Should be 3.10+
```

### API Key Not Found

Make sure your API key is set in the environment:

```bash
echo $OPENAI_API_KEY
```

### Type Errors

If you're using strict type checking, ensure you have `typing_extensions` installed (it's included with pydantic):

```bash
pip install typing_extensions
```

## Next Steps

- [Core Concepts](concepts/index.md) - Learn the fundamentals
- [Basic Usage Example](examples/basic-usage.md) - Your first subagent
- [API Reference](api/index.md) - Complete API documentation
