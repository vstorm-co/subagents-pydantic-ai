# Quickstart

Get started with subagents-pydantic-ai in minutes.

## Installation

```bash
pip install subagents-pydantic-ai
```

Or with uv:

```bash
uv add subagents-pydantic-ai
```

## Basic Setup

### 1. Create Dependencies

Your dependencies class must implement `SubAgentDepsProtocol`:

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Deps:
    """Dependencies for the agent."""

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        """Create isolated deps for a subagent."""
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
        )
```

### 2. Define Subagents

Create configurations for your specialized subagents:

```python
from subagents_pydantic_ai import SubAgentConfig

subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics and gathers information",
        instructions="""You are a research assistant.

When given a topic:
1. Break it down into key aspects
2. Investigate each aspect thoroughly
3. Provide a comprehensive summary""",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes content based on provided information",
        instructions="""You are a technical writer.

Write clear, concise content using:
- Simple language
- Short paragraphs
- Bullet points where appropriate""",
    ),
    SubAgentConfig(
        name="reviewer",
        description="Reviews and improves content",
        instructions="Review content for clarity, accuracy, and style.",
        can_ask_questions=False,  # Works independently
    ),
]
```

### 3. Create Toolset

```python
from subagents_pydantic_ai import create_subagent_toolset

toolset = create_subagent_toolset(
    subagents=subagents,
    default_model="openai:gpt-4o",
    include_general_purpose=True,  # Adds fallback agent
)
```

### 4. Create Agent

```python
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You are a helpful assistant with specialized subagents.

Delegate tasks to the appropriate subagent:
- Use 'researcher' for gathering information
- Use 'writer' for creating content
- Use 'reviewer' for editing

For long tasks, use mode="async" to run in background.""",
)
```

### 5. Run

```python
async def main():
    result = await agent.run(
        "Research the benefits of async programming and write a blog post",
        deps=Deps(),
    )
    print(result.output)

import asyncio
asyncio.run(main())
```

## Complete Example

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

# Define subagents
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics thoroughly",
        instructions="You are a research assistant. Investigate topics in depth.",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes clear content",
        instructions="You are a technical writer. Write clear, engaging content.",
    ),
]

# Create agent with subagent toolset
toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate tasks to specialized subagents.",
)

# Run
result = agent.run_sync("Research Python decorators and explain them", deps=Deps())
print(result.output)
```

## Give Subagents Tools

Use `toolsets_factory` to provide tools:

```python
from pydantic_ai_backends import create_console_toolset

def create_subagent_tools(deps):
    """Factory that creates toolsets for subagents."""
    return [
        create_console_toolset(),  # File operations
    ]

toolset = create_subagent_toolset(
    subagents=subagents,
    toolsets_factory=create_subagent_tools,
)
```

## Dynamic Agent Creation

Create agents at runtime:

```python
from subagents_pydantic_ai import (
    create_subagent_toolset,
    create_agent_factory_toolset,
    DynamicAgentRegistry,
)

registry = DynamicAgentRegistry()

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[
        create_subagent_toolset(),
        create_agent_factory_toolset(
            registry=registry,
            allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
            max_agents=5,
        ),
    ],
)

# The agent can now create new specialized agents at runtime
# using the create_agent tool
```

## Configuration Options

### SubAgentConfig

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier |
| `description` | `str` | Yes | Shown to parent agent |
| `instructions` | `str` | Yes | System prompt |
| `model` | `str` | No | LLM model (default: parent's) |
| `can_ask_questions` | `bool` | No | Can ask parent (default: True) |
| `max_questions` | `int` | No | Max questions per task |

### create_subagent_toolset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subagents` | `list[SubAgentConfig]` | `None` | Subagent configurations |
| `default_model` | `str` | `"openai:gpt-4.1"` | Default model |
| `toolsets_factory` | `Callable` | `None` | Creates tools for subagents |
| `include_general_purpose` | `bool` | `True` | Add fallback agent |
| `max_nesting_depth` | `int` | `0` | Allow nested subagents |
| `id` | `str` | `"subagents"` | Toolset ID |

## Next Steps

- [Dual-Mode Execution](dual-mode.md) - Learn sync vs async
- [API Reference](api-reference.md) - Complete API docs
