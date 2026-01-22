<h1 align="center">Subagents for Pydantic AI</h1>
<p align="center">
  <em>Multi-Agent Orchestration for Pydantic AI</em>
</p>
<p align="center">
  <a href="https://github.com/vstorm-co/subagents-pydantic-ai/actions/workflows/ci.yml"><img src="https://github.com/vstorm-co/subagents-pydantic-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://coveralls.io/github/vstorm-co/subagents-pydantic-ai?branch=main"><img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage"></a>
  <a href="https://pypi.org/project/subagents-pydantic-ai/"><img src="https://img.shields.io/pypi/v/subagents-pydantic-ai.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
</p>

---

**Subagents for Pydantic AI** adds multi-agent delegation to any [Pydantic AI](https://ai.pydantic.dev/) agent. Spawn specialized subagents that run **synchronously** (blocking), **asynchronously** (background), or let the system **auto-select** the best mode.

Think of it as the building blocks for multi-agent systems - where a parent agent can delegate specialized tasks to child agents, and those children can have their own children.

## Why use Subagents?

1. **Specialization**: Each subagent has focused instructions and tools for its domain. A "researcher" agent researches, a "writer" agent writes.

2. **Parallel Execution**: Run multiple tasks simultaneously in async mode. Start a research task, continue with other work, check results later.

3. **Nested Hierarchies**: Subagents can spawn their own subagents. Build complex multi-agent workflows with natural delegation patterns.

4. **Smart Mode Selection**: Auto-mode intelligently chooses sync vs async based on task complexity and requirements.

## Hello World Example

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

# Define specialized subagents
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics and gathers information",
        instructions="You are a research assistant. Investigate thoroughly.",
    ),
]

# Create toolset and agent
toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate tasks to specialized subagents.",
)

# Run the agent
result = agent.run_sync("Research Python async patterns", deps=Deps())
print(result.output)
```

## Core Features

| Feature | Description |
|---------|-------------|
| **Dual-Mode Execution** | Run tasks sync (blocking) or async (background) |
| **Auto-Mode Selection** | Intelligent mode selection based on task characteristics |
| **Nested Subagents** | Subagents can spawn their own subagents |
| **Runtime Agent Creation** | Create specialized agents on-the-fly |
| **Parent-Child Q&A** | Subagents can ask parent for clarification |
| **Task Cancellation** | Soft and hard cancellation support |
| **Pluggable Message Bus** | Extensible communication layer |

## Available Tools

When you add the subagent toolset, your agent gets these tools:

| Tool | Description |
|------|-------------|
| `task` | Delegate a task to a subagent (sync, async, or auto) |
| `check_task` | Check status and get result of a background task |
| `answer_subagent` | Answer a question from a blocked subagent |
| `list_active_tasks` | List all running background tasks |
| `soft_cancel_task` | Request cooperative cancellation |
| `hard_cancel_task` | Immediately cancel a task |

## Part of the Pydantic AI Ecosystem

Subagents for Pydantic AI is part of a modular ecosystem:

| Package | Description |
|---------|-------------|
| [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) | Full agent framework (uses this library) |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage and Docker sandbox backends |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning toolset |
| [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) | Context management processors |

## Installation

```bash
pip install subagents-pydantic-ai
```

## Next Steps

- [Installation](installation.md) - Get started in minutes
- [Core Concepts](concepts/index.md) - Learn about subagents, toolsets, and types
- [Examples](examples/index.md) - See subagents in action
- [API Reference](api/index.md) - Complete API documentation

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>
