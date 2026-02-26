<h1 align="center">Subagents for Pydantic AI</h1>

<p align="center">
  <em>Multi-Agent Orchestration for Pydantic AI</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/subagents-pydantic-ai/"><img src="https://img.shields.io/pypi/v/subagents-pydantic-ai.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://coveralls.io/github/vstorm-co/subagents-pydantic-ai?branch=main"><img src="https://coveralls.io/repos/github/vstorm-co/subagents-pydantic-ai/badge.svg?branch=main" alt="Coverage Status"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

<p align="center">
  <b>Nested Subagents</b> — subagents spawn their own subagents
  &nbsp;&bull;&nbsp;
  <b>Runtime Agent Creation</b> — create specialists on-the-fly
  &nbsp;&bull;&nbsp;
  <b>Auto-Mode Selection</b> — intelligent sync/async decision
</p>

---

**Subagents for Pydantic AI** adds multi-agent delegation to any [Pydantic AI](https://ai.pydantic.dev/) agent. Spawn specialized subagents that run **synchronously** (blocking), **asynchronously** (background), or let the system **auto-select** the best mode.

> **Full framework?** Check out [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) - complete agent framework with planning, filesystem, subagents, and skills.

## Use Cases

| What You Want to Build | How Subagents Help |
|------------------------|-------------------|
| **Research Assistant** | Delegate research to specialists, synthesize with a writer agent |
| **Code Review System** | Security agent, style agent, and performance agent work in parallel |
| **Content Pipeline** | Researcher → Analyst → Writer chain with handoffs |
| **Data Processing** | Spawn workers dynamically based on data volume |
| **Customer Support** | Route to specialized agents (billing, technical, sales) |
| **Document Analysis** | Extract, summarize, and categorize with focused agents |

## Installation

```bash
pip install subagents-pydantic-ai
```

Or with uv:

```bash
uv add subagents-pydantic-ai
```

## Quick Start

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

# Dependencies must implement SubAgentDepsProtocol
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
    SubAgentConfig(
        name="writer",
        description="Writes content based on research",
        instructions="You are a technical writer. Write clear, concise content.",
    ),
]

# Create toolset and agent (with optional custom tool descriptions)
toolset = create_subagent_toolset(
    subagents=subagents,
    descriptions={
        "task": "Assign a task to a specialized subagent",
        "check_task": "Check the status of a delegated task",
    },
)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate tasks to specialized subagents.",
)

# Run the agent
result = agent.run_sync(
    "Research Python async patterns and write a blog post about it",
    deps=Deps(),
)
print(result.output)
```

## Execution Modes

Choose how subagents execute their tasks:

| Mode | Description | Use Case |
|------|-------------|----------|
| `sync` | Block until complete | Quick tasks, when result is needed immediately |
| `async` | Run in background | Long research, parallel tasks |
| `auto` | Smart selection | Let the system decide based on task characteristics |

### Sync Mode (Default)

```python
# Agent calls: task(description="...", subagent_type="researcher", mode="sync")
# Parent waits for result before continuing
```

### Async Mode

```python
# Agent calls: task(description="...", subagent_type="researcher", mode="async")
# Returns task_id immediately, agent continues working
# Later: check_task(task_id) to get result
```

### Auto Mode

```python
# Agent calls: task(description="...", subagent_type="researcher", mode="auto")
# System decides based on:
# - Task complexity (simple → sync, complex → async)
# - Independence (can run without user context → async)
# - Subagent preferences (from config)
```

## Give Subagents Tools

Provide toolsets so subagents can interact with files, APIs, or other services:

```python
from pydantic_ai_backends import create_console_toolset

def my_toolsets_factory(deps):
    """Factory that creates toolsets for subagents."""
    return [
        create_console_toolset(),  # File operations
        create_search_toolset(),   # Web search
    ]

toolset = create_subagent_toolset(
    subagents=subagents,
    toolsets_factory=my_toolsets_factory,
)
```

## Dynamic Agent Creation

Create agents on-the-fly and delegate to them seamlessly:

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
        # Pass registry so task() can resolve dynamically created agents
        create_subagent_toolset(registry=registry),
        create_agent_factory_toolset(
            registry=registry,
            allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
            max_agents=5,
        ),
    ],
)

# Now the agent can:
# 1. create_agent(name="analyst", ...) — creates a new agent in registry
# 2. task(description="...", subagent_type="analyst") — delegates to it
```

## Subagent Questions

Enable subagents to ask the parent for clarification:

```python
SubAgentConfig(
    name="analyst",
    description="Analyzes data",
    instructions="Ask for clarification when data is ambiguous.",
    can_ask_questions=True,
    max_questions=3,
)
```

The parent agent can then respond using `answer_subagent(task_id, answer)`.

## Available Tools

| Tool | Description |
|------|-------------|
| `task` | Delegate a task to a subagent (sync, async, or auto) |
| `check_task` | Check status and get result of a background task |
| `answer_subagent` | Answer a question from a blocked subagent |
| `list_active_tasks` | List all running background tasks |
| `soft_cancel_task` | Request cooperative cancellation |
| `hard_cancel_task` | Immediately cancel a task |

## Per-Subagent Configuration

```python
SubAgentConfig(
    name="coder",
    description="Writes and reviews code",
    instructions="Follow project coding rules.",
    context_files=["/CODING_RULES.md"],  # Loaded by consumer library
    extra={"memory": "project", "cost_budget": 100},  # Custom metadata
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Parent Agent                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Subagent Toolset                   │    │
│  │  task() │ check_task() │ answer_subagent()      │    │
│  └─────────────────────────────────────────────────┘    │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼               ▼               ▼               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │ researcher │  │   writer   │  │   coder    │         │
│  │  (sync)    │  │  (async)   │  │  (auto)    │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│                                                         │
│              Message Bus (pluggable)                    │
└─────────────────────────────────────────────────────────┘
```

## Related Projects

| Package | Description |
|---------|-------------|
| [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) | Full agent framework (uses this library) |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage and Docker sandbox backends |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning toolset |
| [summarization-pydantic-ai](https://github.com/vstorm-co/summarization-pydantic-ai) | Context management processors |
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation - agent framework by Pydantic |

## Contributing

```bash
git clone https://github.com/vstorm-co/subagents-pydantic-ai.git
cd subagents-pydantic-ai
make install
make test  # 100% coverage required
make all   # lint + typecheck + test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>
