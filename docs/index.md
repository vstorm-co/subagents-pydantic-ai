<p align="center">
  <img src="assets/social-preview.png" alt="Subagents for Pydantic AI" width="100%">
</p>

<h1 align="center">Subagents for Pydantic AI</h1>

<p align="center"><em>Declarative multi-agent orchestration — sync, async, or auto.</em></p>

<p align="center">
  <a href="https://pypi.org/project/subagents-pydantic-ai/"><img src="https://img.shields.io/pypi/v/subagents-pydantic-ai.svg" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/subagents-pydantic-ai"><img src="https://static.pepy.tech/badge/subagents-pydantic-ai/month" alt="PyPI Downloads"></a>
  <a href="https://github.com/vstorm-co/subagents-pydantic-ai/stargazers"><img src="https://img.shields.io/github/stars/vstorm-co/subagents-pydantic-ai?style=flat&logo=github&color=yellow" alt="GitHub Stars"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://coveralls.io/github/vstorm-co/subagents-pydantic-ai?branch=main"><img src="https://coveralls.io/repos/github/vstorm-co/subagents-pydantic-ai/badge.svg?branch=main" alt="Coverage Status"></a>
  <a href="https://github.com/vstorm-co/subagents-pydantic-ai/actions/workflows/ci.yml"><img src="https://github.com/vstorm-co/subagents-pydantic-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

---

!!! tip "Part of Pydantic Deep Agents"
    **Subagents for Pydantic AI** is one library in [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) — the open-source
    Claude Code alternative & Python agent framework. Use it standalone, or get every
    library wired together in a single `create_deep_agent()` call.

**Subagents for Pydantic AI** adds multi-agent delegation to any [Pydantic AI](https://ai.pydantic.dev/) agent. Spawn specialized subagents that run **synchronously** (blocking), **asynchronously** (background), or let the system **auto-select** the best mode.

Think of it as the building blocks for multi-agent systems - where a parent agent can delegate specialized tasks to child agents, and those children can have their own children.

## Why use Subagents?

1. **Specialization**: each subagent has focused instructions and tools for its domain. A "researcher" agent researches, a "writer" agent writes, and neither carries the other's context.

2. **Parallel execution**: run several tasks in the background, keep working, and collect results with `wait_tasks` — reacting to the first finisher instead of stalling on the slowest.

3. **Course correction**: [steer a running subagent](advanced/steering.md) when you learn something new, keeping its partial progress, and [answer its questions](advanced/questions.md) when it needs clarification.

4. **Accountability**: every delegation records its [cost, tokens, tool calls, and traceparent](concepts/observability.md), so a fan-out is not a black box.

## Quick Start (Capability API)

The recommended way to add subagent delegation — one import, plug and play:

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        subagents=[
            SubAgentConfig(
                name="researcher",
                description="Researches topics and gathers information",
                instructions="You are a research assistant. Investigate thoroughly.",
            ),
        ],
    )],
)

result = await agent.run("Research Python async patterns")
```

`SubAgentCapability` automatically registers all tools and injects a dynamic system
prompt listing available subagents. A general-purpose subagent is included by default.

### Alternative: Toolset API

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

toolset = create_subagent_toolset(
    subagents=[SubAgentConfig(name="researcher", description="...", instructions="...")],
)
agent = Agent("openai:gpt-4.1", toolsets=[toolset])
```

!!! note
    With the toolset API you need to wire `get_subagent_system_prompt()` into the
    agent's instructions manually. `SubAgentCapability` handles this automatically.

## Core features

| Feature | Description |
|---------|-------------|
| [**Dual-mode execution**](advanced/execution-modes.md) | Run a task blocking or in the background, or let `mode="auto"` decide |
| [**Chat traces**](advanced/chat-traces.md) | Continue a subagent's conversation across delegations |
| [**Questions**](advanced/questions.md) | A subagent asks its parent for clarification mid-task |
| [**Steering**](advanced/steering.md) | Redirect a running task without losing its progress |
| [**Cancellation**](advanced/cancellation.md) | Cooperative (clean boundary) or immediate |
| [**Retries**](advanced/retries.md) | Transient gateway failures resume from accumulated history |
| [**Usage limits**](advanced/usage-limits.md) | One ceiling for every delegation, or one computed per task |
| [**Observability**](concepts/observability.md) | Per-task cost, tokens, tool-call counts, traceparent |
| [**Dynamic agents**](advanced/dynamic-agents.md) | Create specialists at runtime, reusable or one-shot |

## Available tools

Your agent gets these tools. `create_agent` and `delegate` are opt-in — see
[delegation configuration](concepts/toolset.md).

| Tool | Description |
|------|-------------|
| `task` | Delegate a task to a configured or registry-backed subagent |
| `create_agent` | Create a reusable specialist at runtime (opt-in) |
| `delegate` | Create an ephemeral specialist and run a task in one call (opt-in) |
| `check_task` | Check status and get the full result of a background task |
| `wait_tasks` | Wait for background tasks, in `all` or `any` mode |
| `list_active_tasks` | List the running background tasks of this run |
| `answer_subagent` | Answer a question from a blocked subagent |
| `send_message_to_subagent` | Steer a running background subagent |
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

## Next steps

- [Installation](installation.md) — get started in minutes
- [Core concepts](concepts/index.md) — subagents, toolsets, observability, types
- [Execution modes](advanced/execution-modes.md) — when to block and when not to
- [Failure handling](advanced/errors.md) — what propagates and what is contained
- [Examples](examples/index.md) — see subagents in action
- [API reference](api/index.md) — the generated reference

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>
