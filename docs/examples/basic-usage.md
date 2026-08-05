# Basic Usage

## Using SubAgentCapability (Recommended)

The simplest way to add subagent delegation:

```python
import asyncio
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        default_model="openai:gpt-4.1",
        subagents=[
            SubAgentConfig(
                name="researcher",
                description="Researches topics and gathers information",
                instructions="You are a research assistant. Investigate thoroughly.",
            ),
            SubAgentConfig(
                name="summarizer",
                description="Summarizes long content into concise points",
                instructions="You are a summarization expert. Be concise.",
            ),
        ],
    )],
)

async def main():
    result = await agent.run("Research the benefits of async programming in Python")
    print(result.output)

asyncio.run(main())
```

`SubAgentCapability` handles everything:

- Registers `task`, `check_task`, `answer_subagent`, and management tools
- Injects system prompt listing available subagents
- Includes a general-purpose subagent by default

## With Nesting

Allow subagents to spawn their own subagents:

```python
agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        default_model="openai:gpt-4.1",
        subagents=[...],
        max_nesting_depth=2,  # Subagents can nest 2 levels deep
    )],
)
```

## YAML Agent Definition

```yaml
# agent.yaml
model: openai:gpt-4.1
capabilities:
  - SubAgentCapability:
      subagents:
        - name: researcher
          description: Researches topics
          instructions: You are a research assistant.
        - name: writer
          description: Writes content
          instructions: You are a technical writer.
```

YAML/JSON loading is handled by pydantic-ai's `Agent.from_file` (which builds an
`AgentSpec` from the file). `SubAgentCapability` opts into this via its
[`get_serialization_name()`][subagents_pydantic_ai.capability.SubAgentCapability]
classmethod, which returns `"SubAgentCapability"` — the key used in the YAML
above.

Because `SubAgentCapability` is a third-party capability, pydantic-ai does not
register it automatically: you must declare it with `custom_capability_types` so
the deserializer can resolve the `SubAgentCapability:` block.

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability

agent = Agent.from_file(
    "agent.yaml",
    custom_capability_types=[SubAgentCapability],
)
```

Each entry under `subagents:` is loaded as a plain
[`SubAgentConfig`][subagents_pydantic_ai.types.SubAgentConfig] dict.

!!! note "`SubAgentSpec` vs `Agent.from_file`"
    [`SubAgentSpec`][subagents_pydantic_ai.spec.SubAgentSpec] is a separate,
    standalone Pydantic model for when you load and parse YAML/JSON **yourself**
    (e.g. `specs = [SubAgentSpec(**s) for s in data["subagents"]]`, then
    `spec.to_config()`). It is *not* part of the `Agent.from_file` path, which
    deserializes the capability and its `subagents` dicts directly. Use
    `SubAgentSpec` when you want validation and defaults for hand-loaded subagent
    definitions.

## Using the Toolset API (Alternative)

For lower-level control:

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, get_subagent_system_prompt, SubAgentConfig

configs = [
    SubAgentConfig(name="researcher", description="Researches topics", instructions="..."),
]

toolset = create_subagent_toolset(default_model="openai:gpt-4.1", subagents=configs)
agent = Agent(
    "openai:gpt-4.1",
    toolsets=[toolset],
    system_prompt=get_subagent_system_prompt(configs),
)
```

!!! note
    With `SubAgentCapability`, the system prompt is injected dynamically.
    With the toolset API, you need to wire `get_subagent_system_prompt()` manually.

## What Happens at Runtime

1. User asks: "Research the benefits of async programming"
2. Parent agent decides to delegate to "researcher"
3. Parent calls: `task(description="...", subagent_type="researcher")`
4. Researcher subagent executes the task
5. Result is returned to parent
6. Parent formats and returns final response

## Next Steps

- [Sync vs Async](sync-async.md) — Learn about execution modes
- [Giving Subagents Tools](toolsets.md) — Provide capabilities to subagents
- [Research Team](research-team.md) — Build a multi-agent research pipeline
