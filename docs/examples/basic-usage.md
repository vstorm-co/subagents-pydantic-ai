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

```python
from pydantic_ai import Agent

agent = Agent.from_file("agent.yaml")
```

## Using the Toolset API (Alternative)

For lower-level control:

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, get_subagent_system_prompt, SubAgentConfig

configs = [
    SubAgentConfig(name="researcher", description="Researches topics", instructions="..."),
]

toolset = create_subagent_toolset(subagents=configs)
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
