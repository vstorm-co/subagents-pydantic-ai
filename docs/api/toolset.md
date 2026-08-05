# Toolset API

## create_subagent_toolset

::: subagents_pydantic_ai.create_subagent_toolset
    options:
      show_root_heading: true
      show_source: true

## create_agent_factory_toolset

::: subagents_pydantic_ai.create_agent_factory_toolset
    options:
      show_root_heading: true
      show_source: true

## SubAgentToolset

::: subagents_pydantic_ai.SubAgentToolset
    options:
      show_root_heading: true
      show_source: true
      members:
        - task
        - check_task
        - answer_subagent
        - list_active_tasks
        - wait_tasks
        - soft_cancel_task
        - hard_cancel_task

!!! info "Prompt builders"
    `get_subagent_system_prompt` and `get_task_instructions_prompt` are documented
    on the [Prompts &amp; Retry](prompts.md) page.

---

## Usage Example

```python
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

# Define subagents
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics",
        instructions="You research topics thoroughly.",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes content",
        instructions="You write clear content.",
    ),
]

# Create toolset
toolset = create_subagent_toolset(
    subagents=subagents,
    default_model="openai:gpt-4o",
    max_nesting_depth=1,
)

# Add to agent
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
)
```

## With Custom Tool Descriptions

Override default tool descriptions for better LLM behavior:

```python
toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    descriptions={
        "task": "Assign a task to a specialized subagent",
        "check_task": "Check the status of a delegated task",
        "list_active_tasks": "Show all currently running background tasks",
    },
)
```

Available tool names: `task`, `check_task`, `answer_subagent`, `list_active_tasks`, `wait_tasks`, `soft_cancel_task`, `hard_cancel_task`.

## With Toolsets Factory

```python
from pydantic_ai_backends import create_console_toolset

def my_toolsets_factory(deps):
    return [create_console_toolset()]

toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=subagents,
    toolsets_factory=my_toolsets_factory,
)
```

## With Dynamic Agent Creation

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
        create_subagent_toolset(default_model="openai:gpt-4.1", subagents=subagents),
        create_agent_factory_toolset(
            default_model="openai:gpt-4.1",
            registry=registry,
            allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
            max_agents=5,
        ),
    ],
)
```

## Programmatic access

`answer_task` and `steer_task` are the Python halves of the `answer_subagent` and
`send_message_to_subagent` tools, for an application that drives delegation itself.
See [Steering](../advanced/steering.md#driving-it-from-python).

::: subagents_pydantic_ai.SubAgentToolset
    options:
      show_root_heading: false
      show_source: true
      heading_level: 3
      members:
        - answer_task
        - steer_task
        - cancel_run_tasks
        - get_total_usage
