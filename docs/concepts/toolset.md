# Toolset

The `create_subagent_toolset()` function creates a toolset that adds delegation capabilities to your Pydantic AI agent.

## Creating a Toolset

```python
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig

subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics",
        instructions="You are a research assistant.",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
```

## Factory Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subagents` | `list[SubAgentConfig]` | `[]` | List of subagent configurations |
| `default_model` | `str \| None` | `None` | Default model for subagents |
| `toolsets_factory` | `Callable` | `None` | Factory to create toolsets for subagents |
| `max_nesting_depth` | `int` | `2` | Maximum subagent nesting depth |
| `general_purpose_config` | `SubAgentConfig \| None` | Auto | Config for the "general" subagent |
| `descriptions` | `dict[str, str] \| None` | `None` | Override default tool descriptions by tool name |

## Adding to an Agent

```python
from pydantic_ai import Agent

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate tasks to specialized subagents.",
)
```

## Available Tools

The toolset provides these tools to your agent:

### task

Delegate a task to a subagent.

```python
# The agent calls:
task(
    description="Research Python async patterns",
    subagent_type="researcher",
    mode="sync",  # or "async" or "auto"
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | `str` | What the subagent should do |
| `subagent_type` | `str` | Name of the subagent to use |
| `mode` | `str` | `"sync"`, `"async"`, or `"auto"` |

### check_task

Check the status of a background task.

```python
# The agent calls:
check_task(task_id="abc123")
```

**Returns:** Status, result (if complete), or pending question (if waiting).

### answer_subagent

Answer a question from a blocked subagent.

```python
# The agent calls:
answer_subagent(task_id="abc123", answer="Use PostgreSQL for this project")
```

### list_active_tasks

List all running background tasks.

```python
# The agent calls:
list_active_tasks()
```

**Returns:** List of task IDs, subagent names, and statuses.

### soft_cancel_task

Request cooperative cancellation.

```python
# The agent calls:
soft_cancel_task(task_id="abc123")
```

The subagent will receive a cancellation request and should stop gracefully.

### hard_cancel_task

Immediately cancel a task.

```python
# The agent calls:
hard_cancel_task(task_id="abc123")
```

Forces immediate termination.

## Toolsets Factory

Provide tools to your subagents:

```python
from pydantic_ai_backends import create_console_toolset

def my_toolsets_factory(deps):
    """Create toolsets for each subagent."""
    return [
        create_console_toolset(),  # File operations
    ]

toolset = create_subagent_toolset(
    subagents=subagents,
    toolsets_factory=my_toolsets_factory,
)
```

The factory is called for each subagent with cloned dependencies.

## Nesting Depth

Control how deep subagents can nest:

```python
# Allow subagents to have their own subagents (2 levels deep)
toolset = create_subagent_toolset(
    subagents=subagents,
    max_nesting_depth=2,
)

# No nesting - subagents can't delegate
toolset = create_subagent_toolset(
    subagents=subagents,
    max_nesting_depth=0,
)
```

## General Purpose Subagent

By default, a "general" subagent is added for tasks that don't match specific subagents:

```python
# Customize the general subagent
toolset = create_subagent_toolset(
    subagents=subagents,
    general_purpose_config=SubAgentConfig(
        name="general",
        description="Handles miscellaneous tasks",
        instructions="You are a general-purpose assistant.",
    ),
)

# Disable the general subagent
toolset = create_subagent_toolset(
    subagents=subagents,
    general_purpose_config=None,
)
```

## Custom Tool Descriptions

Override the default tool descriptions to better guide LLM behavior. This is useful when you want descriptions that are more specific to your use case:

```python
toolset = create_subagent_toolset(
    subagents=subagents,
    descriptions={
        "task": "Assign a task to a specialized subagent",
        "check_task": "Check the status of a delegated task",
        "list_active_tasks": "Show all currently running background tasks",
    },
)
```

Only the tool names you include in the dictionary are overridden; the rest keep their built-in defaults. Available tool names:

| Tool Name | Description |
|-----------|-------------|
| `task` | Delegate a task to a subagent |
| `check_task` | Check status of a background task |
| `answer_subagent` | Answer a question from a blocked subagent |
| `list_active_tasks` | List all running background tasks |
| `wait_tasks` | Wait for background tasks to complete |
| `soft_cancel_task` | Request cooperative cancellation |
| `hard_cancel_task` | Immediately cancel a task |

## System Prompt

Add context about available subagents to your agent's system prompt:

```python
from subagents_pydantic_ai import get_subagent_system_prompt

# Generate prompt listing available subagents
prompt = get_subagent_system_prompt(deps, compiled_subagents)
```

This generates text like:

```
## Available Subagents

You can delegate tasks to these specialized subagents:

- **researcher**: Researches topics and gathers information
- **writer**: Writes content based on research
- **coder**: Writes and tests Python code

Use the `task` tool to delegate work.
```

## Next Steps

- [Types](types.md) - Data structures and enums
- [Execution Modes](../advanced/execution-modes.md) - Sync vs async
- [Examples](../examples/index.md) - Working examples
