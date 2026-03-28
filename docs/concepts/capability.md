# SubAgentCapability

`SubAgentCapability` is the recommended way to add subagent delegation to a Pydantic AI agent.
It's a [pydantic-ai capability](https://ai.pydantic.dev/capabilities/) that bundles
delegation tools and instructions into a single plug-and-play unit.

## Why Capability over Toolset?

| Feature | SubAgentCapability | create_subagent_toolset |
|---------|:-:|:-:|
| Tools registered automatically | Yes | Yes |
| Dynamic system prompt (lists subagents) | Yes | Manual wiring |
| AgentSpec YAML support | Yes | No |
| Single import | Yes | Need toolset + prompt function |
| `task_manager` access | Property | `getattr(toolset, "task_manager")` |

## Basic Usage

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        subagents=[
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
            ),
        ],
    )],
)
```

## Configuration

```python
SubAgentCapability(
    subagents=[...],                    # Subagent configurations
    default_model="openai:gpt-4.1",    # Default model for subagents
    include_general_purpose=True,       # Include GP subagent (default: True)
    max_nesting_depth=0,                # Allow nested subagents (0 = no nesting)
    toolsets_factory=my_factory,        # Custom toolsets for subagents
    registry=my_registry,              # Dynamic agent registry
    descriptions={                      # Override tool descriptions
        "task": "Delegate work to a specialist",
    },
)
```

## How It Works

When you pass `SubAgentCapability` to an agent, pydantic-ai calls:

1. **`get_toolset()`** — returns the `FunctionToolset` containing delegation tools
   (`task`, `check_task`, `answer_subagent`, `list_active_tasks`,
   `soft_cancel_task`, `hard_cancel_task`)

2. **`get_instructions()`** — returns a callable that generates the system prompt
   listing available subagents with their descriptions

## Observability

Access the task manager for monitoring background tasks:

```python
cap = SubAgentCapability(subagents=[...])
agent = Agent("openai:gpt-4.1", capabilities=[cap])

# After agent runs, check active tasks
task_mgr = cap.task_manager
if task_mgr:
    active = task_mgr.list_active_tasks()
```

## Composing with Other Capabilities

```python
from pydantic_ai import Agent
from pydantic_ai_todo import TodoCapability
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        TodoCapability(enable_subtasks=True),
        SubAgentCapability(subagents=[...]),
    ],
)
```

## AgentSpec (YAML)

```yaml
model: openai:gpt-4.1
capabilities:
  - SubAgentCapability:
      subagents:
        - name: researcher
          description: Researches topics
          instructions: You are a research assistant.
      include_general_purpose: true
```
