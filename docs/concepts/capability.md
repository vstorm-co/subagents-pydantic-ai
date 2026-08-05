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
        default_model="openai:gpt-4.1",
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
    default_model="openai:gpt-4.1",    # Fallback model; no implicit default
    include_general_purpose=True,       # GP subagent (default True; needs default_model or a factory)
    max_nesting_depth=0,                # Allow nested subagents (0 = no nesting)
    toolsets_factory=my_factory,        # Custom toolsets for subagents
    registry=my_registry,              # Dynamic agent registry
    descriptions={                      # Override tool descriptions
        "task": "Delegate work to a specialist",
    },
    usage_limits=UsageLimits(           # Static limits, or a factory (see below)
        request_limit=10,
    ),
    delegation_configuration="default", # Task-only, persisted, combined, or oneshot
    allowed_models=None,                # Model allow-list for dynamic specialists
    capabilities_map=None,              # Capability factories for dynamic specialists
    default_agent_factory=None,         # Custom agent factory for dynamic specialists
    max_agents=10,                      # Persistent-agent limit
)
```

### Delegation configuration

Choose which delegation entry points the capability exposes:

| Mode | Entry-point tools |
|------|-------------------|
| `"default"` | `task` |
| `"persisted"` | `create_agent`, `task` |
| `"persisted_and_oneshot"` | `create_agent`, `task`, `delegate` |
| `"oneshot_only"` | `delegate` |

The one-shot `delegate` tool creates an ephemeral specialist and runs its task
in one call. The specialist is not registered in the dynamic agent registry.

```python
SubAgentCapability(
    default_model="openai:gpt-4.1",
    delegation_configuration="persisted_and_oneshot",
    allowed_models=["openai:gpt-4.1"],
    capabilities_map={"filesystem": lambda deps: [create_fs_toolset(deps.backend)]},
)
```

Use `delegate` for one-off specialists. Use `task` with pre-configured or registry-backed agents for reusable delegation.

### Usage limits

`usage_limits` caps token/request usage for delegated subagent runs. Pass a
`UsageLimits` (from pydantic-ai) instance to reuse the same limits
for every task, or a
[`UsageLimitsFactory`][subagents_pydantic_ai.types.UsageLimitsFactory] —
`(ctx, config) -> UsageLimits | None` — called once per delegated task with the
parent run context and the selected subagent config. A factory may return `None`
to run that task without explicit limits. Limits are enforced on every retry
attempt as well.

```python
from pydantic_ai import RunContext, UsageLimits
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

def limits_for(ctx: RunContext, config: SubAgentConfig) -> UsageLimits | None:
    # Give the researcher a larger budget than other subagents.
    if config["name"] == "researcher":
        return UsageLimits(request_limit=20)
    return UsageLimits(request_limit=5)

cap = SubAgentCapability(default_model="openai:gpt-4.1", subagents=[...], usage_limits=limits_for)
```

## How It Works

When you pass `SubAgentCapability` to an agent, pydantic-ai calls:

1. **`get_toolset()`** — returns the configured delegation entry points plus
   task lifecycle tools (`check_task`, `answer_subagent`, `list_active_tasks`,
   `wait_tasks`, `soft_cancel_task`, `hard_cancel_task`)

2. **`get_instructions()`** — returns a callable that generates the system prompt
   listing available subagents with their descriptions (via
   [`get_subagent_system_prompt`][subagents_pydantic_ai.prompts.get_subagent_system_prompt])

!!! note "Sync-mode questions need `ask_user`"
    In **sync** mode the parent's run loop is blocked inside the delegation, so a
    subagent's `ask_parent` cannot be answered with `answer_subagent`. Pass
    `ask_user=...` to the capability to give it a channel; without one, a subagent
    that asks gets a configuration error back. In **async** mode the parent answers
    via `answer_subagent`. See [Parent-Child Questions](../advanced/questions.md).

## Observability

Access the task manager for monitoring background tasks:

```python
cap = SubAgentCapability(default_model="openai:gpt-4.1", subagents=[...])
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
        SubAgentCapability(default_model="openai:gpt-4.1", subagents=[...]),
    ],
)
```

## Custom Agents via SubAgentConfig

`SubAgentCapability` supports using custom agent instances through the `agent` and
`agent_factory` fields on `SubAgentConfig`. When the capability compiles subagents
internally, it follows the same resolution priority as `_compile_subagent`:
`agent` > `agent_factory` > default `Agent()`.

```python
from pydantic_ai import Agent
from pydantic_deep import create_deep_agent
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[SubAgentCapability(
        default_model="openai:gpt-4.1",
        subagents=[
            # Pre-built agent
            SubAgentConfig(
                name="researcher",
                description="Researches topics",
                instructions="You are a research assistant.",
                agent=create_deep_agent(model="openai:gpt-4.1"),
            ),
            # Agent factory
            SubAgentConfig(
                name="coder",
                description="Writes code",
                instructions="You write Python code.",
                agent_factory=lambda cfg: create_deep_agent(
                    model=cfg.get("model", "openai:gpt-4.1"),
                    system_prompt=cfg["instructions"],
                ),
            ),
            # Default: plain Agent is created automatically
            SubAgentConfig(
                name="writer",
                description="Writes content",
                instructions="You write clear documentation.",
            ),
        ],
    )],
)
```

See [SubAgentConfig](types.md#subagentconfig) for full details on these fields.

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
