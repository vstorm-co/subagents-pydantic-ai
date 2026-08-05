# Dynamic Agent Creation

Create reusable or ephemeral specialized agents at runtime.

## Overview

While pre-configured subagents cover most use cases, sometimes you need to create agents dynamically:

- User requests a specialist for an unexpected domain
- Task requires a unique combination of capabilities
- Experimentation with different agent configurations

## Delegation Toolset

`create_subagent_toolset` exposes `task` by default. Opt into persistent
creation with `delegation_configuration="persisted"`:

```python
from subagents_pydantic_ai import (
    create_subagent_toolset,
    DynamicAgentRegistry,
)

registry = DynamicAgentRegistry()

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[create_subagent_toolset(
        default_model="openai:gpt-4o",
        subagents=base_subagents,
        registry=registry,
        delegation_configuration="persisted",
        allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
    )],
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry` | `DynamicAgentRegistry \| None` | `None` | Registry for created agents; an internal registry is created when omitted |
| `allowed_models` | `list[str] \| None` | `None` (all allowed) | Models agents can use |
| `default_model` | `str \| Model \| None` | `None` | Model for a `create_agent` call that names none. No implicit default: when unset, such a call is refused rather than run on a library-chosen model |
| `max_agents` | `int` | `10` | Maximum dynamic agents |
| `toolsets_factory` | `ToolsetFactory \| None` | `None` | Factory that builds toolsets for every new agent. Takes priority over `capabilities_map` when both are set |
| `capabilities_map` | `dict[str, CapabilityFactory] \| None` | `None` | Maps capability names to factory functions, e.g. `{"filesystem": ..., "todo": ...}`. Used when `capabilities` are passed to `create_agent` |
| `delegation_configuration` | `DelegationConfiguration` | `"default"` | Select default, persisted, combined, or one-shot-only entry points |
| `default_agent_factory` | `Callable \| None` | `None` | Factory used to build every dynamic agent |

See [`create_subagent_toolset`][subagents_pydantic_ai.toolset.create_subagent_toolset]
for the full signature.

## DynamicAgentRegistry

The [`DynamicAgentRegistry`][subagents_pydantic_ai.registry.DynamicAgentRegistry] tracks dynamically created agents. It stores the `Agent` instance, the `SubAgentConfig`, and a [`CompiledSubAgent`][subagents_pydantic_ai.types.CompiledSubAgent] for each registered agent.

```python
from subagents_pydantic_ai import DynamicAgentRegistry

registry = DynamicAgentRegistry()

# Check registered agents
agents = registry.list_agents()

# Get a specific agent
agent = registry.get("custom-analyst")

# Remove an agent
registry.remove("custom-analyst")
```

### Registry Lifecycle

The lifecycle of a dynamically created agent follows four stages:

```
Creation → Registration → Usage → Removal
```

#### 1. Creation

An agent is created when the parent calls `create_agent()` through the factory toolset. The factory validates the name, model, and capabilities, then builds a `pydantic-ai` `Agent` instance.

#### 2. Registration

The agent, its `SubAgentConfig`, and a `CompiledSubAgent` wrapper are stored together in the registry via `registry.register(config, agent)`. At this point the agent becomes discoverable by the `task()` tool.

```python
# Internal flow (handled by create_agent_factory_toolset):
config = SubAgentConfig(name="rust-expert", description="...", instructions="...")
agent = Agent("openai:gpt-4.1", system_prompt=config["instructions"])
registry.register(config, agent)
```

#### 3. Usage

Once registered, the `task()` tool can delegate work to the dynamic agent by name. The toolset looks up the agent in the compiled dict first, then falls back to the registry:

```python
# Parent agent calls:
task(description="Review this Rust code", subagent_type="rust-expert", mode="sync")
```

#### 4. Removal

When a dynamic agent is no longer needed, the parent calls `remove_agent()` through the factory toolset. This removes all three entries (agent, config, compiled) from the registry:

```python
# Parent agent calls:
remove_agent(name="rust-expert")
```

### Registry Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `register(config, agent)` | `None` | Register a new agent (raises `ValueError` if name exists or limit reached) |
| `get(name)` | `Agent \| None` | Get the `Agent` instance by name |
| `get_config(name)` | `SubAgentConfig \| None` | Get the configuration by name |
| `get_compiled(name)` | `CompiledSubAgent \| None` | Get the compiled wrapper by name |
| `remove(name)` | `bool` | Remove an agent, returns `True` if found |
| `list_agents()` | `list[str]` | Get all registered agent names |
| `list_configs()` | `list[SubAgentConfig]` | Get all configurations |
| `list_compiled()` | `list[CompiledSubAgent]` | Get all compiled agents |
| `exists(name)` | `bool` | Check if an agent is registered |
| `count()` | `int` | Number of registered agents |
| `clear()` | `None` | Remove all agents |
| `get_summary()` | `str` | Formatted summary of all agents |

### Integration with pydantic-deep

When using subagents with [`pydantic-deep`](https://github.com/vstorm-co/pydantic-deep), pass the registry to both the subagent toolset and the agent factory toolset so they share state:

```python
from pydantic_ai import Agent
from subagents_pydantic_ai import (
    create_subagent_toolset,
    create_agent_factory_toolset,
    DynamicAgentRegistry,
    SubAgentConfig,
)

# Shared registry
registry = DynamicAgentRegistry(max_agents=5)

# Pre-configured subagents
base_subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics",
        instructions="You are a research assistant.",
    ),
]

# The subagent toolset receives the registry so task() can find dynamic agents
subagent_toolset = create_subagent_toolset(
    default_model="openai:gpt-4.1",
    subagents=base_subagents,
    registry=registry,
)

# The factory toolset uses the same registry to register new agents
factory_toolset = create_agent_factory_toolset(
    default_model="openai:gpt-4.1",
    registry=registry,
    allowed_models=["openai:gpt-4.1", "openai:gpt-4o-mini"],
    max_agents=5,
)

agent = Agent(
    "openai:gpt-4.1",
    deps_type=Deps,
    toolsets=[subagent_toolset, factory_toolset],
)
```

With this setup:

- The parent can delegate to pre-configured subagents ("researcher") via `task()`
- The parent can create new subagents at runtime via `create_agent()`
- Newly created agents are immediately available to `task()` because both toolsets share the same `registry` instance
- When the parent removes a dynamic agent, it is no longer discoverable by `task()`

## Default Agent Factory

By default, `create_agent_factory_toolset` builds a plain `pydantic_ai.Agent` for every
dynamically created agent. Pass `default_agent_factory` to override this with your own
builder. The factory receives a `SubAgentConfig` and must return an agent instance.

This is useful when you want every dynamically created agent to be built with a specific
framework (e.g., pydantic-deep) instead of a bare `Agent`:

```python
from pydantic_deep import create_deep_agent
from subagents_pydantic_ai import (
    create_agent_factory_toolset,
    DynamicAgentRegistry,
    SubAgentConfig,
)

def my_factory(config: SubAgentConfig):
    return create_deep_agent(
        model=config.get("model", "openai:gpt-4.1"),
        system_prompt=config["instructions"],
    )

factory_toolset = create_agent_factory_toolset(
    registry=DynamicAgentRegistry(),
    default_agent_factory=my_factory,
)
```

When `default_agent_factory` is set, it is called for **every** `create_agent()` invocation.
If it is `None` (the default), the toolset falls back to creating a standard `pydantic_ai.Agent`.

!!! note
    `default_agent_factory` on `create_agent_factory_toolset` applies to dynamically created
    agents only. For pre-configured subagents, use the `agent` or `agent_factory` fields on
    [`SubAgentConfig`](../concepts/types.md#subagentconfig) instead.

## Creating Agents at Runtime

The parent agent can create new subagents:

```python
# Parent agent calls:
create_agent(
    name="rust-expert",
    description="Expert in Rust programming",
    instructions="You are a Rust programming expert. Help with Rust code.",
    model="openai:gpt-4o",
)
# Returns: "Created agent 'rust-expert'"

# Now the agent can be used:
task(
    description="Review this Rust code for memory safety",
    subagent_type="rust-expert",
    mode="sync",
)
```

## Use Cases

### Domain-Specific Experts

Create experts for domains not covered by pre-configured agents:

```python
# User asks about a niche topic
# Parent creates a specialist:
create_agent(
    name="kubernetes-expert",
    description="Expert in Kubernetes and container orchestration",
    instructions="""You are a Kubernetes expert.
    Help with:
    - Deployment configurations
    - Service mesh setup
    - Troubleshooting pods
    - Resource optimization
    """,
)
```

### Language-Specific Helpers

Create helpers for different programming languages:

```python
create_agent(
    name="go-helper",
    description="Go programming assistant",
    instructions="You help with Go programming. Follow Go idioms and best practices.",
)

create_agent(
    name="swift-helper",
    description="Swift/iOS development assistant",
    instructions="You help with Swift and iOS development.",
)
```

### Persona-Based Agents

Create agents with specific personas:

```python
create_agent(
    name="devil-advocate",
    description="Challenges ideas constructively",
    instructions="""You play devil's advocate.
    When given an idea or plan:
    - Find potential weaknesses
    - Challenge assumptions
    - Suggest alternative approaches
    - Be constructive, not dismissive
    """,
)
```

## Limits and Security

### Model Restrictions

Only allow specific models:

```python
create_agent_factory_toolset(
    registry=registry,
    allowed_models=["openai:gpt-4o-mini"],  # Only allow mini
)
```

### Agent Limits

Prevent unlimited agent creation when the toolset creates its internal registry:

```python
create_subagent_toolset(
    default_model="openai:gpt-4.1",
    delegation_configuration="persisted",
    max_agents=3,  # Maximum 3 dynamic agents
)
```

After reaching the limit, creating new agents will fail until existing ones are removed.

`max_agents` only applies to the registry `create_subagent_toolset` builds for
itself. Pass your own `registry` and it keeps its own cap instead.

#### Letting the parent free a slot

`create_subagent_toolset` exposes no `remove_agent` tool, so a parent in
`"persisted"` or `"persisted_and_oneshot"` cannot free a slot once `max_agents`
is hit — it is told to remove an agent with no tool to do it. Give it the
management tools from `create_agent_factory_toolset` over a shared registry, and
leave the subagent toolset on `"default"`:

```python
registry = DynamicAgentRegistry()

agent = Agent(
    "openai:gpt-4o",
    toolsets=[
        # Task-only. Creating, listing and removing agents lives in the factory
        # toolset, the only one that exposes `remove_agent`.
        create_subagent_toolset(default_model="openai:gpt-4o", registry=registry),
        create_agent_factory_toolset(
            default_model="openai:gpt-4o", registry=registry, max_agents=3
        ),
    ],
)
```

Agents the parent creates are immediately reachable through `task`, because both
toolsets share one registry.

!!! warning "One owner for `create_agent`"

    Do not combine `"persisted"` or `"persisted_and_oneshot"` with
    `create_agent_factory_toolset`. Both define a tool named `create_agent`, and
    pydantic-ai rejects duplicate tool names across toolsets at run time:

    ```
    UserError: FunctionToolset 'agent_factory' defines a tool whose name conflicts
    with existing tool from FunctionToolset 'subagents': 'create_agent'.
    ```

    Pick one owner for agent creation: `"persisted"` for creation without
    management, or `"default"` plus the factory toolset for both. Note that
    `create_agent_factory_toolset` writes its `max_agents` onto the registry, so
    pass the limit there rather than to `DynamicAgentRegistry(...)`.

## Delegation Modes

`delegation_configuration` controls the primary tools exposed to the parent:

| Mode | Entry-point tools | Use case |
|------|-------------------|----------|
| `"default"` | `task` | Backward-compatible configured/registry delegation |
| `"persisted"` | `create_agent`, `task` | Reusable specialists |
| `"persisted_and_oneshot"` | `create_agent`, `task`, `delegate` | Reusable and ad-hoc specialists |
| `"oneshot_only"` | `delegate` | Minimal one-shot delegation |

Async lifecycle tools such as `check_task` remain available in every mode.

A mode that hides a tool also hides that tool's configuration, so passing
configuration the mode can never consult raises `ValueError` at construction
instead of being dropped in silence:

| Mode | Rejects | Because |
|------|---------|---------|
| `"oneshot_only"` | `subagents`, `registry` | Only `task` can reach either |
| `"default"` | `allowed_models`, `capabilities_map`, `default_agent_factory` | Only `create_agent` / `delegate` consult them |

Custom `default_agent_factory` implementations should not attach an
`ask_parent` toolset; the toolset injects it at run time for registry-backed
and one-shot agents.

```python
from subagents_pydantic_ai import create_subagent_toolset

toolset = create_subagent_toolset(
    default_model="openai:gpt-4o",
    delegation_configuration="persisted_and_oneshot",
    allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
    capabilities_map={
        "filesystem": lambda deps: [create_fs_toolset(deps.backend)],
    },
)
```

### `delegate` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `description` | `str` | Task prompt for the specialist |
| `instructions` | `str` | Specialist system prompt |
| `name` | `str` | Label for logs and async task handles (letters, numbers, hyphens) |
| `model` | `str \| None` | Optional model override |
| `capabilities` | `list[str] \| None` | Optional capability names |
| `can_ask_questions` | `bool` | Whether the specialist can ask the parent (default: `True`) |
| `mode` | `str` | `"sync"`, `"async"`, or `"auto"` (default: `"sync"`) |

### Registry Semantics

One-shot specialists are **ephemeral**:

- They are **not** registered in `DynamicAgentRegistry`
- They do **not** count toward `max_agents`
- They cannot be reused via `task(subagent_type=...)`, even though they have a `name`
- `name` is a display label for logging and async task handles
- They report **no chat trace**. A `Chat Trace ID` is only useful if `task` can
  resolve the subagent it belongs to, which is never true for a one-shot, so
  `delegate` results omit it and one-shot runs never occupy a slot in the
  `max_chat_traces` LRU that a continuable conversation could use. Use
  `create_agent` + `task` when you want a resumable conversation.

Use `delegate` for ad-hoc specialists. Use `create_agent` + `task` when you need a reusable agent that persists in the registry.

### Naming Conflicts

Dynamic agents cannot override pre-configured agents:

```text
# If "researcher" is pre-configured:
create_agent(name="researcher", ...)
# Returns: "Error: Agent 'researcher' already exists"
```

## Best Practices

### 1. Use Pre-Configured When Possible

Dynamic creation has overhead. Pre-configure common agents:

```text
# Good: Pre-configure known specialists
subagents = [
    SubAgentConfig(name="researcher", ...),
    SubAgentConfig(name="writer", ...),
    SubAgentConfig(name="coder", ...),
]

# Use dynamic only for truly unexpected needs
```

### 2. Clear Naming

Use descriptive names for dynamic agents:

```text
# Good
create_agent(name="react-typescript-expert", ...)

# Bad
create_agent(name="agent1", ...)
```

### 3. Focused Instructions

Keep dynamic agent instructions focused:

```python
# Good: Focused
create_agent(
    name="sql-optimizer",
    instructions="You optimize SQL queries for PostgreSQL.",
)

# Bad: Too broad
create_agent(
    name="helper",
    instructions="You help with everything.",
)
```

## Next Steps

- [Message Bus](message-bus.md) - Communication layer
- [Examples](../examples/index.md) - Working examples
