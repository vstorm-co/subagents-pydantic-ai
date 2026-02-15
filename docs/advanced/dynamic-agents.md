# Dynamic Agent Creation

Create specialized agents at runtime using the agent factory toolset.

## Overview

While pre-configured subagents cover most use cases, sometimes you need to create agents dynamically:

- User requests a specialist for an unexpected domain
- Task requires a unique combination of capabilities
- Experimentation with different agent configurations

## Agent Factory Toolset

Add dynamic creation capabilities:

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
        create_subagent_toolset(subagents=base_subagents),
        create_agent_factory_toolset(
            registry=registry,
            allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
            max_agents=5,
        ),
    ],
)
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `registry` | `DynamicAgentRegistry` | Required | Registry for created agents |
| `allowed_models` | `list[str]` | All | Models agents can use |
| `max_agents` | `int` | `10` | Maximum dynamic agents |
| `default_model` | `str \| None` | `None` | Default model for new agents |

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
    subagents=base_subagents,
    registry=registry,
)

# The factory toolset uses the same registry to register new agents
factory_toolset = create_agent_factory_toolset(
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

Prevent unlimited agent creation:

```python
create_agent_factory_toolset(
    registry=registry,
    max_agents=3,  # Maximum 3 dynamic agents
)
```

After reaching the limit, creating new agents will fail until existing ones are removed.

### Naming Conflicts

Dynamic agents cannot override pre-configured agents:

```python
# If "researcher" is pre-configured:
create_agent(name="researcher", ...)
# Returns: "Error: Agent 'researcher' already exists"
```

## Best Practices

### 1. Use Pre-Configured When Possible

Dynamic creation has overhead. Pre-configure common agents:

```python
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

```python
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
