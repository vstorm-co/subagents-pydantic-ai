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

The registry tracks dynamically created agents:

```python
from subagents_pydantic_ai import DynamicAgentRegistry

registry = DynamicAgentRegistry()

# Check registered agents
agents = registry.list_agents()

# Get a specific agent
agent = registry.get_agent("custom-analyst")

# Remove an agent
registry.remove_agent("custom-analyst")
```

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
