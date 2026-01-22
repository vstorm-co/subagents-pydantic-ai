# Nested Subagents

This example shows how subagents can have their own subagents.

## Overview

Subagents can delegate to other subagents, creating hierarchical workflows:

```
Parent Agent
    └── Manager Subagent
            ├── Researcher Subagent
            └── Writer Subagent
```

## Basic Nesting

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        # Important: Reduce depth for nested subagents
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
        )


# Level 2: Leaf subagents (no further delegation)
leaf_subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches specific topics",
        instructions="You research topics and provide facts.",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes content",
        instructions="You write clear, engaging content.",
    ),
]

# Level 1: Manager that delegates to leaf subagents
manager_subagents = [
    SubAgentConfig(
        name="content-manager",
        description="Manages content creation by delegating to specialists",
        instructions="""You manage content creation.

You have access to these specialists:
- researcher: Researches topics
- writer: Writes content

Workflow:
1. Delegate research to researcher
2. Delegate writing to writer
3. Review and compile results
""",
    ),
]


def create_manager_toolsets(deps: Deps) -> list:
    """Create toolsets for the manager, including nested delegation."""
    return [
        create_subagent_toolset(
            subagents=leaf_subagents,
            max_nesting_depth=0,  # Leaf subagents can't delegate further
        ),
    ]


# Top-level toolset
toolset = create_subagent_toolset(
    subagents=manager_subagents,
    toolsets_factory=create_manager_toolsets,
    max_nesting_depth=1,  # Allow one level of nesting
)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate complex content tasks to the content-manager.",
)


async def main():
    result = await agent.run(
        "Create a blog post about machine learning",
        deps=Deps(),
    )

    print(result.output)
    # The content-manager delegated to researcher, then to writer


asyncio.run(main())
```

## Controlling Nesting Depth

Use `max_nesting_depth` to limit how deep delegation can go:

```python
# No nesting (depth=0)
toolset = create_subagent_toolset(
    subagents=subagents,
    max_nesting_depth=0,  # Subagents cannot delegate
)

# One level (depth=1)
toolset = create_subagent_toolset(
    subagents=subagents,
    max_nesting_depth=1,  # Subagents can delegate to leaf agents
)

# Two levels (depth=2)
toolset = create_subagent_toolset(
    subagents=subagents,
    max_nesting_depth=2,  # Subagents can delegate to agents that can delegate
)
```

## Hierarchical Team Structure

```python
# Executive level
executive = SubAgentConfig(
    name="product-lead",
    description="Leads product development",
    instructions="""You lead product development.

Delegate to:
- engineering-manager: For technical work
- design-manager: For design work
""",
)

# Manager level
managers = [
    SubAgentConfig(
        name="engineering-manager",
        description="Manages engineering tasks",
        instructions="""You manage engineering.

Delegate to:
- backend-dev: Backend development
- frontend-dev: Frontend development
""",
    ),
    SubAgentConfig(
        name="design-manager",
        description="Manages design tasks",
        instructions="You manage design work.",
    ),
]

# Individual contributor level
ics = [
    SubAgentConfig(
        name="backend-dev",
        description="Backend developer",
        instructions="You write backend code.",
    ),
    SubAgentConfig(
        name="frontend-dev",
        description="Frontend developer",
        instructions="You write frontend code.",
    ),
]


def create_manager_toolsets(deps):
    return [create_subagent_toolset(subagents=ics, max_nesting_depth=0)]


def create_executive_toolsets(deps):
    return [
        create_subagent_toolset(
            subagents=managers,
            toolsets_factory=create_manager_toolsets,
            max_nesting_depth=1,
        )
    ]


toolset = create_subagent_toolset(
    subagents=[executive],
    toolsets_factory=create_executive_toolsets,
    max_nesting_depth=2,
)
```

## Clone for Subagent

The `clone_for_subagent` method is crucial for nesting:

```python
@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)
    shared_state: dict = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        """Create isolated deps for subagent.

        Args:
            max_depth: How many more levels of nesting to allow.
                       0 = this subagent cannot delegate further.
        """
        return Deps(
            # Clear subagents if no more nesting allowed
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            # Share read-only state
            shared_state=self.shared_state,
        )
```

## Best Practices

### 1. Limit Depth

Don't nest too deeply. 2-3 levels is usually enough:

```python
# Good: Clear hierarchy
Parent → Manager → Worker

# Avoid: Too deep
Parent → Director → Manager → Lead → Worker → Helper
```

### 2. Clear Responsibilities

Each level should have distinct responsibilities:

```python
# Good: Clear separation
"project-manager": Coordinates overall project
"tech-lead": Makes technical decisions
"developer": Writes code

# Avoid: Overlapping roles
"helper1": Does stuff
"helper2": Also does stuff
```

### 3. Avoid Circular Delegation

Ensure subagents don't create delegation loops:

```python
# Bad: A delegates to B, B delegates to A
SubAgentConfig(name="A", instructions="Delegate to B")
SubAgentConfig(name="B", instructions="Delegate to A")

# Good: Clear hierarchy with termination
SubAgentConfig(name="manager", instructions="Delegate to workers")
SubAgentConfig(name="worker", instructions="Complete the task directly")
```

## Next Steps

- [Research Team](research-team.md) - Multi-agent workflow example
