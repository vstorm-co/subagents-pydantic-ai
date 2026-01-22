# Sync vs Async Execution

This example demonstrates the difference between sync and async execution modes.

## Sync Mode Example

Sync mode blocks until the subagent completes:

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
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


subagents = [
    SubAgentConfig(
        name="calculator",
        description="Performs calculations",
        instructions="You perform mathematical calculations accurately.",
        preferred_mode="sync",  # Hint: this is a quick task
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You can delegate calculations to the calculator subagent.
Use mode="sync" for quick calculations where you need the result immediately.
""",
)


async def main():
    result = await agent.run(
        "Calculate the factorial of 10",
        deps=Deps(),
    )
    print(result.output)
    # The agent delegated with mode="sync"
    # It waited for the result before responding


asyncio.run(main())
```

## Async Mode Example

Async mode runs in the background:

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
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


subagents = [
    SubAgentConfig(
        name="researcher",
        description="Performs thorough research",
        instructions="You research topics comprehensively.",
        preferred_mode="async",  # Hint: this takes time
        typical_complexity="complex",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You can delegate research to the researcher subagent.

For long research tasks:
1. Use mode="async" to start the task
2. You'll get a task_id back
3. Continue with other work if needed
4. Use check_task(task_id) to get results
""",
)


async def main():
    # First interaction: Start the research
    result1 = await agent.run(
        "Start researching the history of Python. Use async mode.",
        deps=Deps(),
    )
    print("First response:", result1.output)
    # Output: "I've started the research task. Task ID: abc123"

    # Second interaction: Check results
    result2 = await agent.run(
        "Check the status of the research task",
        deps=Deps(),
        message_history=result1.all_messages(),
    )
    print("Second response:", result2.output)
    # Output: "The research is complete. Here are the findings..."


asyncio.run(main())
```

## Parallel Tasks Example

Run multiple async tasks simultaneously:

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
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches any topic",
        instructions="You research topics thoroughly.",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You can delegate research tasks.

For multiple independent research topics:
1. Start each as an async task
2. Collect all task IDs
3. Check each task for results
4. Combine findings
""",
)


async def main():
    deps = Deps()

    # Start parallel research
    result1 = await agent.run(
        """Research these three topics in parallel:
        1. Python async/await
        2. Python generators
        3. Python context managers

        Start each as an async task, then check results.""",
        deps=deps,
    )

    print(result1.output)
    # The agent started 3 async tasks, then checked each for results


asyncio.run(main())
```

## Auto Mode Example

Let the system decide:

```python
subagents = [
    SubAgentConfig(
        name="worker",
        description="General purpose worker",
        instructions="You complete various tasks.",
        # No preferred_mode - will use auto
    ),
]

# In the agent's prompt:
system_prompt = """You can delegate tasks to the worker subagent.

Use mode="auto" and the system will decide:
- Simple tasks → sync (immediate result)
- Complex tasks → async (background)
"""
```

## Choosing the Right Mode

| Scenario | Mode | Why |
|----------|------|-----|
| Quick calculation | `sync` | Need result immediately |
| Simple transformation | `sync` | Fast to complete |
| Deep research | `async` | Takes time |
| Multiple tasks | `async` | Run in parallel |
| Interactive refinement | `sync` | Need back-and-forth |
| Unknown complexity | `auto` | Let system decide |

## Next Steps

- [Giving Subagents Tools](toolsets.md) - Provide capabilities
- [Questions](questions.md) - Parent-child communication
