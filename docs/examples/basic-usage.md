# Basic Usage

This example shows how to add subagent delegation to a Pydantic AI agent.

## Complete Example

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


# Step 1: Create dependencies that implement SubAgentDepsProtocol
@dataclass
class Deps:
    """Dependencies for our agent."""

    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        """Create a copy for subagents."""
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


# Step 2: Define specialized subagents
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics and gathers information",
        instructions="""You are a research assistant.

When given a research task:
1. Break down the topic into key questions
2. Provide factual, well-organized information
3. Include relevant examples
4. Cite sources when possible
""",
    ),
    SubAgentConfig(
        name="summarizer",
        description="Summarizes long content into concise points",
        instructions="""You are a summarization expert.

When summarizing:
1. Identify the main points
2. Remove redundancy
3. Keep essential details
4. Use bullet points for clarity
""",
    ),
]

# Step 3: Create the toolset
toolset = create_subagent_toolset(subagents=subagents)

# Step 4: Create the parent agent with the toolset
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You are a helpful assistant that can delegate tasks
to specialized subagents.

Available subagents:
- researcher: For research tasks
- summarizer: For summarizing content

Use the task() tool to delegate work to the appropriate subagent.
""",
)


async def main():
    deps = Deps()

    # The agent will delegate to the researcher subagent
    result = await agent.run(
        "Research the benefits of async programming in Python",
        deps=deps,
    )

    print("Result:")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
```

## Step-by-Step Breakdown

### 1. Dependencies

Your dependencies must implement `SubAgentDepsProtocol`:

```python
@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())
```

The `clone_for_subagent` method creates isolated dependencies for each subagent.

### 2. Subagent Configuration

Define what each subagent does:

```python
SubAgentConfig(
    name="researcher",  # Unique identifier
    description="Researches topics",  # Shown to parent
    instructions="You are a research assistant...",  # System prompt
)
```

### 3. Toolset Creation

Create the toolset that adds delegation tools:

```python
toolset = create_subagent_toolset(subagents=subagents)
```

### 4. Agent Integration

Add the toolset to your agent:

```python
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
)
```

## What Happens at Runtime

1. User asks: "Research the benefits of async programming"
2. Parent agent decides to delegate to "researcher"
3. Parent calls: `task(description="...", subagent_type="researcher")`
4. Researcher subagent executes the task
5. Result is returned to parent
6. Parent formats and returns final response

## Running the Example

```bash
# Set your API key
export OPENAI_API_KEY=your-key

# Run the script
python basic_usage.py
```

## Next Steps

- [Sync vs Async](sync-async.md) - Learn about execution modes
- [Giving Subagents Tools](toolsets.md) - Provide capabilities to subagents
