# Examples

Practical examples showing how to use subagents-pydantic-ai in real scenarios.

## Basic Research Assistant

A simple setup with a researcher and writer subagent.

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    """Dependencies for the agent."""
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


# Define subagents
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics and gathers information",
        instructions="""You are a research assistant.

When given a topic:
1. Break it down into key aspects
2. Investigate each aspect
3. Provide a comprehensive summary with sources""",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes content based on research",
        instructions="""You are a technical writer.

Write clear, engaging content:
- Use simple language
- Include code examples where appropriate
- Structure with headers and bullet points""",
    ),
]

# Create agent
toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You are an assistant that can delegate tasks.

Use the researcher for gathering information.
Use the writer for creating content.
For long research tasks, use mode="async" to run in background.""",
)

# Run
result = agent.run_sync(
    "Research Python type hints and write a tutorial about them",
    deps=Deps(),
)
print(result.output)
```

## Parallel Research Tasks

Run multiple research tasks concurrently using async mode.

```python
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
        description="Researches topics thoroughly",
        instructions="Research the given topic in depth.",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You can run multiple research tasks in parallel.

When asked to compare topics:
1. Start async tasks for each topic: task(..., mode="async")
2. Use list_active_tasks() to monitor progress
3. Use check_task(task_id) to get results
4. Combine results into final comparison""",
)

result = agent.run_sync(
    "Compare Python, Rust, and Go for web development",
    deps=Deps(),
)
print(result.output)
```

## Code Review Pipeline

A multi-agent pipeline for code review.

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)
    code_to_review: str = ""

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            code_to_review=self.code_to_review,
        )


subagents = [
    SubAgentConfig(
        name="security-reviewer",
        description="Reviews code for security vulnerabilities",
        instructions="""You are a security expert.

Check for:
- SQL injection
- XSS vulnerabilities
- Insecure data handling
- Authentication issues""",
        can_ask_questions=False,
    ),
    SubAgentConfig(
        name="style-reviewer",
        description="Reviews code style and best practices",
        instructions="""You are a code style expert.

Check for:
- PEP 8 compliance
- Naming conventions
- Code organization
- Documentation""",
        can_ask_questions=False,
    ),
    SubAgentConfig(
        name="performance-reviewer",
        description="Reviews code for performance issues",
        instructions="""You are a performance expert.

Check for:
- Algorithm complexity
- Memory usage
- Database query efficiency
- Caching opportunities""",
        can_ask_questions=False,
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You coordinate code reviews.

For each review request:
1. Send code to all reviewers in parallel (mode="async")
2. Collect all feedback
3. Synthesize into a unified review report""",
)

code = '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
'''

result = agent.run_sync(
    f"Review this code:\n```python\n{code}\n```",
    deps=Deps(code_to_review=code),
)
print(result.output)
```

## Interactive Q&A with Subagents

Subagents that can ask the parent for clarification.

```python
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
        name="planner",
        description="Creates project plans",
        instructions="""You create detailed project plans.

If requirements are unclear, ask for clarification.
Include:
- Phases and milestones
- Resource requirements
- Timeline estimates
- Risk factors""",
        can_ask_questions=True,
        max_questions=3,
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You help users plan projects.

When the planner asks questions:
1. Check task status with check_task(task_id)
2. If status is "waiting_for_answer", use answer_subagent()
3. Provide helpful answers based on user context""",
)

result = agent.run_sync(
    "Create a plan for building a web application",
    deps=Deps(),
)
print(result.output)
```

## Dynamic Agent Creation

Create specialized agents at runtime.

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from subagents_pydantic_ai import (
    create_subagent_toolset,
    create_agent_factory_toolset,
    DynamicAgentRegistry,
)


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


registry = DynamicAgentRegistry(max_agents=5)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[
        create_subagent_toolset(),  # Base toolset
        create_agent_factory_toolset(
            registry=registry,
            allowed_models=["openai:gpt-4o", "openai:gpt-4o-mini"],
        ),
    ],
    system_prompt="""You can create specialized agents on demand.

Use create_agent() when you need a specialist that doesn't exist.
Use list_agents() to see available agents.
Use remove_agent() to clean up unused agents.

Example: For a task about databases, create a "database-expert" agent.""",
)

result = agent.run_sync(
    "Help me optimize my PostgreSQL queries for a high-traffic application",
    deps=Deps(),
)
print(result.output)
```

## Subagents with Tools

Give subagents access to external tools.

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


# Create toolsets for subagents
def create_subagent_tools(deps: Deps) -> list[FunctionToolset[Deps]]:
    """Factory that creates toolsets for subagents."""

    toolset: FunctionToolset[Deps] = FunctionToolset()

    @toolset.tool
    async def search_documentation(query: str) -> str:
        """Search the documentation for information."""
        # Implement actual search logic
        return f"Documentation results for: {query}"

    @toolset.tool
    async def run_code(code: str) -> str:
        """Execute Python code and return the result."""
        # Implement sandboxed execution
        return f"Executed: {code}"

    return [toolset]


subagents = [
    SubAgentConfig(
        name="code-assistant",
        description="Helps with coding tasks, can search docs and run code",
        instructions="""You are a coding assistant.

Use search_documentation() to find relevant information.
Use run_code() to test code snippets.
Provide working solutions with explanations.""",
    ),
]

toolset = create_subagent_toolset(
    subagents=subagents,
    toolsets_factory=create_subagent_tools,
)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="Delegate coding tasks to the code-assistant.",
)

result = agent.run_sync(
    "Write a function to calculate fibonacci numbers efficiently",
    deps=Deps(),
)
print(result.output)
```

## Task Cancellation

Handle long-running tasks with cancellation support.

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
        name="analyzer",
        description="Performs deep analysis (may take a while)",
        instructions="Perform thorough analysis of the given topic.",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="""You can manage long-running tasks.

For analysis tasks:
1. Start with mode="async"
2. Check status periodically with check_task()
3. Use soft_cancel_task() for graceful cancellation
4. Use hard_cancel_task() for immediate termination""",
)


async def main():
    result = await agent.run(
        "Analyze the entire history of computing, but cancel if it takes too long",
        deps=Deps(),
    )
    print(result.output)


asyncio.run(main())
```

## Nested Subagents

Allow subagents to delegate to other subagents.

```python
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
        name="project-manager",
        description="Manages complex projects",
        instructions="""You are a project manager.

You can delegate to other subagents:
- Use 'developer' for coding tasks
- Use 'tester' for testing tasks
- Use 'documenter' for documentation""",
    ),
    SubAgentConfig(
        name="developer",
        description="Writes code",
        instructions="Write clean, well-structured code.",
    ),
    SubAgentConfig(
        name="tester",
        description="Writes and runs tests",
        instructions="Write comprehensive tests.",
    ),
    SubAgentConfig(
        name="documenter",
        description="Writes documentation",
        instructions="Write clear documentation.",
    ),
]

toolset = create_subagent_toolset(
    subagents=subagents,
    max_nesting_depth=1,  # Allow one level of nesting
)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="Delegate complex projects to the project-manager.",
)

result = agent.run_sync(
    "Build a REST API for a todo list application",
    deps=Deps(),
)
print(result.output)
```

## Structured Output

Get structured responses from subagents.

```python
from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel
from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


class AnalysisResult(BaseModel):
    """Structured analysis result."""
    summary: str
    key_points: list[str]
    recommendations: list[str]
    confidence: float


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(subagents={} if max_depth <= 0 else self.subagents.copy())


subagents = [
    SubAgentConfig(
        name="analyst",
        description="Analyzes topics and provides structured insights",
        instructions="""Analyze the given topic.

Provide:
- A brief summary
- Key points (3-5 items)
- Recommendations (2-3 items)
- Confidence level (0.0-1.0)""",
    ),
]

toolset = create_subagent_toolset(subagents=subagents)

# Main agent with structured output
agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    output_type=AnalysisResult,
    toolsets=[toolset],
    system_prompt="""Use the analyst subagent and return structured results.""",
)

result = agent.run_sync(
    "Analyze the benefits of microservices architecture",
    deps=Deps(),
)
print(f"Summary: {result.output.summary}")
print(f"Key Points: {result.output.key_points}")
print(f"Recommendations: {result.output.recommendations}")
print(f"Confidence: {result.output.confidence}")
```

## Next Steps

- [Quickstart](quickstart.md) - Basic setup guide
- [Dual-Mode Execution](dual-mode.md) - Sync vs async in detail
- [API Reference](api-reference.md) - Complete API documentation
