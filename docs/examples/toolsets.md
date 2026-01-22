# Giving Subagents Tools

This example shows how to provide tools to your subagents.

## Using toolsets_factory

The `toolsets_factory` creates tools for each subagent:

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from subagents_pydantic_ai import create_subagent_toolset, SubAgentConfig


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)
    workspace: str = "/tmp/workspace"

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            workspace=self.workspace,
        )


# Custom toolset for file operations
def create_file_tools():
    """Create simple file operation tools."""
    from pydantic_ai import Toolset

    toolset = Toolset()

    @toolset.tool
    async def read_file(path: str) -> str:
        """Read a file's contents."""
        with open(path) as f:
            return f.read()

    @toolset.tool
    async def write_file(path: str, content: str) -> str:
        """Write content to a file."""
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"

    return toolset


def my_toolsets_factory(deps: Deps) -> list:
    """Create toolsets for subagents."""
    return [create_file_tools()]


subagents = [
    SubAgentConfig(
        name="coder",
        description="Writes and manages code files",
        instructions="""You write Python code.

You have access to file operations:
- read_file(path): Read a file
- write_file(path, content): Write to a file

Write clean, well-documented code.
""",
    ),
]

toolset = create_subagent_toolset(
    subagents=subagents,
    toolsets_factory=my_toolsets_factory,
)

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    toolsets=[toolset],
    system_prompt="You can delegate coding tasks to the coder subagent.",
)


async def main():
    result = await agent.run(
        "Create a Python script that prints 'Hello World' and save it to /tmp/hello.py",
        deps=Deps(),
    )
    print(result.output)


asyncio.run(main())
```

## Per-Subagent Toolsets

Give different tools to different subagents:

```python
def selective_toolsets_factory(deps: Deps) -> list:
    """Create toolsets based on the subagent context."""
    # You could check deps or other context to customize
    return [create_file_tools()]


# Or specify toolsets directly in the config
subagents = [
    SubAgentConfig(
        name="reader",
        description="Reads and analyzes files",
        instructions="You read files and analyze their contents.",
        toolsets=[create_read_only_tools()],  # Only read access
    ),
    SubAgentConfig(
        name="writer",
        description="Writes files",
        instructions="You write files.",
        toolsets=[create_file_tools()],  # Full access
    ),
]
```

## Using pydantic-ai-backend

For comprehensive file operations, use pydantic-ai-backend:

```python
from pydantic_ai_backends import create_console_toolset, StateBackend


@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)
    backend: StateBackend = field(default_factory=StateBackend)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy(),
            backend=self.backend,  # Share the backend
        )


def backend_toolsets_factory(deps: Deps) -> list:
    """Create console toolset from backend."""
    return [create_console_toolset(deps.backend)]


subagents = [
    SubAgentConfig(
        name="coder",
        description="Writes code with full filesystem access",
        instructions="""You have access to filesystem operations:
- ls(path): List directory contents
- read_file(path): Read a file
- write_file(path, content): Write to a file
- edit_file(path, old, new): Edit a file
- glob(pattern): Find files matching pattern
- grep(pattern, path): Search file contents
""",
    ),
]

toolset = create_subagent_toolset(
    subagents=subagents,
    toolsets_factory=backend_toolsets_factory,
)
```

## Built-in Tools

Use Pydantic AI's built-in tools:

```python
from pydantic_ai import BuitinTools

subagents = [
    SubAgentConfig(
        name="web-researcher",
        description="Researches using web search",
        instructions="You research topics using web search.",
        agent_kwargs={
            "builtin_tools": [BuitinTools.web_search],
        },
    ),
]
```

## Combining Multiple Toolsets

```python
def comprehensive_toolsets_factory(deps: Deps) -> list:
    """Create multiple toolsets for subagents."""
    return [
        create_console_toolset(deps.backend),  # File operations
        create_todo_toolset(),  # Task tracking
        create_search_toolset(),  # Search capabilities
    ]
```

## Best Practices

### 1. Principle of Least Privilege

Only give subagents the tools they need:

```python
# Good: Specific tools for specific tasks
SubAgentConfig(
    name="reader",
    instructions="You read and analyze files.",
    toolsets=[create_read_only_tools()],  # No write access
)

# Avoid: Giving everything to everyone
SubAgentConfig(
    name="helper",
    toolsets=[all_tools],  # Too permissive
)
```

### 2. Document Available Tools

Include tool documentation in instructions:

```python
SubAgentConfig(
    name="coder",
    instructions="""You write Python code.

Available tools:
- read_file(path): Read file contents
- write_file(path, content): Write to file
- execute(command): Run shell command

Always read existing files before modifying them.
""",
)
```

### 3. Handle Tool Errors

Guide subagents on error handling:

```python
instructions="""
If a tool call fails:
1. Check if the path/parameters are correct
2. Try an alternative approach
3. Report the error clearly if unresolvable
"""
```

## Next Steps

- [Questions](questions.md) - Parent-child communication
- [Nesting](nesting.md) - Subagents with their own subagents
