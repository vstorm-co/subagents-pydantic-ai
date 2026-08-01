# Subagents

Subagents are specialized agents that handle specific types of tasks. Each subagent has its own system prompt, optional tools, and configuration.

## SubAgentConfig

The [`SubAgentConfig`][subagents_pydantic_ai.types.SubAgentConfig] TypedDict defines a subagent:

```python
from subagents_pydantic_ai import SubAgentConfig

subagent = SubAgentConfig(
    name="researcher",
    description="Researches topics and gathers information",
    instructions="You are a research assistant. Investigate thoroughly and provide sources.",
)
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier for the subagent |
| `description` | `str` | Brief description shown to the parent agent |
| `instructions` | `str` | System prompt for the subagent |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | Parent's model | LLM model to use |
| `can_ask_questions` | `bool` | `True` | Attach the `ask_parent` tool; `False` removes it |
| `max_questions` | `int` | `None` (unlimited) | Enforced cap on `ask_parent` calls per delegation |
| `preferred_mode` | `str` | `"auto"` | `"sync"`, `"async"`, or `"auto"` |
| `typical_complexity` | `str` | `"moderate"` | `"simple"`, `"moderate"`, or `"complex"` |
| `typically_needs_context` | `bool` | `False` | Hint for auto-mode selection |
| `toolsets` | `list` | `[]` | Additional toolsets for the subagent |
| `agent_kwargs` | `dict` | `{}` | Extra kwargs for Agent constructor |
| `context_files` | `list[str]` | `[]` | Context file paths injected into the system prompt (with pydantic-deep) |
| `extra` | `dict` | `{}` | Extensibility dict for consumer libraries (not read by this library) |
| `max_retries` | `int` | `3` | Extra retries on transient failures (`0` disables) |
| `retry_initial_delay` | `float` | `1.0` | Seconds before the first retry |
| `retry_max_delay` | `float` | `30.0` | Cap on the backoff delay |
| `retry_backoff_multiplier` | `float` | `2.0` | Delay growth factor per attempt |
| `retry_jitter` | `bool` | `True` | Randomise the backoff delay |
| `retry_on` | `Callable` | built-in | Custom transient-error predicate |

See [Auto-Retry](../advanced/retries.md) for the retry fields in detail.

## Defining Subagents

### Basic Definition

```python
subagents = [
    SubAgentConfig(
        name="researcher",
        description="Researches topics and gathers information",
        instructions="You are a research assistant.",
    ),
    SubAgentConfig(
        name="writer",
        description="Writes content based on research",
        instructions="You are a technical writer.",
    ),
]
```

### With Model Override

```python
SubAgentConfig(
    name="fast-helper",
    description="Quick helper for simple tasks",
    instructions="You help with simple tasks quickly.",
    model="openai:gpt-4o-mini",  # Use a faster/cheaper model
)
```

### With Questions Enabled

```python
SubAgentConfig(
    name="analyst",
    description="Analyzes data with clarifying questions",
    instructions="You analyze data. Ask questions when data is ambiguous.",
    can_ask_questions=True,
    max_questions=5,
)
```

### With Execution Mode Hints

```python
SubAgentConfig(
    name="deep-researcher",
    description="Does thorough research that takes time",
    instructions="You do comprehensive research.",
    preferred_mode="async",  # Suggest async execution
    typical_complexity="complex",  # Hint for auto-mode
)
```

### With Additional Tools

```python
from pydantic_ai_backends import create_console_toolset

SubAgentConfig(
    name="coder",
    description="Writes and tests code",
    instructions="You write Python code.",
    toolsets=[create_console_toolset()],  # Give file access
)
```

### With Built-in Tools

```python
from pydantic_ai.builtin_tools import WebSearchTool

SubAgentConfig(
    name="web-researcher",
    description="Researches using web search",
    instructions="You research topics using web search.",
    agent_kwargs={"builtin_tools": [WebSearchTool()]},
)
```

## Subagent Descriptions

The `description` field is crucial - it tells the parent agent when to use each subagent:

```python
# Good: the description says what the subagent can actually do
SubAgentConfig(
    name="sql-expert",
    description="Writes and optimizes SQL queries for PostgreSQL databases",
    instructions="You write and optimize PostgreSQL queries.",
)

SubAgentConfig(
    name="code-reviewer",
    description="Reviews Python code for bugs, security issues, and style",
    instructions="You review Python code and report concrete findings.",
)

# Bad: the parent model has no basis for choosing this over anything else
SubAgentConfig(
    name="helper",
    description="Helps with stuff",
    instructions="You help.",
)
```

## Subagent Instructions

Write clear, focused instructions:

```python
SubAgentConfig(
    name="code-reviewer",
    description="Reviews code for quality issues",
    instructions="""You are an expert code reviewer.

When reviewing code:
1. Look for bugs and logic errors
2. Check for security vulnerabilities
3. Assess code style and readability
4. Suggest improvements with examples

Format your response as:
## Summary
[Brief assessment]

## Issues Found
[List of issues with severity]

## Recommendations
[Actionable suggestions]
""",
)
```

## Best Practices

### 1. Single Responsibility

Each subagent should do one thing well:

```text
# Good - focused
SubAgentConfig(name="test-writer", description="Generates pytest tests", ...)
SubAgentConfig(name="doc-writer", description="Writes documentation", ...)

# Bad - too broad
SubAgentConfig(name="developer", description="Does all development tasks", ...)
```

### 2. Clear Naming

Use descriptive, hyphenated names:

```python
# Good
"code-reviewer", "sql-expert", "test-writer"

# Bad
"cr", "agent1", "myAgent"
```

### 3. Appropriate Tools

Only give subagents the tools they need:

```python
# Researcher doesn't need file write access
SubAgentConfig(
    name="researcher",
    description="Researches topics",
    instructions="...",
    # No toolsets needed
)

# Coder needs file access
SubAgentConfig(
    name="coder",
    description="Writes code",
    instructions="...",
    toolsets=[create_console_toolset()],
)
```

## Next Steps

- [Toolset](toolset.md) - Learn about the delegation toolset
- [Execution Modes](../advanced/execution-modes.md) - Sync vs async execution
- [Questions](../advanced/questions.md) - Parent-child communication
