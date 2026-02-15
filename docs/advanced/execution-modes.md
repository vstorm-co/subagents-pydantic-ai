# Execution Modes

Subagents for Pydantic AI supports three execution modes: **sync**, **async**, and **auto**.

## Overview

| Mode | Behavior | Use Case |
|------|----------|----------|
| `sync` | Block until complete | Quick tasks, immediate results needed |
| `async` | Run in background | Long tasks, parallel execution |
| `auto` | System decides | Let the library choose optimally |

## Sync Mode

In sync mode, the parent agent waits for the subagent to complete before continuing.

```python
# Parent agent calls:
task(
    description="Calculate the sum of 1 to 100",
    subagent_type="calculator",
    mode="sync",
)
# Blocks here until done
# Returns: "The sum is 5050"
```

**Best for:**

- Quick calculations
- When result is needed immediately
- Back-and-forth communication
- Simple transformations

## Async Mode

In async mode, the task runs in the background and returns immediately with a task ID.

```python
# Parent agent calls:
task(
    description="Research the history of Python",
    subagent_type="researcher",
    mode="async",
)
# Returns immediately: "Task started with ID: abc123"

# Later, check status:
check_task(task_id="abc123")
# Returns status, result if complete, or pending question
```

**Best for:**

- Long-running research
- Parallel tasks
- When parent can do other work meanwhile
- Complex analysis

### Task Lifecycle

```
PENDING → RUNNING → COMPLETED
                 ↘ WAITING_FOR_ANSWER → RUNNING → ...
                 ↘ FAILED
                 ↘ CANCELLED
```

## Auto Mode

Auto mode uses heuristics to decide between sync and async.

```python
# Parent agent calls:
task(
    description="Analyze this dataset",
    subagent_type="analyst",
    mode="auto",
)
# System decides based on task characteristics
```

### Decision Logic

The [`decide_execution_mode()`][subagents_pydantic_ai.types.decide_execution_mode] function considers the following in order of priority:

1. **Explicit override**: If `force_mode` is set (not "auto"), use it directly
2. **Config preference**: Subagent's `preferred_mode` setting (if not "auto")
3. **User context**: Tasks requiring user context → sync
4. **Clarification + time-sensitive**: May need clarification AND is time-sensitive → sync
5. **Complexity + independence**: Complex tasks that can run independently → async
6. **Simplicity**: Simple tasks → sync
7. **Independence fallback**: Moderate complexity that can run independently → async
8. **Final default**: Everything else → sync

```python
from subagents_pydantic_ai import decide_execution_mode, TaskCharacteristics

characteristics = TaskCharacteristics(
    estimated_complexity="complex",
    requires_user_context=False,
    is_time_sensitive=False,
    can_run_independently=True,
    may_need_clarification=False,
)

mode = decide_execution_mode(characteristics, config)
# Returns "async" for complex independent tasks
```

### TaskCharacteristics Fields

The [`TaskCharacteristics`][subagents_pydantic_ai.types.TaskCharacteristics] dataclass holds the properties used for auto-mode decision-making:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `estimated_complexity` | `"simple" \| "moderate" \| "complex"` | `"moderate"` | Expected task complexity level |
| `requires_user_context` | `bool` | `False` | Whether task needs ongoing user interaction |
| `is_time_sensitive` | `bool` | `False` | Whether quick response is important |
| `can_run_independently` | `bool` | `True` | Whether task can complete without further input |
| `may_need_clarification` | `bool` | `False` | Whether task might need clarifying questions |

These fields can be set explicitly when calling the `task()` tool, or they are inferred from the subagent's `SubAgentConfig` at runtime.

### Auto-Mode Decision Examples

The following table shows how different combinations of `TaskCharacteristics` resolve to sync or async:

| Scenario | Complexity | User Context | Time Sensitive | Independent | Clarification | Result |
|----------|-----------|-------------|---------------|-------------|--------------|--------|
| Quick lookup | `simple` | `False` | `True` | `True` | `False` | **sync** |
| Deep research | `complex` | `False` | `False` | `True` | `False` | **async** |
| Interactive editing | `moderate` | `True` | `False` | `False` | `True` | **sync** |
| Background analysis | `moderate` | `False` | `False` | `True` | `False` | **async** |
| Urgent clarification | `moderate` | `False` | `True` | `True` | `True` | **sync** |
| Complex but dependent | `complex` | `False` | `False` | `False` | `True` | **sync** |

#### Example 1: Auto picks sync for a simple task

```python
# Simple question answering - auto resolves to sync
task(
    description="What is the capital of France?",
    subagent_type="general-purpose",
    mode="auto",
    complexity="simple",
)
# Auto-mode sees: simple complexity → sync
# Parent gets the answer immediately
```

#### Example 2: Auto picks async for complex independent work

```python
# Deep research - auto resolves to async
task(
    description="Analyze the codebase architecture and produce a report",
    subagent_type="researcher",
    mode="auto",
    complexity="complex",
)
# Auto-mode sees: complex + can_run_independently → async
# Returns task handle immediately, parent continues working
```

#### Example 3: Auto picks sync when user context is needed

```python
# Task needing ongoing user interaction - auto resolves to sync
task(
    description="Help me refine this paragraph",
    subagent_type="editor",
    mode="auto",
    requires_user_context=True,
)
# Auto-mode sees: requires_user_context=True → sync
# Ensures interactive back-and-forth is possible
```

#### Example 4: Auto picks sync for time-sensitive tasks that may need clarification

```python
# Urgent task that might need questions - auto resolves to sync
task(
    description="Fix this production bug quickly",
    subagent_type="coder",
    mode="auto",
    complexity="moderate",
    may_need_clarification=True,
)
# Auto-mode sees: may_need_clarification + is_time_sensitive → sync
# (assuming the subagent config has is_time_sensitive context)
```

#### Example 5: Auto picks async for moderate independent work

```python
# Moderate task that can run alone - auto resolves to async
task(
    description="Generate test cases for the user service",
    subagent_type="coder",
    mode="auto",
    complexity="moderate",
)
# Auto-mode sees: moderate + can_run_independently → async
# Parent can start other tasks in parallel
```

### Configuring Auto-Mode Hints

Guide auto-mode with subagent configuration. The `preferred_mode`, `typical_complexity`, and `typically_needs_context` fields in `SubAgentConfig` provide hints to the auto-mode decision logic:

```python
SubAgentConfig(
    name="deep-researcher",
    description="Does thorough research",
    instructions="...",
    preferred_mode="async",  # Hint: prefer async
    typical_complexity="complex",  # Usually complex tasks
    typically_needs_context=False,  # Can work independently
)
```

When a subagent has `preferred_mode` set to "sync" or "async", auto-mode respects that preference before evaluating task characteristics. The `typical_complexity` value is used as a fallback when the caller does not pass an explicit `complexity` parameter to `task()`.

```python
# This subagent always runs sync due to preferred_mode
SubAgentConfig(
    name="quick-helper",
    description="Answers simple questions instantly",
    instructions="...",
    preferred_mode="sync",  # Always sync, regardless of characteristics
)

# This subagent lets auto-mode decide but hints at complexity
SubAgentConfig(
    name="analyst",
    description="Performs data analysis",
    instructions="...",
    preferred_mode="auto",  # Let auto decide
    typical_complexity="complex",  # Hint: tasks are usually complex
    typically_needs_context=False,  # Hint: can work independently
)
```

## Managing Async Tasks

### Checking Status

```python
# Agent calls:
check_task(task_id="abc123")
```

Returns different responses based on status:

| Status | Response |
|--------|----------|
| `PENDING` | "Task is queued" |
| `RUNNING` | "Task is running" |
| `WAITING_FOR_ANSWER` | "Task needs answer: [question]" |
| `COMPLETED` | "Task complete: [result]" |
| `FAILED` | "Task failed: [error]" |
| `CANCELLED` | "Task was cancelled" |

### Listing Active Tasks

```python
# Agent calls:
list_active_tasks()
```

Returns a list of all non-completed tasks with their status.

### Answering Questions

If a subagent asks a question (when `can_ask_questions=True`):

```python
# Agent calls:
answer_subagent(task_id="abc123", answer="Use PostgreSQL")
```

The subagent continues with the provided answer.

### Cancellation

**Soft cancel** - request cooperative cancellation:

```python
soft_cancel_task(task_id="abc123")
```

The subagent receives a cancellation request and should stop gracefully.

**Hard cancel** - immediate termination:

```python
hard_cancel_task(task_id="abc123")
```

Forces immediate stop, may leave resources in inconsistent state.

## Parallel Execution

Run multiple tasks simultaneously:

```python
# Agent's thought process:
# "I need to research three topics. I'll start them all in parallel."

# Agent calls:
task(description="Research topic A", subagent_type="researcher", mode="async")
# Returns: "Task started with ID: task-a"

task(description="Research topic B", subagent_type="researcher", mode="async")
# Returns: "Task started with ID: task-b"

task(description="Research topic C", subagent_type="researcher", mode="async")
# Returns: "Task started with ID: task-c"

# Agent does other work...

# Agent checks results:
check_task(task_id="task-a")
check_task(task_id="task-b")
check_task(task_id="task-c")
```

## Best Practices

### 1. Use Sync for Interactive Tasks

When you need immediate feedback or multiple rounds of communication:

```python
SubAgentConfig(
    name="editor",
    description="Edits text interactively",
    instructions="...",
    preferred_mode="sync",
)
```

### 2. Use Async for Heavy Work

For tasks that take significant time:

```python
SubAgentConfig(
    name="analyzer",
    description="Performs deep analysis",
    instructions="...",
    preferred_mode="async",
    typical_complexity="complex",
)
```

### 3. Let Auto Decide When Unsure

If you're not sure which mode is best:

```python
task(
    description="...",
    subagent_type="worker",
    mode="auto",  # Let the system decide
)
```

## Next Steps

- [Questions](questions.md) - Parent-child communication
- [Cancellation](cancellation.md) - Managing task lifecycle
- [Examples](../examples/sync-async.md) - Working examples
