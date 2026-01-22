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

The `decide_execution_mode()` function considers:

1. **Explicit override**: If `force_mode` is set (not "auto"), use it
2. **Config preference**: Subagent's `preferred_mode` setting
3. **User context**: Tasks needing user context → sync
4. **Complexity**: Complex + independent → async
5. **Simplicity**: Simple tasks → sync
6. **Independence**: Can run without input → async

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

### Configuring Auto-Mode Hints

Guide auto-mode with subagent configuration:

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
