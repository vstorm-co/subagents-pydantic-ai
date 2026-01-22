# Task Cancellation

Subagents for Pydantic AI provides two cancellation mechanisms: **soft cancel** and **hard cancel**.

## Soft Cancellation

Soft cancel requests cooperative cancellation. The subagent receives a signal and should stop gracefully.

```python
# Parent agent calls:
soft_cancel_task(task_id="abc123")
```

### How It Works

1. Parent calls `soft_cancel_task()`
2. A `CANCEL_REQUEST` message is sent to the subagent
3. Subagent should check for cancellation and stop gracefully
4. Subagent returns partial results or cancellation acknowledgment
5. Task status becomes `CANCELLED`

### When to Use

- Task is taking too long
- Requirements have changed
- Parent wants to try a different approach
- Graceful shutdown needed

## Hard Cancellation

Hard cancel forces immediate termination.

```python
# Parent agent calls:
hard_cancel_task(task_id="abc123")
```

### How It Works

1. Parent calls `hard_cancel_task()`
2. Task is immediately terminated
3. No cleanup or partial results
4. Task status becomes `CANCELLED`

### When to Use

- Soft cancel didn't work
- Urgent need to stop
- Task is stuck or unresponsive
- Resource constraints

!!! warning "Use with Caution"
    Hard cancellation may leave resources in an inconsistent state. Prefer soft cancellation when possible.

## Implementing Cancellation Handling

Guide subagents to handle cancellation gracefully:

```python
SubAgentConfig(
    name="researcher",
    instructions="""You are a research assistant.

## Cancellation Handling

If you receive a cancellation request:
1. Stop your current work
2. Save any partial results
3. Return what you have with a note that work was cancelled

Example response for cancelled task:
"Research was cancelled. Partial findings:
- Found 3 relevant sources
- Initial analysis suggests [...]
- Further investigation was not completed"
""",
)
```

## Cancellation States

| State | Description |
|-------|-------------|
| `PENDING` | Can be cancelled (never started) |
| `RUNNING` | Can be cancelled (soft or hard) |
| `WAITING_FOR_ANSWER` | Can be cancelled |
| `COMPLETED` | Cannot be cancelled (already done) |
| `FAILED` | Cannot be cancelled (already done) |
| `CANCELLED` | Already cancelled |

## Example: Cancellation Workflow

```python
# Start a long-running task
task(
    description="Comprehensive market research",
    subagent_type="researcher",
    mode="async",
)
# Returns: "Task started with ID: research-123"

# Check progress
check_task(task_id="research-123")
# Returns: "Task is running"

# Decide to cancel
soft_cancel_task(task_id="research-123")
# Returns: "Cancellation requested for task research-123"

# Check final status
check_task(task_id="research-123")
# Returns: "Task cancelled. Partial results: [...]"
```

## Handling Multiple Tasks

When cancelling multiple tasks:

```python
# List active tasks
list_active_tasks()
# Returns list of task IDs and statuses

# Cancel specific tasks
soft_cancel_task(task_id="task-a")
soft_cancel_task(task_id="task-b")

# Or hard cancel if needed
hard_cancel_task(task_id="task-c")
```

## Best Practices

### 1. Prefer Soft Cancellation

Always try soft cancel first:

```python
# First attempt
soft_cancel_task(task_id="abc123")

# Wait and check
check_task(task_id="abc123")

# Only if still running after reasonable time
hard_cancel_task(task_id="abc123")
```

### 2. Design for Cancellation

Structure subagent work so it can be stopped at checkpoints:

```python
instructions="""
Work in phases:
1. Gather sources (checkpoint)
2. Initial analysis (checkpoint)
3. Deep dive (checkpoint)
4. Synthesis

At each checkpoint, check for cancellation and save progress.
If cancelled, return results from completed phases.
"""
```

### 3. Communicate Cancellation Clearly

Return useful information when cancelled:

```python
instructions="""
If cancelled, return:
- What was completed
- What was in progress
- What wasn't started
- Any preliminary findings
"""
```

## Next Steps

- [Dynamic Agents](dynamic-agents.md) - Runtime agent creation
- [Message Bus](message-bus.md) - Communication layer
