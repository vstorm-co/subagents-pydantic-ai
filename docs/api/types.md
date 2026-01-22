# Types API

## SubAgentConfig

::: subagents_pydantic_ai.SubAgentConfig
    options:
      show_root_heading: true
      show_source: true

## CompiledSubAgent

::: subagents_pydantic_ai.CompiledSubAgent
    options:
      show_root_heading: true
      show_source: true

## TaskHandle

::: subagents_pydantic_ai.TaskHandle
    options:
      show_root_heading: true
      show_source: true

## TaskStatus

::: subagents_pydantic_ai.TaskStatus
    options:
      show_root_heading: true
      show_source: true

## TaskPriority

::: subagents_pydantic_ai.TaskPriority
    options:
      show_root_heading: true
      show_source: true

## ExecutionMode

::: subagents_pydantic_ai.ExecutionMode
    options:
      show_root_heading: true
      show_source: true

## TaskCharacteristics

::: subagents_pydantic_ai.TaskCharacteristics
    options:
      show_root_heading: true
      show_source: true

## AgentMessage

::: subagents_pydantic_ai.AgentMessage
    options:
      show_root_heading: true
      show_source: true

## MessageType

::: subagents_pydantic_ai.MessageType
    options:
      show_root_heading: true
      show_source: true

## ToolsetFactory

::: subagents_pydantic_ai.ToolsetFactory
    options:
      show_root_heading: true
      show_source: true

## decide_execution_mode

::: subagents_pydantic_ai.decide_execution_mode
    options:
      show_root_heading: true
      show_source: true

---

## Usage Examples

### Creating a SubAgentConfig

```python
from subagents_pydantic_ai import SubAgentConfig

config = SubAgentConfig(
    name="researcher",
    description="Researches topics",
    instructions="You are a research assistant.",
    model="openai:gpt-4o",
    can_ask_questions=True,
    max_questions=3,
    preferred_mode="async",
    typical_complexity="complex",
)
```

### Working with TaskHandle

```python
from subagents_pydantic_ai import TaskHandle, TaskStatus

# TaskHandle is returned by async tasks
handle: TaskHandle = ...

# Check status
if handle.status == TaskStatus.COMPLETED:
    print(f"Result: {handle.result}")
elif handle.status == TaskStatus.WAITING_FOR_ANSWER:
    print(f"Question: {handle.pending_question}")
elif handle.status == TaskStatus.FAILED:
    print(f"Error: {handle.error}")
```

### Using decide_execution_mode

```python
from subagents_pydantic_ai import (
    decide_execution_mode,
    TaskCharacteristics,
    SubAgentConfig,
)

characteristics = TaskCharacteristics(
    estimated_complexity="complex",
    requires_user_context=False,
    can_run_independently=True,
)

config = SubAgentConfig(
    name="worker",
    description="...",
    instructions="...",
)

mode = decide_execution_mode(characteristics, config)
# Returns "async" for complex, independent tasks
```

### Message Types

```python
from subagents_pydantic_ai import MessageType, AgentMessage

# Create a question message
message = AgentMessage(
    type=MessageType.QUESTION,
    sender="subagent-123",
    receiver="parent-456",
    payload={"question": "Which database?"},
    task_id="task-789",
)
```
