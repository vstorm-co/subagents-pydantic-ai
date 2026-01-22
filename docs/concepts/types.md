# Types

This page documents the key data structures used in Subagents for Pydantic AI.

## Configuration Types

### SubAgentConfig

Configuration for defining a subagent.

```python
from subagents_pydantic_ai import SubAgentConfig

config = SubAgentConfig(
    name="researcher",
    description="Researches topics",
    instructions="You are a research assistant.",
    model="openai:gpt-4o",  # optional
    can_ask_questions=True,  # optional
    max_questions=5,  # optional
)
```

See [Subagents](subagents.md) for full documentation.

### CompiledSubAgent

A pre-compiled subagent ready for use. Created internally by the toolset.

```python
from subagents_pydantic_ai import CompiledSubAgent

@dataclass
class CompiledSubAgent:
    name: str               # Unique identifier
    description: str        # Brief description
    config: SubAgentConfig  # Original configuration
    agent: object | None    # Agent instance
```

## Execution Types

### ExecutionMode

How a task should be executed.

```python
from subagents_pydantic_ai import ExecutionMode

# Type alias
ExecutionMode = Literal["sync", "async", "auto"]
```

| Mode | Description |
|------|-------------|
| `sync` | Block until complete |
| `async` | Run in background |
| `auto` | Automatically decide |

### TaskStatus

Status of a background task.

```python
from subagents_pydantic_ai import TaskStatus

class TaskStatus(str, Enum):
    PENDING = "pending"           # Queued but not started
    RUNNING = "running"           # Currently executing
    WAITING_FOR_ANSWER = "waiting_for_answer"  # Blocked on question
    COMPLETED = "completed"       # Finished successfully
    FAILED = "failed"             # Failed with error
    CANCELLED = "cancelled"       # Was cancelled
```

### TaskPriority

Priority levels for background tasks.

```python
from subagents_pydantic_ai import TaskPriority

class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
```

### TaskHandle

Handle for managing a background task.

```python
from subagents_pydantic_ai import TaskHandle

@dataclass
class TaskHandle:
    task_id: str              # Unique identifier
    subagent_name: str        # Name of executing subagent
    description: str          # Task description
    status: TaskStatus        # Current status
    priority: TaskPriority    # Task priority
    created_at: datetime      # When created
    started_at: datetime | None   # When started
    completed_at: datetime | None # When finished
    result: str | None        # Result (if completed)
    error: str | None         # Error (if failed)
    pending_question: str | None  # Question waiting for answer
```

### TaskCharacteristics

Characteristics used for auto-mode selection.

```python
from subagents_pydantic_ai import TaskCharacteristics

@dataclass
class TaskCharacteristics:
    estimated_complexity: Literal["simple", "moderate", "complex"] = "moderate"
    requires_user_context: bool = False
    is_time_sensitive: bool = False
    can_run_independently: bool = True
    may_need_clarification: bool = False
```

## Message Types

### MessageType

Types of messages between agents.

```python
from subagents_pydantic_ai import MessageType

class MessageType(str, Enum):
    TASK_ASSIGNED = "task_assigned"
    TASK_UPDATE = "task_update"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    QUESTION = "question"
    ANSWER = "answer"
    CANCEL_REQUEST = "cancel_request"
    CANCEL_FORCED = "cancel_forced"
```

### AgentMessage

Message passed between agents via the message bus.

```python
from subagents_pydantic_ai import AgentMessage

@dataclass
class AgentMessage:
    type: MessageType       # Message type
    sender: str             # Sender agent ID
    receiver: str           # Receiver agent ID
    payload: Any            # Message data
    task_id: str            # Associated task
    id: str                 # Unique message ID
    timestamp: datetime     # When created
    correlation_id: str | None  # For request-response
```

## Protocol Types

### SubAgentDepsProtocol

Protocol that your dependencies must implement.

```python
from subagents_pydantic_ai import SubAgentDepsProtocol

class SubAgentDepsProtocol(Protocol):
    subagents: dict[str, Any]

    def clone_for_subagent(self, max_depth: int = 0) -> Self:
        """Create a copy of deps for a subagent."""
        ...
```

**Example implementation:**

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Deps:
    subagents: dict[str, Any] = field(default_factory=dict)

    def clone_for_subagent(self, max_depth: int = 0) -> "Deps":
        return Deps(
            subagents={} if max_depth <= 0 else self.subagents.copy()
        )
```

### MessageBusProtocol

Protocol for custom message bus implementations.

```python
from subagents_pydantic_ai import MessageBusProtocol

class MessageBusProtocol(Protocol):
    async def send(self, message: AgentMessage) -> None: ...
    async def receive(self, agent_id: str, timeout: float | None = None) -> AgentMessage | None: ...
    async def subscribe(self, agent_id: str) -> None: ...
    async def unsubscribe(self, agent_id: str) -> None: ...
```

## Utility Functions

### decide_execution_mode

Decide between sync and async based on characteristics.

```python
from subagents_pydantic_ai import decide_execution_mode, TaskCharacteristics

characteristics = TaskCharacteristics(
    estimated_complexity="complex",
    can_run_independently=True,
)

mode = decide_execution_mode(
    characteristics=characteristics,
    config=my_subagent_config,
    force_mode=None,  # or "sync"/"async" to override
)
# Returns "async" for complex independent tasks
```

## Type Exports

All types are exported from the main module:

```python
from subagents_pydantic_ai import (
    # Configuration
    SubAgentConfig,
    CompiledSubAgent,
    # Execution
    ExecutionMode,
    TaskStatus,
    TaskPriority,
    TaskHandle,
    TaskCharacteristics,
    # Messages
    MessageType,
    AgentMessage,
    # Protocols
    SubAgentDepsProtocol,
    MessageBusProtocol,
    # Functions
    decide_execution_mode,
)
```

## Next Steps

- [Execution Modes](../advanced/execution-modes.md) - Learn about sync vs async
- [API Reference](../api/types.md) - Complete API documentation
