"""Type definitions for the subagent system.

This module contains all the data structures used throughout the library,
including configuration types, message types, and task management types.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic_ai.models import Model
from typing_extensions import NotRequired, TypedDict


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""

    TASK_ASSIGNED = "task_assigned"
    """A new task has been assigned to a subagent."""

    TASK_UPDATE = "task_update"
    """Progress update from a running task."""

    TASK_COMPLETED = "task_completed"
    """Task finished successfully."""

    TASK_FAILED = "task_failed"
    """Task failed with an error."""

    QUESTION = "question"
    """Subagent is asking the parent a question."""

    ANSWER = "answer"
    """Parent's response to a subagent question."""

    CANCEL_REQUEST = "cancel_request"
    """Request to cancel a task (soft cancel)."""

    CANCEL_FORCED = "cancel_forced"
    """Immediate cancellation (hard cancel)."""


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    """Task is queued but not started."""

    RUNNING = "running"
    """Task is currently executing."""

    WAITING_FOR_ANSWER = "waiting_for_answer"
    """Task is blocked waiting for parent response."""

    COMPLETED = "completed"
    """Task finished successfully."""

    FAILED = "failed"
    """Task failed with an error."""

    CANCELLED = "cancelled"
    """Task was cancelled."""


# Type aliases
ExecutionMode = Literal["sync", "async", "auto"]
"""Execution mode for subagent tasks.

- sync: Execute synchronously, blocking until completion (default)
- async: Execute in background, return immediately with task handle
- auto: Automatically decide based on task characteristics
"""


class TaskPriority(str, Enum):
    """Priority levels for background tasks."""

    LOW = "low"
    """Low priority task, can be deferred."""

    NORMAL = "normal"
    """Normal priority task (default)."""

    HIGH = "high"
    """High priority task, should be processed soon."""

    CRITICAL = "critical"
    """Critical priority task, process immediately."""


@dataclass
class TaskCharacteristics:
    """Characteristics that help decide execution mode.

    These characteristics are used by `decide_execution_mode` to automatically
    select between sync and async execution based on task properties.

    Attributes:
        estimated_complexity: Expected task complexity level.
        requires_user_context: Whether task needs ongoing user interaction.
        is_time_sensitive: Whether quick response is important.
        can_run_independently: Whether task can complete without further input.
        may_need_clarification: Whether task might need clarifying questions.
    """

    estimated_complexity: Literal["simple", "moderate", "complex"] = "moderate"
    requires_user_context: bool = False
    is_time_sensitive: bool = False
    can_run_independently: bool = True
    may_need_clarification: bool = False


ToolsetFactory = Callable[[Any], list[Any]]
"""Factory function that creates toolsets for a subagent.

Takes deps as input and returns a list of toolsets to register.

Example:
    ```python
    def my_toolset_factory(deps: MyDeps) -> list[Any]:
        return [
            create_file_toolset(deps.backend),
            create_todo_toolset(),
        ]
    ```
"""


class SubAgentConfig(TypedDict, total=False):
    """Configuration for a subagent.

    Defines the name, description, and instructions for a subagent.
    Used by the toolset to create agent instances.

    Required fields:
        name: Unique identifier for the subagent
        description: Brief description shown to parent agent
        instructions: System prompt for the subagent

    Optional fields:
        model: LLM model to use (defaults to parent's default)
        can_ask_questions: Whether subagent can ask parent questions
        max_questions: Maximum questions per task
        preferred_mode: Default execution mode preference for this subagent
        typical_complexity: Typical task complexity for this subagent
        typically_needs_context: Whether this subagent typically needs user context
        toolsets: Additional toolsets to register with the subagent
        agent_kwargs: Additional kwargs passed to Agent constructor (e.g., builtin_tools)

    Example with builtin_tools:
        ```python
        SubAgentConfig(
            name="researcher",
            description="Research agent with web search",
            instructions="You research topics using web search.",
            agent_kwargs={"builtin_tools": [BuitinTools.web_search]},
        )
        ```
    """

    name: str
    description: str
    instructions: str
    model: NotRequired[str | Model]
    can_ask_questions: NotRequired[bool]
    max_questions: NotRequired[int]
    preferred_mode: NotRequired[Literal["sync", "async", "auto"]]
    typical_complexity: NotRequired[Literal["simple", "moderate", "complex"]]
    typically_needs_context: NotRequired[bool]
    toolsets: NotRequired[list[Any]]
    agent_kwargs: NotRequired[dict[str, Any]]


def _generate_message_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


@dataclass
class AgentMessage:
    """Message passed between agents via the message bus.

    Attributes:
        type: The message type (task_assigned, question, etc.)
        sender: ID of the sending agent
        receiver: ID of the receiving agent
        payload: Message-specific data
        task_id: Associated task ID for correlation
        id: Unique message identifier for tracing/debugging
        timestamp: When the message was created
        correlation_id: ID for request-response correlation
    """

    type: MessageType
    sender: str
    receiver: str
    payload: Any
    task_id: str
    id: str = field(default_factory=_generate_message_id)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str | None = None


@dataclass
class TaskHandle:
    """Handle for managing a background task.

    Returned when a task is started in async mode. Use this to
    check status, get results, or cancel the task.

    Attributes:
        task_id: Unique identifier for the task
        subagent_name: Name of the subagent executing the task
        description: Task description
        status: Current task status
        priority: Task priority level
        created_at: When the task was created
        started_at: When execution started
        completed_at: When execution finished
        result: Task result (if completed)
        error: Error message (if failed)
        pending_question: Question waiting for answer (if any)
    """

    task_id: str
    subagent_name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    pending_question: str | None = None


@dataclass
class CompiledSubAgent:
    """A pre-compiled subagent ready for use.

    After processing SubAgentConfig, the toolset creates a CompiledSubAgent
    that includes the actual agent instance.

    Attributes:
        name: Unique identifier for the subagent.
        description: Brief description of the subagent's purpose.
        agent: The actual agent instance.
        config: The original configuration used to create this agent.
    """

    name: str
    description: str
    config: SubAgentConfig
    agent: object | None = None  # Agent instance - typed as object to avoid circular imports


def decide_execution_mode(
    characteristics: TaskCharacteristics,
    config: SubAgentConfig,
    force_mode: ExecutionMode | None = None,
) -> Literal["sync", "async"]:
    """Decide whether to run sync or async based on task characteristics.

    This function implements the auto-mode selection logic. It considers:
    1. Explicit force_mode override
    2. Config-level preferred_mode
    3. Task characteristics

    Args:
        characteristics: Task characteristics that influence the decision.
        config: Subagent configuration with optional preferences.
        force_mode: Override mode (if specified and not "auto").

    Returns:
        The resolved execution mode: either "sync" or "async".

    Example:
        ```python
        characteristics = TaskCharacteristics(
            estimated_complexity="complex",
            can_run_independently=True,
        )
        config = SubAgentConfig(name="worker", ...)
        mode = decide_execution_mode(characteristics, config)
        # mode will be "async" for complex independent tasks
        ```
    """
    # Explicit override takes precedence
    if force_mode and force_mode != "auto":
        return force_mode

    # Config-level preference
    config_preference = config.get("preferred_mode", "auto")
    if config_preference != "auto":
        return config_preference

    # Always sync if needs user context or clarification likely
    if characteristics.requires_user_context:
        return "sync"
    if characteristics.may_need_clarification and characteristics.is_time_sensitive:
        return "sync"

    # Prefer async for complex, independent tasks
    if characteristics.estimated_complexity == "complex" and characteristics.can_run_independently:
        return "async"

    # Simple tasks - sync is fine
    if characteristics.estimated_complexity == "simple":
        return "sync"

    # Default to async for moderate complexity if can run independently
    if characteristics.can_run_independently:
        return "async"

    return "sync"
