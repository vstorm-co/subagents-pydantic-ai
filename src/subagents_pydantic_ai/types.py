"""Type definitions for the subagent system.

This module contains all the data structures used throughout the library,
including configuration types, message types, and task management types.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import RunContext, UsageLimits
from pydantic_ai.models import Model
from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from pydantic_ai import DeferredToolRequests
    from pydantic_ai.agent import EventStreamHandler
    from pydantic_ai.messages import FinishReason
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.usage import RunUsage


def utcnow() -> datetime:
    """The current time as a timezone-aware UTC timestamp.

    Task timestamps are compared and subtracted (elapsed time, eviction order), so
    they must be unambiguous. Naive local timestamps make that arithmetic wrong
    across a DST transition.
    """
    return datetime.now(timezone.utc)


class _ValueStrEnum(str, Enum):
    """A `str` enum that renders as its value on every supported Python version.

    Python 3.11 changed `Enum.__str__` and `Enum.__format__` for mixin enums to
    render `ClassName.MEMBER`, so `f"{TaskStatus.COMPLETED}"` produced
    `TaskStatus.COMPLETED` instead of `completed` and leaked the member name into
    model-facing tool output. `enum.StrEnum` fixes this but only exists on 3.11+.
    """

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return str.__format__(str(self.value), format_spec)


class MessageType(_ValueStrEnum):
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


class TaskStatus(_ValueStrEnum):
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

    RETRYING = "retrying"
    """Task hit a transient error and is waiting to retry."""

    DEFERRED = "deferred"
    """Task suspended for human approval or a deferred tool call.

    Distinct from `FAILED`: nothing went wrong, the subagent is waiting on a
    decision this library cannot make. Terminal here because resuming a
    suspension belongs to whoever holds the deferred requests -- see
    `TaskHandle.deferred_requests`.
    """


# Type aliases
ExecutionMode = Literal["sync", "async", "auto"]
"""Execution mode for subagent tasks.

- sync: Execute synchronously, blocking until completion (default)
- async: Execute in background, return immediately with task handle
- auto: Automatically decide based on task characteristics
"""

DelegationConfiguration = Literal[
    "default",
    "persisted",
    "persisted_and_oneshot",
    "oneshot_only",
]
"""Controls which delegation entry-point tools are exposed.

- default: Expose ``task`` only (backward-compatible with existing users)
- persisted: Expose ``create_agent`` and ``task``
- persisted_and_oneshot: Expose ``create_agent``, ``task``, and ``delegate``
- oneshot_only: Expose only ``delegate`` as the delegation entry point
"""


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DEFERRED}
)
"""Statuses a task never leaves. The first terminal transition wins."""


class TaskPriority(_ValueStrEnum):
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


ToolsetFactory = Callable[[Any], "Sequence[AbstractToolset[Any]]"]
"""Factory function that creates toolsets for a subagent.

Takes the subagent's deps as input and returns the toolsets to register. The deps
parameter is `Any` because its type belongs to the application, not this library.
The return type is a `Sequence` so a factory annotated with a concrete toolset
type (`list[FunctionToolset[MyDeps]]`) still satisfies it -- `list` is invariant,
`Sequence` is not.

Example:
    ```python
    def my_toolset_factory(deps: MyDeps) -> list[FunctionToolset[MyDeps]]:
        return [
            create_file_toolset(deps.backend),
            create_todo_toolset(),
        ]
    ```
"""


AskUserCallback = Callable[[str], Awaitable[str]]
"""Callback invoked when a subagent calls `ask_parent` in sync mode.

Receives the subagent's question and must return the answer. Typically wired
to a human-in-the-loop UI, a CLI `input()` prompt, or a pre-canned answerer
for tests.

Example:
    ```python
    async def ask_user(question: str) -> str:
        return input(f"Subagent asks: {question}\n> ")

    toolset = create_subagent_toolset(
        subagents=subagents,
        ask_user=ask_user,
    )
    ```
"""


class _SubAgentConfigRequired(TypedDict):
    """The keys every `SubAgentConfig` must carry.

    Split out so `total=False` on `SubAgentConfig` applies only to the optional
    keys. The library indexes these three directly (`config["name"]`), so making
    them optional to the type checker turned a configuration mistake into a
    `KeyError` raised in the middle of a delegation.
    """

    name: str
    description: str
    instructions: str


class SubAgentConfig(_SubAgentConfigRequired, total=False):
    """Configuration for a subagent.

    Defines the name, description, and instructions for a subagent.
    Used by the toolset to create agent instances.

    Required fields:
        name: Unique identifier for the subagent
        description: Brief description shown to parent agent
        instructions: System prompt for the subagent

    Optional fields:
        model: LLM model to use (defaults to parent's default)
        agent: Pre-built agent instance. When provided, `_compile_subagent`
            uses this instead of creating a new `Agent`. Useful for passing
            agents created by frameworks like pydantic-deep.
        agent_factory: Callable that receives the SubAgentConfig and returns
            an agent instance. Called by `_compile_subagent` if `agent`
            is not provided. Signature: `(config: SubAgentConfig) -> Agent`.
        can_ask_questions: Whether subagent can ask parent questions
        max_questions: Maximum questions per task
        preferred_mode: Default execution mode preference for this subagent
        typical_complexity: Typical task complexity for this subagent
        typically_needs_context: Whether this subagent typically needs user context
        toolsets: Additional toolsets to register with the subagent
        agent_kwargs: Additional kwargs passed to Agent constructor (e.g., builtin_tools)
        context_files: List of context file paths in the backend.
            When used with pydantic-deep, these are loaded via ContextToolset
            and injected into this subagent's system prompt. Each subagent
            can have its own context files.
        extra: Generic extensibility dict for consumer libraries.
            subagents-pydantic-ai does not read this field — it's carried
            through for consumers like pydantic-deep to use freely.
            Example keys: `memory`, `team`, `cost_budget`.
        max_retries: Number of extra attempts after a transient failure
            (flaky gateway/network). Defaults to `3` — subagents are
            resilient out of the box; retries resume with the full
            message history so partial progress is not lost. Set `0`
            to disable retrying (legacy `agent.run()` opt-out path).
        retry_initial_delay: Seconds before the first retry (default 1.0).
        retry_max_delay: Cap for the backoff delay (default 30.0).
        retry_backoff_multiplier: Delay growth factor per attempt
            (default 2.0).
        retry_jitter: Randomise the backoff delay in `[0, delay]` to
            avoid a thundering herd (default `True`).
        retry_on: Custom predicate `(exc) -> bool` deciding whether an
            exception is transient. Defaults to the built-in classifier
            (`ModelHTTPError` 5xx/429/... and non-HTTP `ModelAPIError`).
        on_failure: Message returned to the parent as an ordinary tool result
            when this subagent fails, instead of raising `ModelRetry`. Use it
            to steer the parent ("summarise from what you already have")
            rather than letting it retry the delegation.
        contain_errors: Whether an unexpected subagent crash is contained.
            Defaults to `True`: the crash becomes a `ModelRetry` for the
            parent, logged with its traceback, so one failed delegation cannot
            abort the whole run. Set `False` to let crashes propagate.
            Control-flow signals (`CallDeferred`, `ApprovalRequired`,
            `Skip*`), `UserError`, and `UsageLimitExceeded` always propagate
            regardless.

    Example with builtin_tools:
        ```python
        SubAgentConfig(
            name="researcher",
            description="Research agent with web search",
            instructions="You research topics using web search.",
            agent_kwargs={"builtin_tools": [BuitinTools.web_search]},
        )
        ```

    Example with per-subagent context:
        ```python
        SubAgentConfig(
            name="coder",
            description="Code writer",
            instructions="You write code following project rules.",
            context_files=["/agents/coder/AGENTS.md", "/CODING_RULES.md"],
        )
        ```
    """

    model: NotRequired[str | Model]
    agent: NotRequired[Any]
    agent_factory: NotRequired[Callable[..., Any]]
    can_ask_questions: NotRequired[bool]
    max_questions: NotRequired[int]
    preferred_mode: NotRequired[Literal["sync", "async", "auto"]]
    typical_complexity: NotRequired[Literal["simple", "moderate", "complex"]]
    typically_needs_context: NotRequired[bool]
    toolsets: NotRequired[Sequence[AbstractToolset[Any]]]
    agent_kwargs: NotRequired[dict[str, Any]]
    context_files: NotRequired[list[str]]
    extra: NotRequired[dict[str, Any]]
    max_retries: NotRequired[int]
    retry_initial_delay: NotRequired[float]
    retry_max_delay: NotRequired[float]
    retry_backoff_multiplier: NotRequired[float]
    retry_jitter: NotRequired[bool]
    retry_on: NotRequired[Callable[[BaseException], bool]]
    on_failure: NotRequired[str]
    contain_errors: NotRequired[bool]


UsageLimitsFactory = Callable[[RunContext[Any], SubAgentConfig], "UsageLimits | None"]
"""Factory function that resolves usage limits for a delegated subagent task.

Called once per delegated task with the parent run context and selected
subagent config. Return `None` to run that task without explicit limits.
"""


EventStreamHandlerFactory = Callable[
    [RunContext[Any], SubAgentConfig, str], "EventStreamHandler[Any] | None"
]
"""Factory resolving the event-stream handler for one delegated subagent task.

Called once per delegation with the parent run context, the selected subagent
config, and the task id. The task id is the argument that makes a fan-out
readable: three specialists streaming into one handler are otherwise
indistinguishable, and the events themselves carry nothing to tell them apart.

Return `None` to run that task without streaming.

Example:
    ```python
    def handler_for(ctx: RunContext[object], config: SubAgentConfig, task_id: str):
        async def on_events(run_ctx: RunContext[object], events: Any) -> None:
            async for event in events:
                await socket.send({"task_id": task_id, "subagent": config["name"], "event": event})

        return on_events
    ```
"""


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
    timestamp: datetime = field(default_factory=utcnow)
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
        chat_trace_id: Chat trace ID for continuing this subagent conversation
        run_id: Pydantic AI run ID for the subagent run
        conversation_id: Pydantic AI conversation ID for the subagent run
        traceparent: W3C traceparent for the subagent run span, if available
        cost: Total subagent model cost calculated from genai-prices
        parent_run_id: `run_id` of the parent run that started this task. Tools
            use it to refuse cross-run access, and the capability uses it to
            cancel a run's tasks when that run ends.
        deferred_requests: The tool calls a `DEFERRED` subagent is suspended on.
            Set only for that status, and the only place they survive: the
            delegation raises so the signal reaches the parent run, and a raise
            carries no result to read them off.
    """

    task_id: str
    subagent_name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    pending_question: str | None = None
    chat_trace_id: str | None = None
    usage: RunUsage | None = None
    """Token usage from the subagent run."""
    message_history: str | None = None
    run_id: str | None = None
    conversation_id: str | None = None
    traceparent: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    model_name: str | None = None
    provider_name: str | None = None
    provider_url: str | None = None
    provider_response_id: str | None = None
    provider_details: dict[str, Any] | None = None
    finish_reason: FinishReason | None = None
    cost: Decimal | None = None
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    retry_count: int = 0
    """Number of transient-failure retries performed for this task."""
    parent_run_id: str | None = None
    deferred_requests: DeferredToolRequests | None = None

    @property
    def is_finished(self) -> bool:
        """Whether the task reached a terminal status."""
        return self.status in TERMINAL_STATUSES

    def finish(
        self,
        status: TaskStatus,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> bool:
        """Record a terminal outcome, first terminal transition winning.

        Idempotence is what makes `hard_cancel` safe. A task that already set
        `COMPLETED` and is running its `finally` block is still not
        `asyncio.Task.done()`, so a cancel arriving in that window used to
        overwrite the real result with `CANCELLED` and move `completed_at`.

        Args:
            status: The terminal status to record.
            result: Result text, for a completed task.
            error: Error text, for a failed or cancelled task.

        Returns:
            Whether this call recorded the outcome.
        """
        if self.is_finished:
            return False
        self.status = status
        self.result = result
        self.error = error
        self.completed_at = utcnow()
        return True


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
    agent: Any = None
    """The agent that runs when this subagent is delegated to.

    Deliberately untyped: consumers such as pydantic-deep supply their own agent
    objects rather than a `pydantic_ai.Agent`, so narrowing this would reject
    valid callers. Only `run`/`iter` are ever called on it.
    """


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
